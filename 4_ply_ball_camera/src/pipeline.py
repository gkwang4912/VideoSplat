from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import shutil
import time

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont
import torch

from .camera import build_canonical, camera_metadata, continuous_ring_path, normalize
from .export import concat_videos, contact_sheet, make_side_by_side_video, make_video, probe_video_dimensions, write_json
from .gaussian import apply_ply_splat_config, load_gaussian_ply, robust_bounds
from .render import build_background_key_mask, build_repair_aware_mask, render_camera, render_mask_pass, save_camera, save_render
from .source_camera import detect_colmap_first_camera


PATHS = [
    ("01_front_to_left", "front", "left"), ("02_front_to_right", "front", "right"),
    ("03_left_to_back", "left", "back"), ("04_right_to_back", "right", "back"),
    ("05_front_to_top", "front", "top"), ("06_front_to_bottom", "front", "bottom"),
    ("07_top_to_back", "top", "back"), ("08_bottom_to_back", "bottom", "back"),
    ("09_top_to_left", "top", "left"), ("10_top_to_right", "top", "right"),
    ("11_left_to_bottom", "left", "bottom"), ("12_right_to_bottom", "right", "bottom"),
]

COMBINE_RINGS = [
    {
        "id": "horizontal_front_left_back_right",
        "segments": [
            {"path": "01_front_to_left", "reverse": False, "from": "front", "to": "left"},
            {"path": "03_left_to_back", "reverse": False, "from": "left", "to": "back"},
            {"path": "04_right_to_back", "reverse": True, "from": "back", "to": "right"},
            {"path": "02_front_to_right", "reverse": True, "from": "right", "to": "front"},
        ],
    },
    {
        "id": "vertical_front_top_back_bottom",
        "segments": [
            {"path": "05_front_to_top", "reverse": False, "from": "front", "to": "top"},
            {"path": "07_top_to_back", "reverse": False, "from": "top", "to": "back"},
            {"path": "08_bottom_to_back", "reverse": True, "from": "back", "to": "bottom"},
            {"path": "06_front_to_bottom", "reverse": True, "from": "bottom", "to": "front"},
        ],
    },
    {
        "id": "diagonal_top_left_bottom_right",
        "segments": [
            {"path": "09_top_to_left", "reverse": False, "from": "top", "to": "left"},
            {"path": "11_left_to_bottom", "reverse": False, "from": "left", "to": "bottom"},
            {"path": "12_right_to_bottom", "reverse": True, "from": "bottom", "to": "right"},
            {"path": "10_top_to_right", "reverse": True, "from": "right", "to": "top"},
        ],
    },
]

COMBINE_SEQUENCE = [segment for ring in COMBINE_RINGS for segment in ring["segments"]]
OUTPUT_VERSION = "batch_continuous_ring_camera_v1"

CONVENTION = {
    "world": "Right-handed COLMAP/Gaussian world coordinates; Gaussian is never transformed.",
    "camera": "+X right, +Y down, +Z forward (COLMAP/OpenCV). Image origin is top-left.",
    "matrices": "Row-major arrays acting on column vectors: p_camera = W2C @ p_world.",
    "canonical": "Front radial axis comes only from original first camera center; right/up come from its orientation and are orthonormalized.",
}


def find_ply(input_dir: Path) -> Path:
    files = sorted(input_dir.glob("*.ply"))
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one top-level *.ply in {input_dir}, found {len(files)}: " + ", ".join(map(str, files)))
    return files[0]


def discover_input_datasets(input_root: Path) -> list[tuple[str, Path]]:
    """Support the original single input folder and numbered multi-subject folders."""
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory is missing: {input_root}")
    if list(input_root.glob("*.ply")):
        return [("default", input_root)]
    datasets = [(path.name, path) for path in input_root.iterdir() if path.is_dir() and len(list(path.glob("*.ply"))) == 1]
    datasets.sort(key=lambda item: natural_dataset_key(item[0]))
    if not datasets:
        raise RuntimeError(f"No datasets found. Put one *.ply directly in {input_root}, or put each subject in {input_root}/<name>/")
    return datasets


def natural_dataset_key(value: str) -> list:
    import re
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def matrix_angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    relative = a.T @ b
    return math.degrees(math.acos(np.clip((np.trace(relative)-1)/2, -1, 1)))


def dataset_is_complete(output: Path, config: dict) -> bool:
    manifest_path = output / "camera_manifest.json"
    validation_path = output / "validation.json"
    graph_path = output / "path_graph.json"
    if not (manifest_path.is_file() and validation_path.is_file() and graph_path.is_file()):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        manifest.get("output_version") == OUTPUT_VERSION
        and manifest.get("render_settings") == config
        and validation.get("passed")
        and validation.get("front_reprojection", {}).get("passed")
        and graph.get("combined_preview", {}).get("mode") == "ring_first_then_cut"
        and (output / "combine.mp4").is_file()
    )


def make_path_preview_videos(base: Path, fps: int) -> None:
    make_video(base/"images", base/"preview.mp4", fps, digits=3)
    make_video(base/"mask", base/"preview_mask.mp4", fps, digits=3)
    make_side_by_side_video(base/"images", base/"mask", base/"preview_side_by_side.mp4", fps, digits=3)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_ring_frames(output: Path) -> list[dict]:
    rings_metadata = []
    for ring_index, ring in enumerate(COMBINE_RINGS, 1):
        ring_id = f"{ring_index:02d}_{ring['id']}"
        ring_base = output / "rings" / ring_id
        segment_metadata = []
        for segment_index, segment in enumerate(ring["segments"], 1):
            segment_start = (segment_index - 1) * 90
            segment_end = segment_start + 90
            segment_id = f"{segment_index:02d}_{segment['from']}_to_{segment['to']}"
            segment_metadata.append({
                "id": segment_id,
                "source_path": segment["path"],
                "reverse_source_path": False,
                "from": segment["from"],
                "to": segment["to"],
                "start_frame": segment_start,
                "end_frame": segment_end,
                "frame_count": 91,
                "directory": f"rings/{ring_id}/segments/{segment_id}",
            })
        rings_metadata.append({
            "id": ring_id,
            "ring": ring["id"],
            "directory": f"rings/{ring_id}",
            "frame_count": 361,
            "segments": segment_metadata,
        })
    return rings_metadata


def cut_ring_segment_frames(output: Path, ring: dict) -> None:
    ring_base = output / ring["directory"]
    for segment in ring["segments"]:
        for segment_base in (output / segment["directory"], output / "paths" / segment["source_path"]):
            segment_image_dir = segment_base / "images"
            segment_mask_dir = segment_base / "mask"
            segment_camera_dir = segment_base / "cameras"
            reset_directory(segment_image_dir)
            reset_directory(segment_mask_dir)
            reset_directory(segment_camera_dir)
            segment_frames = []
            for target_index, source_index in enumerate(range(segment["start_frame"], segment["end_frame"] + 1)):
                source_stem = f"{source_index:04d}"
                source_name = f"{source_stem}.png"
                target_stem = f"{target_index:03d}"
                target_name = f"{target_stem}.png"
                link_or_copy(ring_base / "images" / source_name, segment_image_dir / target_name)
                link_or_copy(ring_base / "mask" / source_name, segment_mask_dir / target_name)
                camera = json.loads((ring_base / "cameras" / f"{source_stem}.json").read_text(encoding="utf-8"))
                camera["image"] = target_name
                if "mask" in camera:
                    camera["mask"]["path"] = f"mask/{target_name}"
                camera["path"] = f"{segment['from']}_to_{segment['to']}"
                camera["segment_path"] = segment["source_path"]
                camera["start_anchor"] = segment["from"]
                camera["end_anchor"] = segment["to"]
                camera["local_path_angle_degrees"] = float(target_index)
                camera["segment_frame_index"] = target_index
                camera["cut_from_ring"] = ring["id"]
                camera["cut_from_ring_frame"] = source_index
                write_json(segment_camera_dir / f"{target_stem}.json", camera)
                segment_frames.append(camera)
            write_json(segment_base / "cameras.json", {
                "path": segment["source_path"],
                "frame_count": len(segment_frames),
                "cut_from_ring": ring["id"],
                "frames": segment_frames,
            })


def export_ring_videos_parallel(output: Path, rings: list[dict], config: dict) -> None:
    fps = int(config["preview_fps"])
    jobs = max(1, int(config.get("video_jobs", min(4, os.cpu_count() or 1))))
    video_tasks = []
    for ring in rings:
        cut_ring_segment_frames(output, ring)
        ring_base = output / ring["directory"]
        video_tasks.append((ring_base, 4, "ring"))
        for segment in ring["segments"]:
            video_tasks.append((output / segment["directory"], 3, "segment"))
    jobs = min(jobs, len(video_tasks))
    print(f"    ffmpeg video jobs: {jobs}", flush=True)
    if jobs == 1:
        for base, digits, _ in video_tasks:
            make_video(base/"images", base/"preview.mp4", fps, digits=digits)
            make_video(base/"mask", base/"preview_mask.mp4", fps, digits=digits)
            make_side_by_side_video(base/"images", base/"mask", base/"preview_side_by_side.mp4", fps, digits=digits)
        return
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(make_video_bundle, base, fps, digits): (base, kind)
            for base, digits, kind in video_tasks
        }
        completed_count = 0
        for future in as_completed(futures):
            base, kind = futures[future]
            future.result()
            completed_count += 1
            print(f"    videos: {completed_count}/{len(video_tasks)} ({kind}: {base.name})", flush=True)


def make_video_bundle(base: Path, fps: int, digits: int) -> None:
    make_video(base/"images", base/"preview.mp4", fps, digits=digits)
    make_video(base/"mask", base/"preview_mask.mp4", fps, digits=digits)
    make_side_by_side_video(base/"images", base/"mask", base/"preview_side_by_side.mp4", fps, digits=digits)


def render_set(base: Path, cameras: list[dict], gaussian, config: dict, label: str) -> None:
    base.mkdir(parents=True, exist_ok=True)
    for i, camera in enumerate(cameras):
        camera["mask"] = {
            "enabled": bool(config.get("mask_enabled", True)),
            "method": "background_key_mask_pass",
            "background_color": config.get("mask_pass_background_color", [255,0,255]),
            "foreground_color": config.get("mask_pass_foreground_color", [255,255,255]),
            "distance_threshold": float(config.get("mask_bg_distance_threshold", 25.0)),
            "distance_definition": "euclidean_distance(mask_pass_rgb_0_255, background_color) < threshold",
            "close_kernel": int(config.get("mask_pass_close_kernel", 3)),
            "minimum_component_area": int(config.get("mask_pass_min_component_area", 24)),
            "empty_value": 255,
            "occupied_value": 0,
            "path": f"mask/{camera['image']}",
        }
        normal_result = render_camera(gaussian, camera, config)
        mask_rgb = render_mask_pass(gaussian, camera, config)
        mask_stats = save_render(base, camera, normal_result, config, mask_rgb)
        camera["mask"]["statistics"] = mask_stats
        save_camera(base, camera)
        if i == 0 or (i+1) % 15 == 0 or i+1 == len(cameras): print(f"    {label}: {i+1}/{len(cameras)}", flush=True)
    write_json(base / "cameras.json", {"path": label, "frame_count": len(cameras), "frames": cameras})


def validate_dataset(output: Path, anchors_meta: dict, paths: dict, source, basis: dict, config: dict) -> dict:
    center, expected = basis["center"], basis["radius"]; radii = []; radius_errors = []; angle_errors = []
    path_angle_tolerance = float(config.get("path_angle_tolerance_degrees", 1e-5))
    point_mode = float(config.get("ply_splat", {}).get("splat_scale", 1.0)) == 0.0
    orientation_steps = []; look_errors = []; endpoint_position_errors = []; endpoint_intrinsics_errors = []
    endpoint_rotation_deltas = []; applied_roll_corrections = []
    upright_errors = []
    asset_errors = []; checked_masks = 0; mask_values_seen = set(); video_dimensions = {}; mask_difference_by_path = {}
    for folder, frames in paths.items():
        base = output / "paths" / folder
        path_stats = {"previous_alpha_repair_mask_pixels":0, "background_key_mask_pixels":0,
                      "newly_masked_vs_alpha_pixels":0, "newly_unmasked_vs_alpha_pixels":0,
                      "subject_silhouette_pixels":0, "very_strong_coverage_pixels":0,
                      "very_strong_coverage_masked_pixels":0}
        start = normalize(np.array(frames[0]["camera_position"])-center)
        for i, cam in enumerate(frames):
            p = np.array(cam["camera_position"]); actual_radius = float(np.linalg.norm(p-center)); radii.append(actual_radius); radius_errors.append(abs(actual_radius-expected))
            d = normalize(p-center); actual_angle = math.degrees(math.acos(np.clip(np.dot(start,d),-1,1)))
            angle_errors.append(abs(actual_angle-i*float(config["path_degrees_per_frame"])))
            f = np.array(cam["camera_forward"]); look_errors.append(math.degrees(math.acos(np.clip(np.dot(f,normalize(center-p)),-1,1))))
            applied_roll_corrections.append(abs(float(cam.get("applied_roll_correction_degrees", 0.0))))
            stats = cam["mask"]["statistics"]
            for key in path_stats: path_stats[key] += int(stats[key])
            projected_up = basis["up_axis"] - d*np.dot(basis["up_axis"], d)
            if np.linalg.norm(projected_up) > 1e-10:
                upright_errors.append(math.degrees(math.acos(np.clip(np.dot(normalize(projected_up),np.array(cam["camera_up"])),-1,1))))
            image_path = base / "images" / cam["image"]
            mask_path = base / cam["mask"]["path"]
            if not image_path.exists() or not mask_path.exists():
                asset_errors.append(f"{folder}/{cam['image']}: image or mask missing")
            else:
                with Image.open(image_path) as image, Image.open(mask_path) as mask:
                    if mask.mode != "L": asset_errors.append(f"{folder}/{cam['image']}: mask mode is {mask.mode}, expected L")
                    if mask.size != image.size: asset_errors.append(f"{folder}/{cam['image']}: mask/image size mismatch")
                    values = set(np.unique(np.asarray(mask)).tolist()); mask_values_seen.update(values)
                    if not values.issubset({0,255}): asset_errors.append(f"{folder}/{cam['image']}: non-binary mask values {sorted(values)}")
                    checked_masks += 1
        path_stats["newly_masked_fraction_of_silhouette"] = float(path_stats["newly_masked_vs_alpha_pixels"] / max(path_stats["subject_silhouette_pixels"],1))
        path_stats["very_strong_coverage_masked_fraction"] = float(path_stats["very_strong_coverage_masked_pixels"] / max(path_stats["very_strong_coverage_pixels"],1))
        path_stats["over_expansion_check_applicable"] = not point_mode
        path_stats["over_expansion_detected"] = bool(not point_mode and (
            path_stats["newly_masked_fraction_of_silhouette"] > float(config["mask_max_added_fraction_of_silhouette"]) or
            path_stats["very_strong_coverage_masked_fraction"] > float(config["mask_max_strong_coverage_masked_fraction"])))
        mask_difference_by_path[folder] = path_stats
        for a, b in zip(frames, frames[1:]):
            orientation_steps.append(matrix_angle_degrees(np.array(a["camera_to_world"])[:3,:3], np.array(b["camera_to_world"])[:3,:3]))
        for cam, anchor_name in ((frames[0], frames[0]["start_anchor"]), (frames[-1], frames[-1]["end_anchor"])):
            anchor = anchors_meta[anchor_name]
            endpoint_position_errors.append(float(np.max(np.abs(np.array(cam["camera_position"])-np.array(anchor["camera_position"])))))
            endpoint_intrinsics_errors.append(float(np.max(np.abs(np.array(cam["intrinsic_matrix"])-np.array(anchor["intrinsic_matrix"])))))
            endpoint_rotation_deltas.append(matrix_angle_degrees(np.array(cam["camera_to_world"])[:3,:3], np.array(anchor["camera_to_world"])[:3,:3]))
    for ring_index, ring in enumerate(COMBINE_RINGS, 1):
        ring_id = f"{ring_index:02d}_{ring['id']}"
        ring_base = output / "rings" / ring_id
        if len(list((ring_base / "images").glob("*.png"))) != 361:
            asset_errors.append(f"rings/{ring_id}: image count is not 361")
        if len(list((ring_base / "mask").glob("*.png"))) != 361:
            asset_errors.append(f"rings/{ring_id}: mask count is not 361")
        for video_name, expected_size in (("preview.mp4",(source.width,source.height)),
                                          ("preview_mask.mp4",(source.width,source.height)),
                                          ("preview_side_by_side.mp4",(source.width*2,source.height))):
            video_path = ring_base / video_name
            if not video_path.exists():
                asset_errors.append(f"rings/{ring_id}/{video_name}: missing")
            else:
                size = probe_video_dimensions(video_path); video_dimensions[f"rings/{ring_id}/{video_name}"] = list(size)
                if size != expected_size: asset_errors.append(f"rings/{ring_id}/{video_name}: got {size}, expected {expected_size}")
        for segment_index, segment in enumerate(ring["segments"], 1):
            segment_id = f"{segment_index:02d}_{segment['from']}_to_{segment['to']}"
            segment_base = ring_base / "segments" / segment_id
            if len(list((segment_base / "images").glob("*.png"))) != 91:
                asset_errors.append(f"rings/{ring_id}/segments/{segment_id}: image count is not 91")
            if len(list((segment_base / "mask").glob("*.png"))) != 91:
                asset_errors.append(f"rings/{ring_id}/segments/{segment_id}: mask count is not 91")
            for video_name, expected_size in (("preview.mp4",(source.width,source.height)),
                                              ("preview_mask.mp4",(source.width,source.height)),
                                              ("preview_side_by_side.mp4",(source.width*2,source.height))):
                video_path = segment_base / video_name
                if not video_path.exists():
                    asset_errors.append(f"rings/{ring_id}/segments/{segment_id}/{video_name}: missing")
                else:
                    size = probe_video_dimensions(video_path)
                    video_dimensions[f"rings/{ring_id}/segments/{segment_id}/{video_name}"] = list(size)
                    if size != expected_size:
                        asset_errors.append(f"rings/{ring_id}/segments/{segment_id}/{video_name}: got {size}, expected {expected_size}")
    combined_preview = output / "combine.mp4"
    if not combined_preview.exists():
        asset_errors.append("combine.mp4: missing")
    else:
        size = probe_video_dimensions(combined_preview); video_dimensions["combine.mp4"] = list(size)
        if size != (source.width, source.height):
            asset_errors.append(f"combine.mp4: got {size}, expected {(source.width, source.height)}")
    for anchor_name, cam in anchors_meta.items():
        image_path = output / "anchors" / f"{anchor_name}.png"; mask_path = output / "anchors" / cam["mask"]["path"]
        if not image_path.exists() or not mask_path.exists(): asset_errors.append(f"anchor {anchor_name}: image or mask missing")
        else:
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if mask.mode != "L": asset_errors.append(f"anchor {anchor_name}: mask mode is {mask.mode}, expected L")
                if mask.size != image.size: asset_errors.append(f"anchor {anchor_name}: mask/image size mismatch")
                values = set(np.unique(np.asarray(mask)).tolist()); mask_values_seen.update(values)
                if not values.issubset({0,255}): asset_errors.append(f"anchor {anchor_name}: non-binary mask")
                checked_masks += 1
    front = anchors_meta["front"]
    front_pos_error = float(np.max(np.abs(np.array(front["camera_position"])-source.position)))
    front_rotation_error = float(np.max(np.abs(np.array(front["world_to_camera"])[:3,:3]-source.rotation)))
    splat_configuration_path = output / "splat_configuration.json"
    splat_configuration = json.loads(splat_configuration_path.read_text(encoding="utf-8")) if splat_configuration_path.exists() else None
    splat_configuration_passed = bool(
        splat_configuration
        and splat_configuration.get("settings", {}).get("splat_scale") == float(config.get("ply_splat", {}).get("splat_scale", 1.0))
        and splat_configuration.get("settings", {}).get("splat_scale_xyz") == [float(x) for x in config.get("ply_splat", {}).get("splat_scale_xyz", [1,1,1])]
        and splat_configuration.get("settings", {}).get("opacity_multiplier") == float(config.get("ply_splat", {}).get("opacity_multiplier", 1.0))
        and splat_configuration.get("settings", {}).get("opacity_power") == float(config.get("ply_splat", {}).get("opacity_power", 1.0))
    )
    result = {
        "radius": {"expected_radius": expected, "min_radius": min(radii), "max_radius": max(radii),
                   "mean_radius": float(np.mean(radii)),
                   "max_absolute_error": max(radius_errors), "tolerance": config["radius_tolerance"]},
        "path_angles": {"expected_degrees_per_frame": config["path_degrees_per_frame"],
                        "max_angle_error_degrees": max(angle_errors),
                        "tolerance_degrees": path_angle_tolerance},
        "orientation": {"max_consecutive_rotation_degrees": max(orientation_steps), "camera_flip_detected": max(orientation_steps) > 30.0,
                        "max_look_at_error_degrees": max(look_errors),
                        "max_canonical_up_screen_alignment_error_degrees": max(upright_errors),
                        "max_applied_roll_correction_degrees": max(applied_roll_corrections),
                        "orientation_mode": "projected_canonical_up_with_path_specific_poles"},
        "anchor_consistency": {"policy": "shared position and intrinsics; rotation is path-dependent to prevent in-path camera roll",
                               "max_position_error": max(endpoint_position_errors),
                               "max_intrinsics_error": max(endpoint_intrinsics_errors),
                               "max_rotation_difference_from_canonical_anchor_degrees": max(endpoint_rotation_deltas),
                               "passed": max(endpoint_position_errors) <= config["anchor_tolerance"] and max(endpoint_intrinsics_errors) <= config["anchor_tolerance"]},
        "mask_and_videos": {"method": "background_key_mask_pass",
                            "parameters": {"background_color":config["mask_pass_background_color"],
                                           "foreground_color":config["mask_pass_foreground_color"],
                                           "distance_threshold":config["mask_bg_distance_threshold"],
                                           "close_kernel":config["mask_pass_close_kernel"],
                                           "minimum_component_area":config["mask_pass_min_component_area"]},
                            "alpha_vs_background_key_by_path": mask_difference_by_path,
                            "over_expansion_check_applicable": not point_mode,
                            "over_expansion_skip_reason": "point mode intentionally exposes sparse centers" if point_mode else None,
                            "over_expansion_detected": any(x["over_expansion_detected"] for x in mask_difference_by_path.values()),
                            "checked_mask_count": checked_masks, "mask_values_seen": sorted(mask_values_seen),
                            "video_dimensions": video_dimensions, "errors": asset_errors, "passed": not asset_errors},
        "front_reprojection": {"source_image": str(source.image_path.resolve()), "position_error": front_pos_error,
                               "rotation_matrix_error": front_rotation_error,
                               "passed": front_pos_error <= config["anchor_tolerance"] and front_rotation_error <= config["anchor_tolerance"]},
        "ply_splat_configuration": {"path": "splat_configuration.json", "passed": splat_configuration_passed,
                                     "effective": splat_configuration},
    }
    result["passed"] = (result["radius"]["max_absolute_error"] <= config["radius_tolerance"] and
                        result["path_angles"]["max_angle_error_degrees"] <= path_angle_tolerance and
                        not result["orientation"]["camera_flip_detected"] and
                        result["orientation"]["max_applied_roll_correction_degrees"] == 0.0 and
                        result["anchor_consistency"]["passed"] and result["front_reprojection"]["passed"])
    result["passed"] &= result["ply_splat_configuration"]["passed"]
    result["passed"] &= result["mask_and_videos"]["passed"]
    result["passed"] &= not result["mask_and_videos"]["over_expansion_detected"]
    return result


def process_dataset(root: Path, input_dir: Path, output: Path, config: dict, dataset_label: str) -> None:
    if config.get("skip_completed_datasets", True) and dataset_is_complete(output, config):
        print(f"\n=== Dataset {dataset_label}: already complete, skipping ===", flush=True)
        return
    if output.exists():
        shutil.rmtree(output)
    started = time.time(); source_ply = find_ply(input_dir)
    print(f"\n=== Dataset {dataset_label}: {input_dir} -> {output} ===", flush=True)
    print("[1/18] Scanning input...", flush=True)
    print("[2/18] Loading Gaussian PLY...", flush=True)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU is required for gsplat")
    gaussian = load_gaussian_ply(source_ply, torch.device("cuda"))
    splat_configuration = apply_ply_splat_config(gaussian, config)
    write_json(output/"splat_configuration.json", splat_configuration)
    effective_scale = splat_configuration["settings"]["effective_scale_multiplier_xyz"]
    print(f"    PLY mode={splat_configuration['mode']}, scale xyz={effective_scale}, opacity x{splat_configuration['settings']['opacity_multiplier']}, SH={gaussian.sh_degree}")
    print("[3/18] Detecting original camera data...", flush=True)
    source, source_info = detect_colmap_first_camera(input_dir)
    print(f"    COLMAP {source.camera_model}, {source.width}x{source.height}, first={source.image_name}")
    print("[4/18] Matching first frame to first camera...", flush=True)
    print(f"    {source.image_path}")
    print("[5/18] Computing robust subject center...", flush=True)
    bounds = robust_bounds(gaussian.means, tuple(config.get("bounds_percentiles", [1.0,99.0]))); center = np.array(bounds["center"])
    to_center = normalize(center-source.position)
    alignment_error = math.degrees(math.acos(np.clip(np.dot(source.forward,to_center),-1,1)))
    warnings = []
    if alignment_error > 1.0: warnings.append(f"First camera forward differs from robust center by {alignment_error:.6f} degrees; original Front pose is preserved.")
    print(f"    front alignment error={alignment_error:.6f} degrees")
    print("[6/18] Locking first-camera sphere radius...", flush=True)
    print("[7/18] Building canonical coordinate system...", flush=True)
    anchors, basis = build_canonical(center, source); radius = basis["radius"]
    K = source.K; near = max(float(config.get("near_plane",.01)), radius-bounds["bounding_sphere_radius"]*1.5)
    far = radius+bounds["bounding_sphere_radius"]*float(config.get("far_plane_scale",3.0))
    anchors_meta = {name: camera_metadata(f"{name}.png", cam, K, source.width, source.height, source.fov_x, source.fov_y, near, far,
                    anchor=name, source_front_frame=source.image_name if name=="front" else None,
                    sphere_radius=radius, subject_center=center.tolist()) for name, cam in anchors.items()}
    canonical_json = {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in basis.items()}
    canonical_json.update({"source_front_camera": source_info["images_txt"]+" :: "+source.image_name,
                           "front_alignment_error_degrees": alignment_error, "coordinate_convention": CONVENTION,
                           "orthonormality": {"front_dot_right": float(np.dot(basis["front_axis"],basis["right_axis"])),
                           "front_dot_up": float(np.dot(basis["front_axis"],basis["up_axis"])),
                           "right_dot_up": float(np.dot(basis["right_axis"],basis["up_axis"])),
                           "determinant_F_R_U": float(np.linalg.det(np.stack([basis["front_axis"],basis["right_axis"],basis["up_axis"]],axis=1)))}})
    write_json(output/"canonical_coordinate_system.json", canonical_json)
    print("[8/18] Building six fixed anchor cameras...", flush=True)
    print("[9/18] Rendering six anchors...", flush=True)
    render_set(output/"anchors", list(anchors_meta.values()), gaussian, config, "anchors")
    for name in anchors_meta:
        shutil.move(str(output/"anchors"/"images"/f"{name}.png"), str(output/"anchors"/f"{name}.png"))
    (output/"anchors"/"images").rmdir()
    write_json(output/"anchors"/"cameras.json", {"anchors": anchors_meta})
    print("[10/18] Building three continuous 360-degree ring cameras...", flush=True)
    ring_camera_paths = {}
    for ring_index, ring in enumerate(COMBINE_RINGS, 1):
        ring_id = f"{ring_index:02d}_{ring['id']}"
        ring_camera_paths[ring_id] = continuous_ring_path(ring, anchors, basis, float(config["path_degrees_per_frame"]),
            K, source.width, source.height, source.fov_x, source.fov_y, near, far)
    print("[11/18] Rendering 361 frames per ring...", flush=True)
    for ring_id, frames in ring_camera_paths.items(): render_set(output/"rings"/ring_id, frames, gaussian, config, ring_id)
    rings = build_ring_frames(output)
    for ring in rings:
        cut_ring_segment_frames(output, ring)
    paths = {
        segment["source_path"]: json.loads((output / "paths" / segment["source_path"] / "cameras.json").read_text(encoding="utf-8"))["frames"]
        for ring in rings for segment in ring["segments"]
    }
    print("[12/18] Saving camera metadata...", flush=True)
    print("[13/18] Exporting ring-first previews, cut segments, and combine.mp4...", flush=True)
    export_ring_videos_parallel(output, rings, config)
    concat_videos([output / ring["directory"] / "preview.mp4" for ring in rings], output / "combine.mp4")
    print("[14/18] Generating contact sheets...", flush=True)
    preview = output/"preview"; preview.mkdir(parents=True,exist_ok=True)
    order = ["front","left","right","back","top","bottom"]
    contact_sheet([output/"anchors"/f"{x}.png" for x in order],[x.title() for x in order],preview/"anchors_contact_sheet.jpg",columns=3)
    sample_paths=[]; sample_labels=[]
    for folder in paths:
        for degree in (0,30,60,90): sample_paths.append(output/"paths"/folder/"images"/f"{degree:03d}.png"); sample_labels.append(f"{folder} {degree}°")
    contact_sheet(sample_paths,sample_labels,preview/"path_graph_contact_sheet.jpg",columns=4)
    shutil.copy2(output/"anchors"/"front.png", preview/"front_reprojection.png")
    print("[15/18] Validating fixed radius and path angles...", flush=True)
    print("[16/18] Validating shared anchor consistency...", flush=True)
    validation = validate_dataset(output,anchors_meta,paths,source,basis,config)
    print("[17/18] Validating original first-frame Front pose...", flush=True)
    # Prove saved Front JSON alone reproduces the anchor render exactly.
    saved_front = json.loads((output/"anchors"/"cameras"/"front.json").read_text(encoding="utf-8"))
    rgb,a,d = render_camera(gaussian,saved_front,config); front_mask_rgb=render_mask_pass(gaussian,saved_front,config)
    temp=preview/"_front_check"; save_render(temp,saved_front,(rgb,a,d),config,front_mask_rgb)
    repro=Image.open(temp/"images"/"front.png").convert("RGBA"); original=Image.open(output/"anchors"/"front.png").convert("RGBA")
    validation["front_reprojection"]["render_max_pixel_error_8bit"] = int(np.asarray(ImageChops.difference(repro,original)).max())
    validation["front_reprojection"]["passed"] &= validation["front_reprojection"]["render_max_pixel_error_8bit"] == 0
    validation["passed"] &= validation["front_reprojection"]["passed"]
    old_mask, _, _ = build_repair_aware_mask(a,config)
    key_mask, comparison_stats = build_background_key_mask(front_mask_rgb,a,config)
    old_panel = Image.fromarray(old_mask,"L").convert("RGB")
    added = (key_mask==255)&(old_mask==0)
    new_visual = np.repeat(key_mask[...,None],3,axis=2)
    new_visual[added] = np.array([255,0,0],dtype=np.uint8)
    new_panel = Image.fromarray(new_visual,"RGB")
    comparison = Image.new("RGB",(source.width*2,source.height)); comparison.paste(old_panel,(0,0)); comparison.paste(new_panel,(source.width,0))
    draw = ImageDraw.Draw(comparison); font = ImageFont.load_default(size=20)
    draw.text((12,12),"BEFORE: alpha repair-aware",fill=(255,0,0),font=font)
    draw.text((source.width+12,12),"AFTER: background-key (red = newly masked)",fill=(255,0,0),font=font)
    yy,xx=np.where(added & (np.indices(added.shape)[0]>=int(source.height*.55)) &
                                                       (np.indices(added.shape)[1]>=int(source.width*.2)) &
                                                       (np.indices(added.shape)[1]<int(source.width*.8)))
    foot_box = None
    if len(xx):
        pad=12; foot_box=[max(int(xx.min())-pad,0),max(int(yy.min())-pad,0),min(int(xx.max())+pad,source.width-1),min(int(yy.max())+pad,source.height-1)]
        draw.rectangle(tuple(foot_box),outline=(255,0,0),width=4)
        draw.rectangle((foot_box[0]+source.width,foot_box[1],foot_box[2]+source.width,foot_box[3]),outline=(255,0,0),width=4)
    comparison_path=preview/"mask_before_after_foot_comparison.png"; comparison.save(comparison_path)
    validation["mask_and_videos"]["before_after_comparison"]={"path":str(comparison_path.relative_to(output)).replace('\\','/'),
        "source_frame":"paths/01_front_to_left/images/000.png","foot_region_box_xyxy":foot_box,"frame_statistics":comparison_stats}
    shutil.rmtree(temp); write_json(output/"validation.json",validation)
    graph = {"nodes": [{"id":n,"camera":f"anchors/cameras/{n}.json","image":f"anchors/{n}.png","mask":f"anchors/mask/{n}.png"} for n in order],
             "edges": [{"id":folder,"start_anchor":a,"end_anchor":b,"arc_type":"90-degree great-circle",
                        "degrees_per_frame":config["path_degrees_per_frame"],"frame_count":len(paths[folder]),"directory":f"paths/{folder}",
                        "orientation_mode":"projected_canonical_up_with_path_specific_poles","endpoint_rotation_policy":"Top/Bottom rotation is path-dependent; canonical Up stays vertical elsewhere",
                        "assets":{"images":"images/","cameras":"cameras/","masks":"mask/"}}
                       for folder,a,b in PATHS],
             "ring_previews": rings,
             "combined_preview": {"combine":"combine.mp4", "mode":"ring_first_then_cut",
                                  "source_videos":[f"{ring['directory']}/preview.mp4" for ring in rings],
                                  "rings":rings}}
    write_json(output/"path_graph.json",graph)
    source_camera_json = {"image":source.image_name,"image_path":str(source.image_path.resolve()),"camera_model":source.camera_model,
      "width":source.width,"height":source.height,"intrinsic_matrix":K.tolist(),"qvec_wxyz":source.qvec_wxyz.tolist(),
      "rotation_matrix":source.rotation.tolist(),"translation":source.translation.tolist(),"camera_to_world":source.camera_to_world.tolist(),
      "world_to_camera":source.world_to_camera.tolist(),"camera_position":source.position.tolist(),"camera_forward":source.forward.tolist(),
      "camera_up":source.up.tolist(),"camera_right":source.right.tolist()}
    manifest={"output_version":OUTPUT_VERSION,
      "source_ply":str(source_ply.resolve()),"source_camera_data":source_info,"source_first_frame":str(source.image_path.resolve()),
      "source_first_camera":source_camera_json,"gaussian":{"vertex_count":gaussian.vertex_count,"sh_degree":gaussian.sh_degree,
      "format":gaussian.ply_format,"properties":gaussian.properties},"bounds":bounds,"subject_center":center.tolist(),"sphere_radius":radius,
      "ply_splat_configuration":splat_configuration,
      "front_alignment_error_degrees":alignment_error,"warnings":warnings,"canonical_basis":canonical_json,
      "canonical_anchors":anchors_meta,"intrinsics":{"camera_model":source.camera_model,"matrix":K.tolist(),"resolution":[source.width,source.height],
      "fov_x_degrees":math.degrees(source.fov_x),"fov_y_degrees":math.degrees(source.fov_y)},"paths":graph["edges"],
      "mask":{"enabled":config["mask_enabled"],"method":"background_key_mask_pass",
              "background_color":config["mask_pass_background_color"],"foreground_color":config["mask_pass_foreground_color"],
              "distance_definition":"euclidean RGB distance to mask-pass background < threshold",
              "parameters":{"distance_threshold":config["mask_bg_distance_threshold"],"close_kernel":config["mask_pass_close_kernel"],
              "minimum_component_area":config["mask_pass_min_component_area"]},
              "empty_value":255,"occupied_value":0,"format":"8-bit grayscale binary PNG",
              "before_after_comparison":"preview/mask_before_after_foot_comparison.png"},
      "render_backend":f"gsplat {__import__('gsplat').__version__} CUDA","render_settings":config,"coordinate_convention":CONVENTION,"validation":validation}
    write_json(output/"camera_manifest.json",manifest)
    print("[18/18] Final report...", flush=True)
    if not validation["passed"] or not validation["front_reprojection"]["passed"]:
        raise RuntimeError(f"Dataset validation failed; inspect {output / 'validation.json'}")
    print(f"Dataset {dataset_label} done in {(time.time()-started)/60:.1f} min | 12x91={sum(map(len,paths.values()))} path frames | radius={radius:.9f} | PASS")


def main(root: Path) -> None:
    config = json.loads((root/"config.json").read_text(encoding="utf-8"))
    input_root = root/"input"
    output_root = root/"output"
    datasets = discover_input_datasets(input_root)
    if len(datasets) == 1 and datasets[0][1] == input_root:
        process_dataset(root, datasets[0][1], output_root, config, datasets[0][0])
    else:
        print(f"Found {len(datasets)} input datasets: " + ", ".join(name for name, _ in datasets), flush=True)
        for name, input_dir in datasets:
            process_dataset(root, input_dir, output_root/name, config, name)
    print(f"\nAll datasets complete: {len(datasets)}", flush=True)
