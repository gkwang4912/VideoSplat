from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


RUN_LOCK = threading.Lock()


@dataclass(frozen=True)
class PipelineConfig:
    max_frames: int = 192
    mask_threshold: int = 128
    erosion_pixels: int = 1
    minimum_island_pixels: int = 64
    minimum_island_ratio: float = 0.0005
    defringe_pixels: int = 3
    minimum_registration_ratio: float = 0.80
    minimum_camera_arc_degrees: float = 300.0
    maximum_camera_step_degrees: float = 30.0
    colmap_camera_model: str = "SIMPLE_PINHOLE"


def _run(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode:
        tail = (result.stdout + "\n" + result.stderr)[-6000:]
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{tail}")
    return result


def resolve_executable(value: str | Path, candidates: tuple[str, ...] = ()) -> Path:
    path = Path(value).expanduser() if value else Path()
    checks: list[Path] = []
    if value:
        checks.append(path)
        if path.is_dir():
            checks.extend(path / name for name in candidates)
            checks.extend(path / "bin" / name for name in candidates)
    for check in checks:
        if check.is_file():
            return check.resolve()
    for name in candidates:
        found = shutil.which(name)
        if found:
            return Path(found).resolve()
    raise FileNotFoundError(f"Executable not found: {value or ', '.join(candidates)}")


def probe_video(video: Path, ffprobe: Path) -> dict[str, Any]:
    result = _run([
        str(ffprobe), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration", "-of", "json", str(video),
    ])
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    numerator, denominator = (stream.get("avg_frame_rate") or "0/1").split("/")
    fps = float(numerator) / max(float(denominator), 1.0)
    duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0)
    frames = int(stream.get("nb_frames") or round(duration * fps))
    return {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "fps": fps, "duration_seconds": duration, "source_frames": frames,
    }


def extract_frames(video: Path, destination: Path, ffmpeg: Path, info: dict[str, Any], maximum: int) -> list[Path]:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    duration = max(float(info["duration_seconds"]), 0.001)
    target = min(maximum, max(1, int(info["source_frames"])))
    sampling_fps = target / duration
    _run([
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-vf", f"fps={sampling_fps:.9f}", "-frames:v", str(target),
        "-start_number", "1", str(destination / "%04d.png"),
    ])
    frames = sorted(destination.glob("*.png"))
    if not frames:
        raise RuntimeError("FFmpeg produced no frames")
    return frames


def clean_binary_mask(mask: np.ndarray, config: PipelineConfig) -> tuple[np.ndarray, dict[str, int]]:
    binary = (mask >= config.mask_threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([], dtype=np.int32)
    cleaned = np.zeros_like(binary)
    removed_components = 0
    if component_areas.size:
        largest = int(component_areas.max())
        threshold = max(config.minimum_island_pixels, int(largest * config.minimum_island_ratio))
        largest_label = int(np.argmax(component_areas)) + 1
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if label == largest_label or area >= threshold:
                cleaned[labels == label] = 1
            else:
                removed_components += 1
    if config.erosion_pixels > 0:
        size = config.erosion_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        eroded = cv2.erode(cleaned, kernel, iterations=1)
        if np.count_nonzero(eroded) >= np.count_nonzero(cleaned) * 0.90:
            cleaned = eroded
    return cleaned.astype(np.uint8) * 255, {
        "components_before": max(0, count - 1),
        "components_removed": removed_components,
        "foreground_pixels": int(np.count_nonzero(cleaned)),
    }


def defringe_rgb(rgb: np.ndarray, mask: np.ndarray, width: int) -> tuple[np.ndarray, int]:
    if width <= 0:
        result = rgb.copy()
        result[mask == 0] = 0
        return result, 0
    foreground = mask > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    interior = cv2.erode(foreground.astype(np.uint8), kernel, iterations=width).astype(bool)
    edge = foreground & ~interior
    if not np.any(interior) or not np.any(edge):
        result = rgb.copy()
        result[~foreground] = 0
        return result, 0

    # OpenCV labels every zero pixel uniquely with DIST_LABEL_PIXEL. Here the
    # trusted interior is zero, so each edge pixel receives its nearest interior label.
    distance_input = np.where(interior, 0, 255).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        distance_input, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    label_to_y = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    label_to_x = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    ys, xs = np.nonzero(interior)
    source_labels = labels[ys, xs]
    label_to_y[source_labels] = ys
    label_to_x[source_labels] = xs
    ey, ex = np.nonzero(edge)
    nearest = labels[ey, ex]
    result = rgb.copy()
    result[ey, ex] = rgb[label_to_y[nearest], label_to_x[nearest]]
    result[~foreground] = 0
    return result, int(edge.sum())


def build_rgba_images(
    frames: list[Path], masks_dir: Path, output_dir: Path, config: PipelineConfig
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    removed_components = changed_pixels = foreground_pixels = 0
    for frame in frames:
        rgb = np.asarray(Image.open(frame).convert("RGB"), dtype=np.uint8)
        raw_mask = np.asarray(Image.open(masks_dir / frame.name).convert("L"), dtype=np.uint8)
        mask, stats = clean_binary_mask(raw_mask, config)
        cleaned_rgb, changed = defringe_rgb(rgb, mask, config.defringe_pixels)
        rgba = np.dstack((cleaned_rgb, mask))
        Image.fromarray(rgba, "RGBA").save(output_dir / frame.name)
        removed_components += stats["components_removed"]
        foreground_pixels += stats["foreground_pixels"]
        changed_pixels += changed
    return {
        "components_removed": removed_components,
        "foreground_pixels": foreground_pixels,
        "defringe_changed_pixels": changed_pixels,
    }


def extract_mask_frames(
    mask_video: Path,
    destination: Path,
    ffmpeg: Path,
    ffprobe: Path,
    expected_source_count: int,
    output_count: int,
    expected_size: tuple[int, int],
) -> list[Path]:
    mask_info = probe_video(mask_video, ffprobe)
    if int(mask_info["source_frames"]) != expected_source_count:
        raise RuntimeError(
            f"Frame count mismatch: source has {expected_source_count}, mask has {mask_info['source_frames']}"
        )
    actual_size = (int(mask_info["width"]), int(mask_info["height"]))
    if actual_size != expected_size:
        raise RuntimeError(f"Size mismatch: source is {expected_size}, mask is {actual_size}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    sampling_fps = output_count / max(float(mask_info["duration_seconds"]), 0.001)
    _run([
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(mask_video),
        "-vf", f"fps={sampling_fps:.9f}", "-frames:v", str(output_count), "-start_number", "1",
        str(destination / "%04d.png"),
    ])
    extracted = sorted(destination.glob("*.png"))
    if len(extracted) != output_count:
        raise RuntimeError(f"Mask extraction produced {len(extracted)} frames; expected {output_count}")
    for path in extracted:
        with Image.open(path) as source:
            gray = np.asarray(source.convert("L"), dtype=np.uint8)
        Image.fromarray(gray, "L").save(path)
    return extracted


def _colmap_environment(colmap: Path) -> dict[str, str]:
    env = os.environ.copy()
    bin_dir = str(colmap.parent)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    return env


def _colmap_binary_count(path: Path) -> int:
    """Read the uint64 item count stored at the start of a COLMAP .bin file."""
    with path.open("rb") as handle:
        header = handle.read(8)
    if len(header) != 8:
        return 0
    return int(struct.unpack("<Q", header)[0])


def run_colmap(
    frames: list[Path], masks: Path, workspace: Path, colmap: Path, config: PipelineConfig
) -> tuple[Path, dict[str, Any]]:
    if workspace.exists():
        shutil.rmtree(workspace)
    images_dir = workspace / "images"
    mask_dir = workspace / "masks"
    sparse_bin = workspace / "sparse_bin"
    sparse_txt = workspace / "sparse" / "0"
    for path in (images_dir, mask_dir, sparse_bin, sparse_txt):
        path.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        shutil.copy2(frame, images_dir / frame.name)
        # COLMAP appends the mask extension to the complete image filename.
        shutil.copy2(masks / frame.name, mask_dir / f"{frame.name}.png")
    database = workspace / "database.db"
    env = _colmap_environment(colmap)
    base = str(colmap)
    _run([
        base, "feature_extractor", "--database_path", str(database), "--image_path", str(images_dir),
        "--ImageReader.mask_path", str(mask_dir), "--ImageReader.camera_model", config.colmap_camera_model,
        "--ImageReader.single_camera", "1", "--SiftExtraction.max_num_features", "16384",
        "--SiftExtraction.peak_threshold", "0.004", "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.gpu_index", "0",
    ], env=env)
    _run([
        base, "sequential_matcher", "--database_path", str(database),
        "--FeatureMatching.use_gpu", "1", "--FeatureMatching.gpu_index", "0",
        "--SequentialMatching.overlap", "20", "--SequentialMatching.quadratic_overlap", "1",
        "--SiftMatching.max_ratio", "0.8", "--SiftMatching.max_distance", "0.7",
    ], env=env)
    _run([
        base, "mapper", "--database_path", str(database), "--image_path", str(images_dir),
        "--output_path", str(sparse_bin), "--Mapper.min_model_size", "2",
        "--Mapper.min_num_matches", "10", "--Mapper.init_min_tri_angle", "4.0",
        "--Mapper.abs_pose_min_num_inliers", "30", "--Mapper.abs_pose_min_inlier_ratio", "0.25",
        "--Mapper.filter_max_reproj_error", "4.0", "--Mapper.tri_min_angle", "1.5",
        "--Mapper.random_seed", "0", "--Mapper.ba_use_gpu", "1", "--Mapper.ba_gpu_index", "0",
    ], env=env)
    models = sorted(path for path in sparse_bin.iterdir() if path.is_dir())
    if not models:
        raise RuntimeError("COLMAP mapper produced no sparse model")
    candidates = []
    for model in models:
        images_bin = model / "images.bin"
        points_bin = model / "points3D.bin"
        candidates.append({
            "model": model.name,
            "path": str(model),
            "registered_images": _colmap_binary_count(images_bin) if images_bin.is_file() else 0,
            "points3d": _colmap_binary_count(points_bin) if points_bin.is_file() else 0,
        })
    selected = max(candidates, key=lambda item: (item["registered_images"], item["points3d"]))
    selected_model = sparse_bin / selected["model"]
    _run([
        base, "model_converter", "--input_path", str(selected_model),
        "--output_path", str(sparse_txt), "--output_type", "TXT",
    ], env=env)
    selection = {
        "strategy": "maximum registered images, then maximum 3D points",
        "selected_model": selected["model"],
        "selected_registered_images": selected["registered_images"],
        "selected_points3d": selected["points3d"],
        "candidates": candidates,
    }
    (workspace / "model_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return sparse_txt, selection


def _quaternion_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)],
    ], dtype=np.float64)


def reconstruction_metrics(sparse: Path, selected_count: int) -> dict[str, Any]:
    image_lines = [line.strip() for line in (sparse / "images.txt").read_text("utf-8").splitlines()
                   if line.strip() and not line.startswith("#")]
    poses: list[tuple[str, np.ndarray]] = []
    for line in image_lines[::2]:
        fields = line.split()
        if len(fields) < 10:
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, fields[1:8])
        rotation = _quaternion_rotation(qw, qx, qy, qz)
        poses.append((fields[9], -(rotation.T @ np.array([tx, ty, tz]))))
    poses.sort(key=lambda item: item[0])
    names = [item[0] for item in poses]
    cameras = [item[1] for item in poses]
    registered = len(cameras)
    points = sum(1 for line in (sparse / "points3D.txt").read_text("utf-8").splitlines()
                 if line.strip() and not line.startswith("#"))
    arc = 0.0
    maximum_step = 360.0
    if registered >= 3:
        xyz = np.stack(cameras)
        centered = xyz - np.median(xyz, axis=0)
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        plane = centered @ axes[:2].T
        angles = np.unwrap(np.arctan2(plane[:, 1], plane[:, 0]))
        ordered = np.degrees(angles)
        steps = np.abs(np.diff(ordered))
        maximum_step = float(steps.max(initial=0.0))
        normalized = np.sort(np.mod(ordered, 360.0))
        circular_gaps = np.diff(np.r_[normalized, normalized[0] + 360.0])
        arc = float(360.0 - circular_gaps.max())
    return {
        "registered_images": registered,
        "selected_images": selected_count,
        "registration_ratio": registered / max(selected_count, 1),
        "points3d": points,
        "camera_arc_degrees": arc,
        "maximum_camera_step_degrees": maximum_step,
        "registered_names": names,
    }


def validate_dataset(dataset: Path, expected_names: list[str]) -> dict[str, Any]:
    images = sorted((dataset / "images").glob("*.png"))
    names = [path.name for path in images]
    modes: dict[str, int] = {}
    nonbinary = hidden_rgb = 0
    sizes: set[tuple[int, int]] = set()
    for path in images:
        with Image.open(path) as source:
            source_mode = source.mode
            rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8)
        modes[source_mode] = modes.get(source_mode, 0) + 1
        sizes.add((rgba.shape[1], rgba.shape[0]))
        alpha = rgba[..., 3]
        nonbinary += int(np.count_nonzero((alpha != 0) & (alpha != 255)))
        hidden_rgb += int(np.count_nonzero(np.any(rgba[..., :3] != 0, axis=2) & (alpha == 0)))
    sparse = dataset / "sparse" / "0"
    sparse_files = [name for name in ("cameras.txt", "images.txt", "points3D.txt") if (sparse / name).is_file()]
    ok = names == expected_names and modes == {"RGBA": len(images)} and nonbinary == 0 and hidden_rgb == 0 and len(sparse_files) == 3
    return {
        "status": "PASS" if ok else "FAIL", "image_count": len(images), "filenames_match": names == expected_names,
        "modes": modes, "sizes": sorted(sizes), "nonbinary_alpha_pixels": nonbinary,
        "hidden_rgb_pixels": hidden_rgb, "sparse_files": sparse_files,
    }


def make_contact_sheet(dataset: Path, maximum: int = 20) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted((dataset / "images").glob("*.png"))
    if not paths:
        raise RuntimeError(f"No images in {dataset}")
    indices = np.linspace(0, len(paths) - 1, min(maximum, len(paths))).round().astype(int)
    tiles: list[Image.Image] = []
    for index in sorted(set(indices)):
        path = paths[index]
        rgba = Image.open(path).convert("RGBA")
        board = Image.new("RGB", rgba.size, (56, 56, 56))
        board.paste(rgba, mask=rgba.getchannel("A"))
        board.thumbnail((220, 300), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (220, 325), "white")
        tile.paste(board, ((220 - board.width) // 2, 0))
        ImageDraw.Draw(tile).text((6, 305), path.name, fill="black")
        tiles.append(tile)
    columns = 5
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * 220, rows * 325), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 220, (index // columns) * 325))
    representative = np.asarray(Image.open(paths[len(paths) // 2]).convert("RGBA"), dtype=np.uint8)
    return np.asarray(sheet), representative[..., 3]


def run_pipeline(
    video: Path,
    mask_video: Path,
    output_root: Path,
    *,
    run_name: str | None = None,
    colmap_path: str | Path = "",
    ffmpeg_path: str | Path = "",
    ffprobe_path: str | Path = "",
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    config = config or PipelineConfig()
    video = video.resolve()
    mask_video = mask_video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if not mask_video.is_file():
        raise FileNotFoundError(mask_video)
    ffmpeg = resolve_executable(ffmpeg_path, ("ffmpeg.exe", "ffmpeg"))
    ffprobe = resolve_executable(ffprobe_path, ("ffprobe.exe", "ffprobe"))
    colmap = resolve_executable(colmap_path, ("colmap.exe", "colmap"))
    requested_name = run_name or video.stem
    safe_name = "".join(character if character.isalnum() or character in "-_." else "_"
                        for character in requested_name).strip("._")
    if not safe_name:
        raise ValueError(f"Invalid run name: {requested_name!r}")
    run_root = output_root.resolve() / safe_name
    work = run_root / "work"
    dataset = run_root / "brush_dataset_rgba_clean"
    with RUN_LOCK:
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        info = probe_video(video, ffprobe)
        frames = extract_frames(video, work / "frames", ffmpeg, info, config.max_frames)
        masks = work / "masks"
        extract_mask_frames(
            mask_video, masks, ffmpeg, ffprobe, int(info["source_frames"]), len(frames),
            (int(info["width"]), int(info["height"])),
        )
        sparse_source, model_selection = run_colmap(frames, masks, work / "colmap", colmap, config)
        if dataset.exists():
            shutil.rmtree(dataset)
        images_dir = dataset / "images"
        clean_stats = build_rgba_images(frames, masks, images_dir, config)
        shutil.copytree(sparse_source.parent, dataset / "sparse")
        geometry = reconstruction_metrics(dataset / "sparse" / "0", len(frames))
        if geometry["registration_ratio"] < config.minimum_registration_ratio:
            raise RuntimeError(f"Registration gate failed: {geometry['registration_ratio']:.1%}")
        if geometry["camera_arc_degrees"] < config.minimum_camera_arc_degrees:
            raise RuntimeError(f"Camera arc gate failed: {geometry['camera_arc_degrees']:.2f} degrees")
        if geometry["maximum_camera_step_degrees"] > config.maximum_camera_step_degrees:
            raise RuntimeError(
                f"Camera continuity gate failed: {geometry['maximum_camera_step_degrees']:.2f} degrees"
            )
        validation = validate_dataset(dataset, [path.name for path in frames])
        report = {
            "status": validation["status"], "dataset_dir": str(dataset), "video": str(video),
            "mask_video": str(mask_video),
            "run_name": safe_name,
            "video_info": info, "config": asdict(config), "mask_cleanup": clean_stats,
            "colmap_model_selection": model_selection,
            "geometry": geometry, "validation": validation,
        }
        (run_root / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
        if report["status"] != "PASS":
            raise RuntimeError(f"Dataset validation failed: {validation}")
        return report
