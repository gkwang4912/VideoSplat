from pathlib import Path
import json
import numpy as np
from PIL import Image

from sys import path as sys_path
sys_path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.export import probe_video_dimensions


def validate_output(out: Path) -> list[str]:
    validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))
    graph = json.loads((out / "path_graph.json").read_text(encoding="utf-8"))
    errors = []
    if not (out / "splat_configuration.json").is_file(): errors.append("splat_configuration.json missing")
    if not validation.get("ply_splat_configuration", {}).get("passed"): errors.append("PLY splat configuration validation failed")
    if len(graph["nodes"]) != 6: errors.append("path_graph must contain 6 nodes")
    if len(graph["edges"]) != 12: errors.append("path_graph must contain 12 edges")
    for edge in graph["edges"]:
        base = out / edge["directory"]
        if edge["frame_count"] != 91: errors.append(f"{edge['id']}: metadata frame_count is not 91")
        if len(list((base / "images").glob("*.png"))) != 91: errors.append(f"{edge['id']}: image count is not 91")
        if len(list((base / "cameras").glob("*.json"))) != 91: errors.append(f"{edge['id']}: camera count is not 91")
        if len(list((base / "mask").glob("*.png"))) != 91: errors.append(f"{edge['id']}: mask count is not 91")
        for image_path in sorted((base / "images").glob("*.png")):
            mask_path = base / "mask" / image_path.name
            camera_path = base / "cameras" / f"{image_path.stem}.json"
            if not mask_path.exists():
                errors.append(f"{edge['id']}/{image_path.name}: mask missing")
                continue
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if mask.mode != "L": errors.append(f"{edge['id']}/{mask_path.name}: mask mode {mask.mode}, expected L")
                if mask.size != image.size: errors.append(f"{edge['id']}/{mask_path.name}: size mismatch")
                if not set(np.unique(np.asarray(mask)).tolist()).issubset({0,255}): errors.append(f"{edge['id']}/{mask_path.name}: mask is not binary")
            metadata = json.loads(camera_path.read_text(encoding="utf-8"))
            if metadata.get("mask",{}).get("method") != "background_key_mask_pass": errors.append(f"{edge['id']}/{camera_path.name}: background-key mask metadata missing")
    rings = graph.get("ring_previews", [])
    if len(rings) != 3: errors.append("ring_previews must contain 3 rings")
    for ring in rings:
        ring_base = out / ring["directory"]
        if len(list((ring_base / "images").glob("*.png"))) != 361: errors.append(f"{ring['id']}: ring image count is not 361")
        if len(list((ring_base / "mask").glob("*.png"))) != 361: errors.append(f"{ring['id']}: ring mask count is not 361")
        for video in ("preview.mp4","preview_mask.mp4","preview_side_by_side.mp4"):
            if not (ring_base / video).exists(): errors.append(f"{ring['id']}: {video} missing")
        if (ring_base / "preview_side_by_side.mp4").exists():
            width, height = probe_video_dimensions(ring_base / "preview_side_by_side.mp4")
            with Image.open(sorted((ring_base / "images").glob("*.png"))[0]) as first_image:
                expected = (first_image.width * 2, first_image.height)
            if (width,height) != expected: errors.append(f"{ring['id']}: side-by-side dimensions {(width,height)}, expected {expected}")
        for segment in ring.get("segments", []):
            segment_base = out / segment["directory"]
            if len(list((segment_base / "images").glob("*.png"))) != 91: errors.append(f"{segment['id']}: segment image count is not 91")
            if len(list((segment_base / "mask").glob("*.png"))) != 91: errors.append(f"{segment['id']}: segment mask count is not 91")
            for video in ("preview.mp4","preview_mask.mp4","preview_side_by_side.mp4"):
                if not (segment_base / video).exists(): errors.append(f"{segment['id']}: {video} missing")
    if not (out / "combine.mp4").is_file(): errors.append("combine.mp4 missing")
    elif rings and probe_video_dimensions(out / "combine.mp4") != probe_video_dimensions(out / rings[0]["directory"] / "preview.mp4"):
        errors.append("combine.mp4 dimensions differ from ring previews")
    for anchor in ("front", "left", "right", "back", "top", "bottom"):
        if not (out / "anchors" / f"{anchor}.png").exists(): errors.append(f"anchor {anchor}.png missing")
        if not (out / "anchors" / "cameras" / f"{anchor}.json").exists(): errors.append(f"anchor {anchor}.json missing")
        if not (out / "anchors" / "mask" / f"{anchor}.png").exists(): errors.append(f"anchor mask {anchor}.png missing")
    if not validation["passed"]: errors.append("recorded geometric validation failed")
    if not validation["front_reprojection"]["passed"]: errors.append("front reprojection failed")
    if validation["mask_and_videos"].get("over_expansion_detected"): errors.append("background-key mask over-expansion detected")
    comparison = out / validation["mask_and_videos"].get("before_after_comparison",{}).get("path","")
    if not comparison.is_file(): errors.append("mask before/after foot comparison missing")
    return errors


root = Path(__file__).resolve().parents[1]
output_root = root / "output"
if (output_root / "validation.json").is_file():
    output_dirs = [output_root]
else:
    output_dirs = sorted([path for path in output_root.iterdir() if (path / "validation.json").is_file()], key=lambda path: path.name)

all_errors = []
for out in output_dirs:
    for error in validate_output(out):
        all_errors.append(f"{out.relative_to(root)}: {error}")
if all_errors:
    raise SystemExit("VALIDATION FAILED\n" + "\n".join(all_errors))
print(f"VALIDATION PASSED: {len(output_dirs)} dataset(s), masks, cameras, 36 path videos per dataset, All previews, combine.mp4, and Front reprojection")
