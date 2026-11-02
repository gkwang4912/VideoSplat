from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .project import ALL_VIDEO_NAMES, RING_NAMES, _json


def _angle_between_rotations(a: np.ndarray, b: np.ndarray) -> float:
    rel = b @ a.T
    return math.degrees(math.acos(float(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))))


def _video_properties(path: Path) -> dict[str, Any] | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required for video validation")
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=codec_name,profile,codec_tag_string,pix_fmt,width,height,r_frame_rate", "-of", "json", str(path)]
    completed = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode:
        return None
    streams = json.loads(completed.stdout).get("streams", [])
    return streams[0] if streams else None


def validate(root: Path, write: bool = True) -> dict[str, Any]:
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8")); out = root / "output"
    failures: list[str] = []; checks: dict[str, Any] = {}
    step = float(cfg["azimuth_step_deg"]); expected = int(round(360 / step)); tol = cfg["validation_tolerances"]
    canonical = json.loads((out / "canonical_coordinate_system.json").read_text(encoding="utf-8"))
    all_cams=[]; max_radius=0.; max_look=0.; max_ortho=0.; max_adj=0.; loops=[]; flips=False; roll_discontinuity=False
    reference_intrinsics=None; file_counts={}
    for ri, name in enumerate(RING_NAMES):
        d=out/"rings"/name; camera_files=sorted((d/"cameras").glob("*.json")); rgb_files=sorted((d/"images").glob("*.png")); mask_files=sorted((d/"mask").glob("*.png")) if cfg["mask_enabled"] else []
        cams=json.loads((d/"cameras.json").read_text(encoding="utf-8")); all_cams.extend(cams)
        file_counts[name]={"cameras":len(cams),"camera_json":len(camera_files),"renders":len(rgb_files),"masks":len(mask_files)}
        if any(x != expected for x in [len(cams),len(camera_files),len(rgb_files)]): failures.append(f"{name}: expected {expected} cameras/json/renders")
        if cfg["mask_enabled"] and len(mask_files)!=expected: failures.append(f"{name}: expected {expected} masks")
        az=np.asarray([c["azimuth_deg"] for c in cams]); expected_az=np.arange(expected)*step
        if not np.allclose(az,expected_az,atol=float(tol["azimuth_deg"]),rtol=0): failures.append(f"{name}: azimuth sequence mismatch")
        circular=np.mod(np.roll(az,-1)-az,360)
        if not np.allclose(circular,step,atol=float(tol["azimuth_deg"]),rtol=0): failures.append(f"{name}: circular azimuth differences are not all {step}")
        elev=np.asarray([c["elevation_deg"] for c in cams])
        if not np.allclose(elev,float(cfg["ring_elevations_deg"][ri]),atol=1e-12,rtol=0): failures.append(f"{name}: elevation mismatch")
        rotations=[]; prev_right=None
        for i,c in enumerate(cams):
            pos=np.asarray(c["camera_position"]); center=np.asarray(c["subject_center"]); f=np.asarray(c["forward"]); r=np.asarray(c["right"]); u=np.asarray(c["up"]); R=np.asarray(c["rotation"])
            max_radius=max(max_radius,abs(np.linalg.norm(pos-center)-float(c["sphere_radius"])))
            target=(center-pos)/np.linalg.norm(center-pos); max_look=max(max_look,float(np.linalg.norm(target-f)))
            max_ortho=max(max_ortho,float(np.max(np.abs(R@R.T-np.eye(3)))))
            if np.linalg.det(R)<0.999999: flips=True
            if prev_right is not None and np.dot(prev_right,r)<0: flips=True
            prev_right=r; rotations.append(R)
            sig=tuple(c[k] if k not in ["K"] else c[k] for k in ["width","height","fx","fy","cx","cy","K","camera_model"])
            packed=json.dumps(sig,sort_keys=True)
            if reference_intrinsics is None: reference_intrinsics=packed
            elif packed!=reference_intrinsics: failures.append("Intrinsics differ between cameras")
            if i < len(rgb_files):
                with Image.open(rgb_files[i]) as im:
                    if im.size!=(c["width"],c["height"]): failures.append(f"{rgb_files[i]} has wrong dimensions")
            if cfg["mask_enabled"] and i < len(mask_files):
                with Image.open(mask_files[i]) as im:
                    vals=np.unique(np.asarray(im));
                    if im.mode!="L" or im.size!=(c["width"],c["height"]) or not set(vals.tolist()).issubset({0,255}): failures.append(f"{mask_files[i]} invalid mode/size/values")
        adjacent=[_angle_between_rotations(rotations[i],rotations[(i+1)%expected]) for i in range(expected)]
        max_adj=max(max_adj,max(adjacent)); loops.append(adjacent[-1])
        if max(adjacent)>float(tol["adjacent_rotation_deg"]): roll_discontinuity=True
        required=[d/"preview.mp4"] + ([d/"preview_mask.mp4",d/"preview_side_by_side.mp4"] if cfg["mask_enabled"] else [])
        for p in required:
            props = _video_properties(p) if p.is_file() else None
            if props is None: failures.append(f"Missing or unreadable video: {p}")
            elif props.get("codec_name") != "h264" or props.get("codec_tag_string") != "avc1" or props.get("pix_fmt") != "yuv420p":
                failures.append(f"Incompatible video format for {p}: {props}")
    if len(all_cams)!=5*expected: failures.append(f"Total camera count {len(all_cams)} != {5*expected}")
    if max_radius>float(tol["radius"]): failures.append(f"Maximum radius error {max_radius} exceeds tolerance")
    if max_look>float(tol["look_at"]): failures.append(f"Maximum look-at error {max_look} exceeds tolerance")
    if max_ortho>float(tol["orthonormal"]): failures.append(f"Maximum orthonormal error {max_ortho} exceeds tolerance")
    if flips: failures.append("Camera flip detected")
    if roll_discontinuity: failures.append("Orientation/roll discontinuity detected")
    for p in [out/"preview"/"ring_layout_3d.png",out/"preview"/"five_ring_front_view.jpg",out/"preview"/"ring_contact_sheet.jpg"]+[out/"All"/n for n in ALL_VIDEO_NAMES]:
        if not p.is_file(): failures.append(f"Missing required output: {p}")
    combined = out / "combine.mp4"
    combined_props = _video_properties(combined) if combined.is_file() else None
    if combined_props is None:
        failures.append(f"Missing or unreadable combined video: {combined}")
    elif combined_props.get("codec_name") != "h264" or combined_props.get("codec_tag_string") != "avc1" or combined_props.get("pix_fmt") != "yuv420p":
        failures.append(f"Incompatible combined video format: {combined_props}")
    checks={"frame_counts":file_counts,"total_camera_count":len(all_cams),"expected_total_camera_count":5*expected,
            "azimuth_step_deg":step,"max_radius_error":max_radius,"max_look_at_error":max_look,"max_orthonormal_error":max_ortho,
            "max_adjacent_camera_rotation_deg":max_adj,"loop_rotation_difference_deg_by_ring":loops,
            "camera_flip_detected":flips,"roll_discontinuity_detected":roll_discontinuity,"intrinsics_identical":not any("Intrinsics differ" in x for x in failures)}
    result={"passed":not failures,"failures":failures,"checks":checks,"tolerances":tol}
    if write: _json(out/"validation.json",result)
    return result


def main() -> int:
    root=Path(__file__).resolve().parents[1]; result=validate(root,write=True); print(json.dumps(result,indent=2,ensure_ascii=False)); return 0 if result["passed"] else 1


if __name__=="__main__": raise SystemExit(main())
