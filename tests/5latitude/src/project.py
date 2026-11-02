from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData


RING_NAMES = ["ring_01_top", "ring_02_upper", "ring_03_middle", "ring_04_lower", "ring_05_bottom"]
ALL_VIDEO_NAMES = ["01_ring_top.mp4", "02_ring_upper.mp4", "03_ring_middle.mp4", "04_ring_lower.mp4", "05_ring_bottom.mp4"]


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _unit(v: np.ndarray, label: str) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError(f"Cannot normalize {label}: norm={n}")
    return v / n


def _qvec_to_rot(q: np.ndarray) -> np.ndarray:
    q = _unit(q.astype(np.float64), "COLMAP quaternion")
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def _sha256(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        with p.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
    return h.hexdigest()


@dataclass
class Inputs:
    ply: Path
    sparse: Path
    images: Path
    masks: Path | None
    duplicate_sparse_models: list[str]


def detect_inputs(input_dir: Path) -> Inputs:
    plys = sorted(input_dir.rglob("*.ply"))
    if len(plys) != 1:
        raise RuntimeError(f"Expected exactly one Gaussian PLY under {input_dir}; found {len(plys)}: {[str(p) for p in plys]}")
    candidates = []
    for camera_file in input_dir.rglob("cameras.txt"):
        d = camera_file.parent
        if (d / "images.txt").is_file() and (d / "points3D.txt").is_file():
            candidates.append(d)
    if not candidates:
        raise RuntimeError("No complete COLMAP text sparse model (cameras.txt/images.txt/points3D.txt) found")
    groups: dict[str, list[Path]] = {}
    for d in candidates:
        digest = _sha256([d / "cameras.txt", d / "images.txt", d / "points3D.txt"])
        groups.setdefault(digest, []).append(d)
    if len(groups) != 1:
        raise RuntimeError(f"Found {len(groups)} distinct COLMAP sparse models; selection would be ambiguous")
    duplicates = sorted(next(iter(groups.values())), key=lambda p: (len(p.parts), str(p)))
    sparse = duplicates[0]

    registered = parse_colmap_images(sparse / "images.txt")
    first_name = registered[0]["name"]
    image_dirs = []
    for d in input_dir.rglob("images"):
        if d.is_dir() and (d / first_name).is_file():
            image_dirs.append(d)
    if not image_dirs:
        raise RuntimeError(f"No registered image directory contains first frame {first_name}")
    image_dirs.sort(key=lambda p: (_tree_distance(sparse, p), len(p.parts), str(p)))
    if len(image_dirs) > 1 and _tree_distance(sparse, image_dirs[0]) == _tree_distance(sparse, image_dirs[1]):
        raise RuntimeError(f"Ambiguous registered image directories: {image_dirs[:2]}")
    images = image_dirs[0]
    missing = [r["name"] for r in registered if not (images / r["name"]).is_file()]
    if missing:
        raise RuntimeError(f"Registered image directory is incomplete; missing {len(missing)} files, first={missing[0]}")
    mask_candidates = []
    sibling = images.parent / "masks"
    if sibling.is_dir():
        mask_candidates.append(sibling)
    for d in input_dir.rglob("masks"):
        if d.is_dir() and d not in mask_candidates and (d / first_name).is_file():
            mask_candidates.append(d)
    mask_candidates = [d for d in mask_candidates if all((d / r["name"]).is_file() for r in registered)]
    mask_candidates.sort(key=lambda p: (_tree_distance(images, p), len(p.parts), str(p)))
    masks = mask_candidates[0] if mask_candidates else None
    if masks is not None:
        with Image.open(masks / first_name) as mask_image, Image.open(images / first_name) as source_image:
            if mask_image.size != source_image.size:
                raise RuntimeError("Source mask dimensions do not match registered images")
    return Inputs(plys[0], sparse, images, masks, [str(p.relative_to(input_dir)) for p in duplicates])


def _tree_distance(a: Path, b: Path) -> int:
    aa, bb = a.resolve().parts, b.resolve().parts
    common = 0
    for x, y in zip(aa, bb):
        if x.lower() != y.lower():
            break
        common += 1
    return len(aa) + len(bb) - 2 * common


def parse_colmap_camera(path: Path) -> dict[str, Any]:
    rows = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]
    if len(rows) != 1:
        raise RuntimeError(f"All output cameras must share one intrinsic camera; cameras.txt has {len(rows)} entries")
    p = rows[0].split()
    cid, model, width, height = int(p[0]), p[1], int(p[2]), int(p[3])
    vals = list(map(float, p[4:]))
    if model == "SIMPLE_PINHOLE" and len(vals) == 3:
        fx = fy = vals[0]; cx, cy = vals[1:]
    elif model == "PINHOLE" and len(vals) == 4:
        fx, fy, cx, cy = vals
    else:
        raise RuntimeError(f"Unsupported camera model {model}; only undistorted SIMPLE_PINHOLE/PINHOLE are accepted")
    return {"camera_id": cid, "camera_model": model, "width": width, "height": height,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            "FOV_x": math.degrees(2 * math.atan(width / (2 * fx))),
            "FOV_y": math.degrees(2 * math.atan(height / (2 * fy)))}


def parse_colmap_images(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    data = [x.strip() for x in lines if x.strip() and not x.startswith("#")]
    if len(data) % 2:
        raise RuntimeError("COLMAP images.txt must contain exactly two data lines per image")
    result = []
    for line in data[::2]:
        p = line.split()
        if len(p) != 10:
            raise RuntimeError(f"Malformed registered image line: {line[:160]}")
        result.append({"image_id": int(p[0]), "qvec_wxyz": list(map(float, p[1:5])),
                       "tvec": list(map(float, p[5:8])), "camera_id": int(p[8]), "name": p[9]})
    result.sort(key=lambda x: x["image_id"])
    if not result:
        raise RuntimeError("COLMAP model contains no registered images")
    return result


def load_gaussians(path: Path, cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise RuntimeError("PLY has no vertex element")
    v = ply["vertex"].data
    names = set(v.dtype.names or [])
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"}
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"Gaussian PLY is missing required properties: {missing}")
    xyz = np.stack([v[x] for x in ["x", "y", "z"]], axis=1).astype(np.float32)
    logits = np.asarray(v["opacity"], dtype=np.float32)
    opacity = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    pcfg = cfg["ply_splat"]
    opacity = np.clip((opacity ** float(pcfg["opacity_power"])) * float(pcfg["opacity_multiplier"]), 0, 1)
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float32))
    scales *= float(pcfg["splat_scale"]) * float(pcfg["point_size_pixels"])
    scales *= np.asarray(pcfg["splat_scale_xyz"], dtype=np.float32)[None]
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float32)
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float32)
    rest_names = sorted([n for n in names if n.startswith("f_rest_")], key=lambda n: int(n.split("_")[-1]))
    if len(rest_names) % 3:
        raise RuntimeError(f"Invalid SH rest coefficient count: {len(rest_names)}")
    bands = 1 + len(rest_names) // 3
    degree = round(math.sqrt(bands) - 1)
    if (degree + 1) ** 2 != bands:
        raise RuntimeError(f"SH coefficient count {bands} is not a square number")
    colors = np.empty((len(v), bands, 3), dtype=np.float32)
    colors[:, 0, :] = dc
    if rest_names:
        rest = np.stack([v[n] for n in rest_names], axis=1).astype(np.float32).reshape(len(v), 3, bands - 1)
        colors[:, 1:, :] = rest.transpose(0, 2, 1)
    active_degree = degree if pcfg["sh_degree"] is None else int(pcfg["sh_degree"])
    if not (0 <= active_degree <= degree):
        raise RuntimeError(f"Configured sh_degree={active_degree} exceeds PLY degree {degree}")
    comments = [str(c) for c in ply.comments]
    return {"means": xyz, "quats": quats, "scales": scales, "opacities": opacity.astype(np.float32), "colors": colors}, {
        "count": int(len(v)), "stored_sh_degree": degree, "active_sh_degree": active_degree, "comments": comments,
        "property_count": len(names), "opacity_min": float(opacity.min()), "opacity_max": float(opacity.max())}


def weighted_median_xyz(xyz: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # Opacity-weighted coordinate median is robust to sparse floaters and deterministic.
    result = []
    w = np.maximum(weights.astype(np.float64), 1e-12)
    for axis in range(3):
        order = np.argsort(xyz[:, axis], kind="stable")
        cw = np.cumsum(w[order])
        result.append(float(xyz[order[np.searchsorted(cw, cw[-1] * 0.5)], axis]))
    return np.asarray(result, dtype=np.float64)


def build_canonical(gauss: dict[str, np.ndarray], registered: list[dict[str, Any]]) -> dict[str, Any]:
    center = weighted_median_xyz(gauss["means"], gauss["opacities"])
    first = registered[0]
    R = _qvec_to_rot(np.asarray(first["qvec_wxyz"]))
    t = np.asarray(first["tvec"], dtype=np.float64)
    camera_center = -R.T @ t
    # PLY declares Y as the vertical axis but does not encode its sign. Resolve that
    # sign against the first registered frame's image-up direction (-camera local Y).
    camera_image_up_world = R.T @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
    up = np.array([0.0, 1.0 if camera_image_up_world[1] >= 0 else -1.0, 0.0], dtype=np.float64)
    front_raw = camera_center - center
    front = _unit(front_raw - np.dot(front_raw, up) * up, "canonical front projection")
    right = _unit(np.cross(up, front), "canonical right")
    front = _unit(np.cross(right, up), "canonical front re-orthogonalization")
    radius = float(np.linalg.norm(camera_center - center))
    if radius <= 0:
        raise RuntimeError("First camera distance is zero; cannot define sphere radius")
    return {"subject_center": center.tolist(), "sphere_radius": radius,
            "canonical_front": front.tolist(), "canonical_right": right.tolist(), "canonical_up": up.tolist(),
            "canonical_back": (-front).tolist(), "canonical_left": (-right).tolist(), "canonical_down": (-up).tolist(),
            "front_reference": {**first, "camera_center": camera_center.tolist()},
            "derivation": {"subject_center": "per-axis opacity-weighted median of Gaussian means",
                           "up": "PLY vertical Y axis with sign selected to agree with first registered frame image-up direction",
                           "front": "first registered camera center projected onto plane perpendicular to canonical up",
                           "right": "cross(canonical_up, canonical_front)",
                           "sphere_radius": "distance(first registered camera center, subject_center)"},
            "coordinate_convention": "right-handed world; OpenCV camera +X right, +Y down, +Z forward"}


def make_camera(canonical: dict[str, Any], intr: dict[str, Any], ring_index: int, ring_name: str,
                elevation: float, frame: int, azimuth: float) -> dict[str, Any]:
    C = np.asarray(canonical["subject_center"]); F = np.asarray(canonical["canonical_front"])
    Rt = np.asarray(canonical["canonical_right"]); U = np.asarray(canonical["canonical_up"])
    th, ph = np.radians(azimuth), np.radians(elevation)
    horizontal = math.cos(th) * F + math.sin(th) * Rt
    radial = _unit(math.cos(ph) * horizontal + math.sin(ph) * U, "radial direction")
    pos = C + canonical["sphere_radius"] * radial
    forward = _unit(C - pos, "camera forward")
    right = _unit(np.cross(forward, U), "camera right")
    down = _unit(np.cross(forward, right), "camera down")
    up = -down
    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ pos
    w2c = np.eye(4); w2c[:3, :3] = rotation; w2c[:3, 3] = translation
    c2w = np.linalg.inv(w2c)
    return {"ring_index": ring_index, "ring_name": ring_name, "frame_index": frame,
            "azimuth_deg": float(azimuth), "elevation_deg": float(elevation),
            "subject_center": C.tolist(), "sphere_radius": float(canonical["sphere_radius"]),
            "camera_position": pos.tolist(), "forward": forward.tolist(), "up": up.tolist(), "right": right.tolist(),
            **{k: intr[k] for k in ["width", "height", "fx", "fy", "cx", "cy", "FOV_x", "FOV_y", "K", "camera_model"]},
            "camera_to_world": c2w.tolist(), "world_to_camera": w2c.tolist(),
            "rotation": rotation.tolist(), "translation": translation.tolist(),
            "camera_convention": "OpenCV/COLMAP: local +X image-right, +Y image-down, +Z forward; image origin top-left",
            "coordinate_convention": "right-handed world coordinates",
            "matrix_format": "row-major 2D arrays; homogeneous p_camera = world_to_camera @ p_world",
            "source_camera_info": canonical["front_reference"]}


class Renderer:
    def __init__(self, gauss: dict[str, np.ndarray], info: dict[str, Any], intr: dict[str, Any], cfg: dict[str, Any]):
        if sys.platform == "win32":
            # This MSVC installation emits UTF-8 while PyTorch 2.11 assumes the OEM codec.
            import torch.utils.cpp_extension as cpp_extension
            cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8",)
        from gsplat import rasterization
        self.rasterization = rasterization
        requested = cfg["render"]["device"]
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("config requests CUDA but torch.cuda.is_available() is false")
        self.device = torch.device(requested)
        self.means = torch.from_numpy(gauss["means"]).to(self.device)
        self.quats = torch.from_numpy(gauss["quats"]).to(self.device)
        self.scales = torch.from_numpy(gauss["scales"]).to(self.device)
        self.opacities = torch.from_numpy(gauss["opacities"]).to(self.device)
        self.colors = torch.from_numpy(gauss["colors"]).to(self.device)
        self.white = torch.ones((len(self.means), 3), dtype=torch.float32, device=self.device)
        self.K = torch.tensor(intr["K"], dtype=torch.float32, device=self.device)[None]
        self.width, self.height = intr["width"], intr["height"]
        self.sh_degree = info["active_sh_degree"]
        self.cfg = cfg

    @torch.inference_mode()
    def render(self, camera: dict[str, Any], mask_enabled: bool) -> tuple[np.ndarray, np.ndarray | None]:
        view = torch.tensor(camera["world_to_camera"], dtype=torch.float32, device=self.device)[None]
        kw = dict(means=self.means, quats=self.quats, scales=self.scales, opacities=self.opacities,
                  viewmats=view, Ks=self.K, width=self.width, height=self.height,
                  near_plane=float(self.cfg["render"]["near_plane"]), far_plane=float(self.cfg["render"]["far_plane"]),
                  radius_clip=float(self.cfg["ply_splat"]["radius_clip"]), eps2d=float(self.cfg["ply_splat"]["eps2d"]),
                  packed=True, camera_model="pinhole")
        bg = torch.tensor(self.cfg["render"]["background_rgb"], dtype=torch.float32, device=self.device)[None]
        rgb, _, _ = self.rasterization(colors=self.colors, sh_degree=self.sh_degree, backgrounds=bg, **kw)
        rgb8 = (rgb[0].clamp(0, 1).mul(255).round().byte().cpu().numpy())
        mask = None
        if mask_enabled:
            magenta = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32, device=self.device)
            mrgb, _, _ = self.rasterization(colors=self.white, sh_degree=None, backgrounds=magenta, **kw)
            m = mrgb[0]
            d_white = torch.linalg.vector_norm(m - torch.tensor([1., 1., 1.], device=self.device), dim=-1)
            d_magenta = torch.linalg.vector_norm(m - torch.tensor([1., 0., 1.], device=self.device), dim=-1)
            mask = torch.where(d_magenta <= d_white, 255, 0).byte().cpu().numpy()
        return rgb8, mask


def _write_video(path: Path, frames: list[Path], fps: int, side_masks: list[Path] | None = None) -> None:
    if not frames:
        raise RuntimeError(f"Cannot write empty video {path}")
    import imageio_ffmpeg
    ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    expected_names = [f"{i:03d}.png" for i in range(len(frames))]
    if [p.name for p in frames] != expected_names:
        raise RuntimeError(f"Video frames must be contiguous 000.png..{len(frames)-1:03d}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frames[0].parent / "%03d.png")
    if side_masks is None:
        cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", pattern,
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)]
    else:
        if len(side_masks) != len(frames) or [p.name for p in side_masks] != expected_names:
            raise RuntimeError("Side-by-side mask sequence does not match RGB frames")
        mask_pattern = str(side_masks[0].parent / "%03d.png")
        cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", pattern,
               "-framerate", str(fps), "-i", mask_pattern,
               "-filter_complex", "[0:v]format=rgb24[left];[1:v]format=rgb24[right];[left][right]hstack=inputs=2[v]",
               "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)]
    completed = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode:
        raise RuntimeError(f"FFmpeg H.264 encoding failed for {path}:\n{completed.stderr[-3000:]}")


def _combine_videos(videos: list[Path], output: Path) -> None:
    import imageio_ffmpeg
    ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    missing = [str(p) for p in videos if not p.is_file()]
    if missing:
        raise RuntimeError(f"Cannot combine missing videos: {missing}")
    concat_list = output.parent / ".combine_inputs.txt"
    lines = ["file '" + p.resolve().as_posix().replace("'", "'\\''") + "'" for p in videos]
    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
               "-c", "copy", "-movflags", "+faststart", str(output)]
        completed = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if completed.returncode:
            raise RuntimeError(f"FFmpeg combine failed for {output}:\n{completed.stderr[-3000:]}")
    finally:
        concat_list.unlink(missing_ok=True)


def _contact_sheet(paths: list[Path], labels: list[str], output: Path, thumb=(240, 460)) -> None:
    if len(paths) != len(labels):
        raise ValueError("Contact sheet path/label count mismatch")
    margin, header = 18, 42
    canvas = Image.new("RGB", (margin + len(paths) * (thumb[0] + margin), thumb[1] + header + 2 * margin), "#181818")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    for i, (path, label) in enumerate(zip(paths, labels)):
        im = Image.open(path).convert("RGB"); im.thumbnail(thumb, Image.Resampling.LANCZOS)
        x = margin + i * (thumb[0] + margin) + (thumb[0] - im.width) // 2
        y = margin + header + (thumb[1] - im.height) // 2
        canvas.paste(im, (x, y)); draw.text((margin + i * (thumb[0] + margin), margin), label, fill="white", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94)


def _layout_plot(canonical: dict[str, Any], cameras_by_ring: list[list[dict[str, Any]]], output: Path) -> None:
    C = np.asarray(canonical["subject_center"]); F = np.asarray(canonical["canonical_front"])
    U = np.asarray(canonical["canonical_up"]); radius = canonical["sphere_radius"]
    fig = plt.figure(figsize=(11, 9), dpi=160); ax = fig.add_subplot(111, projection="3d")
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    x = C[0] + radius*np.cos(u)*np.sin(v); y = C[1] + radius*np.sin(u)*np.sin(v); z = C[2] + radius*np.cos(v)
    ax.plot_wireframe(x, y, z, color="#8ea3b7", alpha=.12, linewidth=.35)
    colors = ["#ff4d6d", "#ff9f1c", "#2ec4b6", "#4d96ff", "#9b5de5"]
    for i, cams in enumerate(cameras_by_ring):
        pts = np.asarray([c["camera_position"] for c in cams] + [cams[0]["camera_position"]])
        ax.plot(pts[:,0], pts[:,1], pts[:,2], color=colors[i], linewidth=2.2, label=f"Ring {i+1}: {cams[0]['elevation_deg']:+g}°")
        p0 = pts[0]; ax.scatter(*p0, color=colors[i], s=35); ax.text(*p0, f"  R{i+1} 0°", color=colors[i])
        p3 = pts[3]; ax.quiver(*p0, *(p3-p0), color=colors[i], arrow_length_ratio=.8, linewidth=1.5)
    ax.scatter(*C, color="black", marker="*", s=110, label="Subject center")
    for label, vec in [("Front",F),("Back",-F),("Up",U),("Down",-U)]:
        p=C+vec*radius*1.12; ax.text(*p,label,weight="bold"); ax.quiver(*C,*(vec*radius),color="#222",arrow_length_ratio=.08)
    ax.set_title("Five spherical latitude camera rings", fontsize=16); ax.set_xlabel("World X"); ax.set_ylabel("World Y"); ax.set_zlabel("World Z")
    ax.set_box_aspect((1,1,1)); ax.legend(loc="upper left"); fig.tight_layout(); fig.savefig(output); plt.close(fig)


def _rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    relative = b @ a.T
    return math.degrees(math.acos(float(np.clip((np.trace(relative)-1)/2, -1, 1))))


def run_project(root: Path) -> dict[str, Any]:
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if len(cfg["ring_elevations_deg"]) != 5 or any(abs(float(x)) >= 90 for x in cfg["ring_elevations_deg"]):
        raise RuntimeError("ring_elevations_deg must contain exactly five values strictly between -90 and +90")
    step = float(cfg["azimuth_step_deg"])
    count_float = 360.0 / step
    if step <= 0 or abs(count_float - round(count_float)) > 1e-10:
        raise RuntimeError("azimuth_step_deg must be positive and divide 360 exactly")
    frame_count = int(round(count_float))
    input_dir, output = root / "input", root / "output"
    if not input_dir.is_dir():
        raise RuntimeError(f"Missing input directory: {input_dir}")
    inputs = detect_inputs(input_dir)
    intr = parse_colmap_camera(inputs.sparse / "cameras.txt")
    registered = parse_colmap_images(inputs.sparse / "images.txt")
    if any(r["camera_id"] != intr["camera_id"] for r in registered):
        raise RuntimeError("Registered images reference more than the single supported camera")
    with Image.open(inputs.images / registered[0]["name"]) as im:
        if im.size != (intr["width"], intr["height"]):
            raise RuntimeError(f"COLMAP intrinsics size does not match registered image: {(intr['width'],intr['height'])} vs {im.size}")
    gauss, ginfo = load_gaussians(inputs.ply, cfg)
    canonical = build_canonical(gauss, registered)

    if output.exists():
        if output.resolve().parent != root.resolve():
            raise RuntimeError(f"Refusing to rebuild unexpected output path: {output}")
        shutil.rmtree(output)
    output.mkdir()
    _json(output / "canonical_coordinate_system.json", canonical)
    _json(output / "splat_configuration.json", {"source_ply": str(inputs.ply), "gaussian": ginfo, "settings": cfg["ply_splat"]})

    cameras_by_ring = []
    all_cameras = []
    for ri, (name, elev) in enumerate(zip(RING_NAMES, cfg["ring_elevations_deg"]), 1):
        cams = [make_camera(canonical, intr, ri, name, float(elev), fi, fi*step) for fi in range(frame_count)]
        cameras_by_ring.append(cams); all_cameras.extend(cams)
        ring_dir = output / "rings" / name
        (ring_dir / "images").mkdir(parents=True); (ring_dir / "cameras").mkdir()
        if cfg["mask_enabled"]: (ring_dir / "mask").mkdir()
        for cam in cams: _json(ring_dir / "cameras" / f"{cam['frame_index']:03d}.json", cam)
        _json(ring_dir / "cameras.json", cams)
    _json(output / "camera_manifest.json", {"camera_count": len(all_cameras), "cameras": all_cameras})
    _json(output / "ring_manifest.json", {"ring_count": 5, "frames_per_ring": frame_count,
        "azimuth_step_deg": step, "rings": [{"ring_index": i+1, "ring_name": RING_NAMES[i], "elevation_deg": float(cfg["ring_elevations_deg"][i]), "frame_count": frame_count} for i in range(5)]})

    renderer = Renderer(gauss, ginfo, intr, cfg)
    print(f"Detected {ginfo['count']:,} Gaussians (SH {ginfo['active_sh_degree']}); rendering {len(all_cameras)} cameras on {renderer.device}...", flush=True)
    for ri, cams in enumerate(cameras_by_ring):
        ring_dir = output / "rings" / RING_NAMES[ri]
        for cam in cams:
            idx = cam["frame_index"]
            rgb, mask = renderer.render(cam, bool(cfg["mask_enabled"]))
            Image.fromarray(rgb, "RGB").save(ring_dir / "images" / f"{idx:03d}.png")
            if mask is not None: Image.fromarray(mask, "L").save(ring_dir / "mask" / f"{idx:03d}.png")
        print(f"Rendered ring {ri+1}/5", flush=True)
    del renderer
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    fps = int(cfg["render"]["video_fps"]); all_dir = output / "All"; all_dir.mkdir()
    for ri, name in enumerate(RING_NAMES):
        ring_dir = output / "rings" / name
        images = sorted((ring_dir / "images").glob("*.png")); masks = sorted((ring_dir / "mask").glob("*.png")) if cfg["mask_enabled"] else []
        _write_video(ring_dir / "preview.mp4", images, fps)
        shutil.copy2(ring_dir / "preview.mp4", all_dir / ALL_VIDEO_NAMES[ri])
        if cfg["mask_enabled"]:
            _write_video(ring_dir / "preview_mask.mp4", masks, fps)
            _write_video(ring_dir / "preview_side_by_side.mp4", images, fps, masks)
    _combine_videos([all_dir / name for name in ALL_VIDEO_NAMES], output / "combine.mp4")
    preview = output / "preview"; preview.mkdir()
    front_paths = [output / "rings" / n / "images" / "000.png" for n in RING_NAMES]
    labels = [f"Ring {i+1}  {float(e):+g} deg" for i,e in enumerate(cfg["ring_elevations_deg"])]
    _contact_sheet(front_paths, labels, preview / "five_ring_front_view.jpg")
    representatives = [output / "rings" / n / "images" / f"{idx:03d}.png" for n in RING_NAMES for idx in [0, 30, 60, 90]]
    rep_labels = [f"R{ri+1} {idx*step:g}°" for ri in range(5) for idx in [0,30,60,90]]
    # 20-image overview in four rows.
    sheets=[]
    for row in range(4):
        row_path=preview/f".row{row}.jpg"; _contact_sheet(representatives[row*5:(row+1)*5], rep_labels[row*5:(row+1)*5], row_path, thumb=(190,360)); sheets.append(Image.open(row_path).copy())
    total=Image.new("RGB",(max(x.width for x in sheets),sum(x.height for x in sheets)),"#181818"); y=0
    for x in sheets: total.paste(x,(0,y)); y+=x.height
    total.save(preview/"ring_contact_sheet.jpg",quality=int(cfg["render"]["jpeg_quality"]))
    for row_path in preview.glob(".row*.jpg"): row_path.unlink()
    _layout_plot(canonical, cameras_by_ring, preview / "ring_layout_3d.png")

    input_manifest = {"input_root": str(input_dir), "gaussian_ply": str(inputs.ply), "selected_sparse_model": str(inputs.sparse),
        "duplicate_identical_sparse_models": inputs.duplicate_sparse_models, "registered_images_directory": str(inputs.images),
        "source_masks_directory": str(inputs.masks) if inputs.masks else None, "registered_image_count": len(registered), "intrinsics": intr,
        "gaussian": ginfo}
    _json(output / "input_manifest.json", input_manifest)
    from .validation import validate
    validation = validate(root, write=True)
    if not validation["passed"]:
        raise RuntimeError(f"Validation failed; inspect {output / 'validation.json'}")
    return {"inputs": input_manifest, "canonical": canonical, "validation": validation}
