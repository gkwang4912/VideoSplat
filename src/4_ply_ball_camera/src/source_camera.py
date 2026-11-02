from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import re

import numpy as np


def natural_key(value: str) -> list:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", value)]


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


@dataclass
class SourceCamera:
    image_name: str
    image_path: Path
    camera_id: int
    camera_model: str
    width: int
    height: int
    K: np.ndarray
    fov_x: float
    fov_y: float
    qvec_wxyz: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    world_to_camera: np.ndarray
    camera_to_world: np.ndarray

    @property
    def position(self) -> np.ndarray: return self.camera_to_world[:3, 3]
    @property
    def right(self) -> np.ndarray: return self.camera_to_world[:3, 0]
    @property
    def up(self) -> np.ndarray: return -self.camera_to_world[:3, 1]
    @property
    def forward(self) -> np.ndarray: return self.camera_to_world[:3, 2]


def _read_cameras(path: Path) -> dict[int, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"): continue
        t = line.split(); cid, model, width, height = int(t[0]), t[1], int(t[2]), int(t[3]); p = list(map(float, t[4:]))
        if model == "SIMPLE_PINHOLE": fx = fy = p[0]; cx, cy = p[1:3]
        elif model == "PINHOLE": fx, fy, cx, cy = p[:4]
        else: raise ValueError(f"Unsupported COLMAP camera model {model}; undistort it before rendering")
        result[cid] = {"model": model, "width": width, "height": height,
                       "K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)}
    return result


def _read_images(path: Path) -> list[dict]:
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]
    result = []
    for i in range(0, len(lines), 2):
        t = lines[i].split()
        if len(t) >= 10:
            result.append({"image_id": int(t[0]), "q": np.array(list(map(float, t[1:5]))),
                           "t": np.array(list(map(float, t[5:8]))), "camera_id": int(t[8]), "name": t[9]})
    return sorted(result, key=lambda x: natural_key(x["name"]))


def detect_colmap_first_camera(input_dir: Path) -> tuple[SourceCamera, dict]:
    candidates = []
    for images_txt in input_dir.rglob("images.txt"):
        cameras_txt = images_txt.with_name("cameras.txt"); dataset_root = images_txt.parent.parent.parent
        image_dir = dataset_root / "images"
        if cameras_txt.exists() and image_dir.is_dir(): candidates.append((images_txt, cameras_txt, image_dir))
    if not candidates: raise FileNotFoundError("Could not find a COLMAP text model with a matching images directory")
    candidates.sort(key=lambda x: (len(x[0].parts), natural_key(str(x[0]))))
    images_txt, cameras_txt, image_dir = candidates[0]
    cameras = _read_cameras(cameras_txt); images = _read_images(images_txt)
    if not images: raise ValueError(f"No registered images in {images_txt}")
    row = images[0]; intr = cameras[row["camera_id"]]; image_path = image_dir / row["name"]
    if not image_path.exists(): raise FileNotFoundError(f"First registered image missing: {image_path}")
    R = quat_wxyz_to_matrix(row["q"]); t = row["t"]
    w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = t; c2w = np.linalg.inv(w2c); K = intr["K"]
    source = SourceCamera(row["name"], image_path, row["camera_id"], intr["model"], intr["width"], intr["height"], K,
                          2*math.atan(intr["width"]/(2*K[0,0])), 2*math.atan(intr["height"]/(2*K[1,1])),
                          row["q"], R, t, w2c, c2w)
    digest = hashlib.sha256(images_txt.read_bytes()).hexdigest()
    duplicates = [str(p.resolve()) for p in input_dir.rglob("images.txt") if p != images_txt and hashlib.sha256(p.read_bytes()).hexdigest() == digest]
    info = {"format": "COLMAP text", "images_txt": str(images_txt.resolve()), "cameras_txt": str(cameras_txt.resolve()),
            "image_directory": str(image_dir.resolve()), "registered_frame_count": len(images), "duplicate_identical_models": duplicates}
    return source, info
