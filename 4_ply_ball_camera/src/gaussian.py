from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re

import numpy as np
import torch
from plyfile import PlyData


def _numbered(names: tuple[str, ...], prefix: str) -> list[str]:
    matches = []
    for name in names:
        m = re.fullmatch(re.escape(prefix) + r"(\d+)", name)
        if m:
            matches.append((int(m.group(1)), name))
    return [name for _, name in sorted(matches)]


@dataclass
class GaussianData:
    means: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    opacities: torch.Tensor
    sh: torch.Tensor
    sh_degree: int
    properties: list[str]
    ply_format: str
    vertex_count: int


def apply_ply_splat_config(gaussian: GaussianData, config: dict) -> dict:
    """Apply config-controlled PLY splat state once before any camera is rendered."""
    settings = config.get("ply_splat", {})
    allowed = {
        "splat_scale", "splat_scale_xyz", "opacity_multiplier", "opacity_power",
        "sh_degree", "eps2d", "radius_clip", "point_size_pixels",
    }
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise ValueError("Unknown ply_splat setting(s): " + ", ".join(unknown))

    splat_scale = float(settings.get("splat_scale", 1.0))
    scale_xyz = settings.get("splat_scale_xyz", [1.0, 1.0, 1.0])
    opacity_multiplier = float(settings.get("opacity_multiplier", 1.0))
    opacity_power = float(settings.get("opacity_power", 1.0))
    eps2d = float(settings.get("eps2d", 0.3))
    radius_clip = float(settings.get("radius_clip", 0.0))
    point_size_pixels = float(settings.get("point_size_pixels", 1.0))
    requested_sh_degree = settings.get("sh_degree")

    if splat_scale < 0:
        raise ValueError("ply_splat.splat_scale must be >= 0; use 0 for point mode")
    if not isinstance(scale_xyz, list) or len(scale_xyz) != 3:
        raise ValueError("ply_splat.splat_scale_xyz must contain exactly three values")
    scale_xyz = [float(value) for value in scale_xyz]
    if any(value <= 0 for value in scale_xyz):
        raise ValueError("Every ply_splat.splat_scale_xyz value must be greater than 0")
    if opacity_multiplier < 0 or opacity_power <= 0:
        raise ValueError("ply_splat opacity_multiplier must be >= 0 and opacity_power must be > 0")
    if eps2d < 0 or radius_clip < 0:
        raise ValueError("ply_splat eps2d and radius_clip must be >= 0")
    if point_size_pixels <= 0:
        raise ValueError("ply_splat.point_size_pixels must be greater than 0")
    original_sh_degree = gaussian.sh_degree
    if requested_sh_degree is not None:
        if isinstance(requested_sh_degree, bool) or not isinstance(requested_sh_degree, int):
            raise ValueError("ply_splat.sh_degree must be null or an integer")
        if not 0 <= requested_sh_degree <= original_sh_degree:
            raise ValueError(f"ply_splat.sh_degree must be between 0 and {original_sh_degree}")

    xyz_multiplier = torch.tensor(scale_xyz, dtype=gaussian.scales.dtype, device=gaussian.scales.device)
    point_mode = splat_scale == 0.0
    gaussian.scales = torch.zeros_like(gaussian.scales) if point_mode else gaussian.scales * splat_scale * xyz_multiplier
    gaussian.opacities = torch.clamp(
        gaussian.opacities.pow(opacity_power) * opacity_multiplier, 0.0, 1.0
    )
    if requested_sh_degree is not None:
        gaussian.sh_degree = requested_sh_degree

    return {
        "method": "in_memory_after_ply_load_before_all_renders",
        "mode": "point" if point_mode else "splat",
        "applies_to": ["anchors", "path_rgb", "path_mask_pass", "preview_videos", "All", "combine.mp4"],
        "settings": {
            "splat_scale": splat_scale,
            "splat_scale_xyz": scale_xyz,
            "effective_scale_multiplier_xyz": [splat_scale * value for value in scale_xyz],
            "opacity_multiplier": opacity_multiplier,
            "opacity_power": opacity_power,
            "sh_degree": requested_sh_degree,
            "eps2d": eps2d,
            "radius_clip": radius_clip,
            "point_size_pixels": point_size_pixels,
            "effective_eps2d": (point_size_pixels / 3.0) ** 2 if point_mode else eps2d,
            "effective_rasterize_mode": "classic" if point_mode else config.get("rasterize_mode", "antialiased"),
        },
        "original_sh_degree": original_sh_degree,
        "effective_sh_degree": gaussian.sh_degree,
    }


def load_gaussian_ply(path: Path, device: torch.device) -> GaussianData:
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError("PLY has no vertex element")
    data = ply["vertex"].data
    names = data.dtype.names or ()
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"}
    missing = sorted(required - set(names))
    if missing:
        raise ValueError("Not a supported Gaussian Splatting PLY; missing: " + ", ".join(missing))

    def array(fields: list[str]) -> np.ndarray:
        return np.stack([np.asarray(data[f], dtype=np.float32) for f in fields], axis=1)

    means_np = array(["x", "y", "z"])
    if not np.isfinite(means_np).all():
        raise ValueError("Gaussian positions contain NaN or infinity")
    scales_np = np.exp(array(["scale_0", "scale_1", "scale_2"]))
    opacities_np = 1.0 / (1.0 + np.exp(-np.asarray(data["opacity"], dtype=np.float32)))
    # Original 3DGS PLY convention and gsplat both use quaternion wxyz.
    quats_np = array(["rot_0", "rot_1", "rot_2", "rot_3"])
    quats_np /= np.maximum(np.linalg.norm(quats_np, axis=1, keepdims=True), 1e-12)

    dc_names = _numbered(names, "f_dc_")
    rest_names = _numbered(names, "f_rest_")
    if len(dc_names) != 3 or len(rest_names) % 3:
        raise ValueError(f"Unsupported SH layout: {len(dc_names)} DC and {len(rest_names)} rest fields")
    dc = array(dc_names)[:, None, :]
    if rest_names:
        # 3DGS stores [R coefficients..., G coefficients..., B coefficients...].
        rest = array(rest_names).reshape(len(data), 3, -1).transpose(0, 2, 1)
        sh_np = np.concatenate([dc, rest], axis=1)
    else:
        sh_np = dc
    sh_degree = int(round(math.sqrt(sh_np.shape[1]) - 1))
    if (sh_degree + 1) ** 2 != sh_np.shape[1]:
        raise ValueError(f"SH coefficient count {sh_np.shape[1]} is not a square")

    as_tensor = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(device=device)
    return GaussianData(
        means=as_tensor(means_np), scales=as_tensor(scales_np), quats=as_tensor(quats_np),
        opacities=as_tensor(opacities_np), sh=as_tensor(sh_np), sh_degree=sh_degree,
        properties=list(names), ply_format=str(ply.text and "ascii" or "binary_little_endian"),
        vertex_count=len(data),
    )


def robust_bounds(means: torch.Tensor, percentiles: tuple[float, float]) -> dict:
    xyz = means.detach().cpu().numpy().astype(np.float64)
    low, high = percentiles
    robust_min = np.percentile(xyz, low, axis=0)
    robust_max = np.percentile(xyz, high, axis=0)
    center = (robust_min + robust_max) * 0.5
    size = robust_max - robust_min
    radius = float(np.linalg.norm(size * 0.5))
    distances = np.linalg.norm(xyz - center, axis=1)
    return {
        "center": center.tolist(),
        "bbox_min": robust_min.tolist(), "bbox_max": robust_max.tolist(), "bbox_size": size.tolist(),
        "raw_bbox_min": xyz.min(axis=0).tolist(), "raw_bbox_max": xyz.max(axis=0).tolist(),
        "bounding_sphere_radius": radius,
        "position_distance_percentiles": {str(p): float(np.percentile(distances, p)) for p in (1, 25, 50, 75, 99)},
        "bounds_percentiles": [low, high],
    }
