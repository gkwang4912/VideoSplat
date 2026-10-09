from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from PIL import Image
from scipy import ndimage
import torch


def _load_cached_gsplat_extension() -> None:
    """Load an already-built Windows gsplat extension before gsplat attempts JIT compilation."""
    import importlib.util
    import sys
    import gsplat
    if "gsplat.csrc" in sys.modules:
        return
    tag = f"py{sys.version_info.major}{sys.version_info.minor}_cu{str(torch.version.cuda).replace('.', '')}"
    roots = [Path.home() / "AppData/Local/torch_extensions/torch_extensions/Cache",
             Path.home() / ".cache/torch_extensions"]
    candidates = []
    for root in roots:
        candidates.extend(root.glob(f"{tag}/gsplat_cuda/gsplat_cuda.pyd"))
    if candidates:
        spec = importlib.util.spec_from_file_location("gsplat_cuda", candidates[0])
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        sys.modules["gsplat.csrc"] = module
        setattr(gsplat, "csrc", module)


_load_cached_gsplat_extension()
from gsplat import rasterization

from .gaussian import GaussianData


def _effective_splat_render_settings(config: dict) -> tuple[float, float, str]:
    splat = config.get("ply_splat", {})
    point_mode = float(splat.get("splat_scale", 1.0)) == 0.0
    if point_mode:
        point_size = float(splat.get("point_size_pixels", 1.0))
        # gsplat's projected radius is approximately 3 sigma. Zero world-space
        # covariance plus this eps2d produces a small screen-space point sprite.
        eps2d = (point_size / 3.0) ** 2
        rasterize_mode = "classic"
    else:
        eps2d = float(splat.get("eps2d", 0.3))
        rasterize_mode = config.get("rasterize_mode", "antialiased")
    return eps2d, float(splat.get("radius_clip", 0.0)), rasterize_mode


def build_repair_aware_mask(alpha: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    hard_threshold = float(config.get("mask_alpha_threshold", 0.01))
    weak_threshold = float(config.get("mask_weak_alpha_threshold", 0.05))
    support_threshold = float(config.get("mask_local_mean_threshold", 0.08))
    blur_kernel = int(config.get("mask_box_blur_kernel", 5))
    min_component = int(config.get("mask_min_occupied_component_pixels", 24))
    silhouette_threshold = float(config.get("mask_silhouette_alpha_threshold", 0.01))
    closing_kernel = int(config.get("mask_closing_kernel", 5))
    if not (0 <= hard_threshold <= weak_threshold <= 1 and 0 <= support_threshold <= 1):
        raise ValueError("mask alpha/support thresholds must be within [0,1], with hard <= weak")
    if blur_kernel < 1 or closing_kernel < 1 or blur_kernel % 2 == 0 or closing_kernel % 2 == 0:
        raise ValueError("mask blur and closing kernels must be positive odd integers")

    empty_hard = alpha < hard_threshold
    weak_coverage = alpha < weak_threshold
    local_mean = ndimage.uniform_filter(alpha.astype(np.float32), size=blur_kernel, mode="constant", cval=0.0)
    low_support = local_mean < support_threshold

    connectivity = np.ones((3, 3), dtype=bool)
    occupied_strong = alpha >= weak_threshold
    occupied_labels, occupied_count = ndimage.label(occupied_strong, structure=connectivity)
    occupied_sizes = np.bincount(occupied_labels.ravel())
    occupied_keep = occupied_sizes >= min_component
    if occupied_keep.size: occupied_keep[0] = False
    occupied_clean = occupied_keep[occupied_labels] if occupied_count else np.zeros_like(occupied_strong)

    subject_loose = alpha >= silhouette_threshold
    subject_labels, subject_count = ndimage.label(subject_loose, structure=connectivity)
    if subject_count:
        subject_sizes = np.bincount(subject_labels.ravel()); subject_sizes[0] = 0
        subject_silhouette = subject_labels == int(np.argmax(subject_sizes))
        subject_silhouette = ndimage.binary_closing(subject_silhouette,
                                                     structure=np.ones((closing_kernel, closing_kernel), dtype=bool))
    else:
        subject_silhouette = np.zeros_like(subject_loose)

    repair_inside = subject_silhouette & ~occupied_clean
    final_mask_bool = empty_hard | repair_inside | (weak_coverage & low_support)
    old_mask = empty_hard.astype(np.uint8) * 255
    final_mask = final_mask_bool.astype(np.uint8) * 255
    added = final_mask_bool & ~empty_hard
    strong_masked = final_mask_bool & occupied_strong
    stats = {
        "total_pixels": int(alpha.size),
        "old_empty_mask_pixels": int(empty_hard.sum()),
        "repair_aware_mask_pixels": int(final_mask_bool.sum()),
        "newly_masked_pixels": int(added.sum()),
        "subject_silhouette_pixels": int(subject_silhouette.sum()),
        "newly_masked_fraction_of_silhouette": float(added.sum() / max(int(subject_silhouette.sum()), 1)),
        "strong_coverage_pixels": int(occupied_strong.sum()),
        "strong_coverage_masked_pixels": int(strong_masked.sum()),
        "strong_coverage_masked_fraction": float(strong_masked.sum() / max(int(occupied_strong.sum()), 1)),
        "removed_small_occupied_pixels": int((occupied_strong & ~occupied_clean).sum()),
    }
    return final_mask, old_mask, stats


def build_background_key_mask(mask_pass_rgb: np.ndarray, alpha: np.ndarray, config: dict) -> tuple[np.ndarray, dict]:
    background = np.asarray(config.get("mask_pass_background_color", [255, 0, 255]), dtype=np.float32)
    threshold = float(config.get("mask_bg_distance_threshold", 25.0))
    close_kernel = int(config.get("mask_pass_close_kernel", 3))
    min_area = int(config.get("mask_pass_min_component_area", 24))
    if background.shape != (3,) or np.any(background < 0) or np.any(background > 255):
        raise ValueError("mask_pass_background_color must contain three values in [0,255]")
    if threshold < 0 or close_kernel < 1 or close_kernel % 2 == 0 or min_area < 1:
        raise ValueError("mask distance threshold/min area must be positive and close kernel must be a positive odd integer")
    rgb255 = np.clip(mask_pass_rgb, 0.0, 1.0).astype(np.float32) * 255.0
    dist_to_background = np.linalg.norm(rgb255 - background[None, None, :], axis=-1)
    raw_mask = dist_to_background < threshold
    closed_mask = ndimage.binary_closing(raw_mask, structure=np.ones((close_kernel, close_kernel), dtype=bool), border_value=1)

    connectivity = np.ones((3, 3), dtype=bool)
    white_labels, white_count = ndimage.label(closed_mask, structure=connectivity)
    if white_count:
        white_sizes = np.bincount(white_labels.ravel()); keep_white = white_sizes >= min_area; keep_white[0] = False
        cleaned = keep_white[white_labels]
    else:
        cleaned = np.zeros_like(closed_mask)
    black_labels, black_count = ndimage.label(~cleaned, structure=connectivity)
    removed_black_island_pixels = 0
    if black_count:
        black_sizes = np.bincount(black_labels.ravel()); small_black = (black_sizes < min_area); small_black[0] = False
        black_islands = small_black[black_labels]; removed_black_island_pixels = int(black_islands.sum()); cleaned[black_islands] = True

    previous_alpha_mask, _, previous_stats = build_repair_aware_mask(alpha, config)
    previous_bool = previous_alpha_mask == 255
    newly_masked = cleaned & ~previous_bool
    newly_unmasked = previous_bool & ~cleaned
    strong_threshold = float(config.get("mask_overexpansion_strong_alpha_threshold", 0.5))
    very_strong = alpha >= strong_threshold
    stats = {
        "total_pixels": int(alpha.size),
        "previous_alpha_repair_mask_pixels": int(previous_bool.sum()),
        "background_key_mask_pixels": int(cleaned.sum()),
        "newly_masked_vs_alpha_pixels": int(newly_masked.sum()),
        "newly_unmasked_vs_alpha_pixels": int(newly_unmasked.sum()),
        "raw_background_key_pixels": int(raw_mask.sum()),
        "post_close_pixels": int(closed_mask.sum()),
        "removed_white_noise_pixels": int((closed_mask & ~cleaned).sum()),
        "removed_black_island_pixels": removed_black_island_pixels,
        "subject_silhouette_pixels": int(previous_stats["subject_silhouette_pixels"]),
        "newly_masked_fraction_of_silhouette": float(newly_masked.sum() / max(previous_stats["subject_silhouette_pixels"], 1)),
        "very_strong_coverage_pixels": int(very_strong.sum()),
        "very_strong_coverage_masked_pixels": int((cleaned & very_strong).sum()),
        "very_strong_coverage_masked_fraction": float((cleaned & very_strong).sum() / max(int(very_strong.sum()), 1)),
        "distance_min": float(dist_to_background.min()),
        "distance_max": float(dist_to_background.max()),
    }
    return cleaned.astype(np.uint8) * 255, stats


def render_camera(gaussians: GaussianData, camera: dict, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    device = gaussians.means.device
    view = torch.tensor(camera["world_to_camera"], dtype=torch.float32, device=device)[None]
    K = torch.tensor(camera["intrinsic_matrix"], dtype=torch.float32, device=device)[None]
    background = config.get("background", "transparent")
    if background in ("transparent", "black"):
        bg = [0.0, 0.0, 0.0]
    elif background == "white":
        bg = [1.0, 1.0, 1.0]
    elif isinstance(background, list) and len(background) == 3:
        bg = [float(x) for x in background]
    else:
        raise ValueError("background must be transparent, black, white, or [r,g,b]")
    render_mode = "RGB+ED" if config.get("render_depth", False) else "RGB"
    eps2d, radius_clip, rasterize_mode = _effective_splat_render_settings(config)
    with torch.inference_mode():
        colors, alpha, _ = rasterization(
            gaussians.means, gaussians.quats, gaussians.scales, gaussians.opacities,
            gaussians.sh, view, K, camera["width"], camera["height"],
            near_plane=camera["near"], far_plane=camera["far"],
            eps2d=eps2d, radius_clip=radius_clip,
            sh_degree=gaussians.sh_degree, backgrounds=torch.tensor([bg], device=device),
            render_mode=render_mode, rasterize_mode=rasterize_mode,
        )
    value = colors[0].float().cpu().numpy()
    rgb = np.clip(value[..., :3], 0.0, 1.0)
    a = np.clip(alpha[0, ..., 0].float().cpu().numpy(), 0.0, 1.0)
    depth = value[..., 3] if value.shape[-1] > 3 else None
    return rgb, a, depth


def render_mask_pass(gaussians: GaussianData, camera: dict, config: dict) -> np.ndarray:
    device = gaussians.means.device
    view = torch.tensor(camera["world_to_camera"], dtype=torch.float32, device=device)[None]
    K = torch.tensor(camera["intrinsic_matrix"], dtype=torch.float32, device=device)[None]
    foreground = torch.tensor(config.get("mask_pass_foreground_color", [255,255,255]), dtype=torch.float32, device=device) / 255.0
    background = torch.tensor([config.get("mask_pass_background_color", [255,0,255])], dtype=torch.float32, device=device) / 255.0
    colors = foreground[None].expand(gaussians.means.shape[0], 3)
    eps2d, radius_clip, rasterize_mode = _effective_splat_render_settings(config)
    with torch.inference_mode():
        rendered, _, _ = rasterization(
            gaussians.means, gaussians.quats, gaussians.scales, gaussians.opacities,
            colors, view, K, camera["width"], camera["height"],
            near_plane=camera["near"], far_plane=camera["far"], sh_degree=None,
            eps2d=eps2d, radius_clip=radius_clip,
            backgrounds=background, render_mode="RGB", rasterize_mode=rasterize_mode,
        )
    return np.clip(rendered[0].float().cpu().numpy()[..., :3], 0.0, 1.0)


def save_render(base: Path, camera: dict, result: tuple[np.ndarray, np.ndarray, np.ndarray | None], config: dict,
                mask_pass_rgb: np.ndarray | None = None) -> dict | None:
    rgb, alpha, depth = result
    name = camera["image"]
    image_dir = base / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rgb8 = np.rint(rgb * 255.0).astype(np.uint8)
    if config.get("background") == "transparent":
        rgba = np.dstack([rgb8, np.rint(alpha * 255.0).astype(np.uint8)])
        Image.fromarray(rgba, "RGBA").save(image_dir / name)
    else:
        Image.fromarray(rgb8, "RGB").save(image_dir / name)
    if config.get("render_alpha", False):
        d = base / "alpha"; d.mkdir(exist_ok=True)
        Image.fromarray(np.rint(alpha * 255.0).astype(np.uint8), "L").save(d / name)
    if config.get("mask_enabled", True):
        if mask_pass_rgb is None:
            raise ValueError("mask_pass_rgb is required when mask_enabled=true")
        repair_mask, mask_stats = build_background_key_mask(mask_pass_rgb, alpha, config)
        d = base / "mask"; d.mkdir(exist_ok=True)
        Image.fromarray(repair_mask, "L").save(d / name)
    if config.get("render_depth", False) and depth is not None:
        d = base / "depth"; d.mkdir(exist_ok=True)
        np.save(d / (Path(name).stem + ".npy"), depth.astype(np.float32))
    return mask_stats if config.get("mask_enabled", True) else None


def save_camera(base: Path, camera: dict) -> None:
    d = base / "cameras"; d.mkdir(parents=True, exist_ok=True)
    with (d / (Path(camera["image"]).stem + ".json")).open("w", encoding="utf-8") as f:
        json.dump(camera, f, ensure_ascii=False, indent=2)
