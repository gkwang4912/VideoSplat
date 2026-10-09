from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

from .core import PipelineConfig, make_contact_sheet, run_pipeline


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
PACKAGE_ROOT = Path(__file__).resolve().parent


def _video_choices() -> list[str]:
    root = PACKAGE_ROOT / "input"
    choices = sorted(str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*")
                     if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS and "mask" not in path.stem.lower())
    return choices or ["NO_VIDEO_FOUND"]


def _mask_video_choices() -> list[str]:
    root = PACKAGE_ROOT / "input"
    choices = sorted(str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*")
                     if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS and "mask" in path.stem.lower())
    return choices or ["NO_MASK_VIDEO_FOUND"]


def _image_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.astype(np.float32) / 255.0)).unsqueeze(0)


def _mask_tensor(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32) / 255.0)).unsqueeze(0)


class VideoSplatDataset(dict):
    pass


class VideoSplatBuildDataset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (_video_choices(),),
                "mask_video": (_mask_video_choices(),),
                "colmap_path": ("STRING", {"default": "", "multiline": False}),
                "output_subfolder": ("STRING", {"default": "video_splat", "multiline": False}),
                "mask_threshold": ("INT", {"default": 128, "min": 1, "max": 254}),
                "erosion_pixels": ("INT", {"default": 1, "min": 0, "max": 3}),
                "defringe_pixels": ("INT", {"default": 3, "min": 0, "max": 8}),
            }
        }

    RETURN_TYPES = ("VIDEO_SPLAT_DATASET", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("dataset", "contact_sheet", "representative_mask", "report_json")
    FUNCTION = "build"
    CATEGORY = "video_splat"
    OUTPUT_NODE = True

    def build(
        self, video: str, mask_video: str, colmap_path: str,
        output_subfolder: str, mask_threshold: int, erosion_pixels: int, defringe_pixels: int,
    ):
        if video == "NO_VIDEO_FOUND" or mask_video == "NO_MASK_VIDEO_FOUND":
            raise RuntimeError(f"Source video and mask video are required in {PACKAGE_ROOT / 'input'}")
        input_root = (PACKAGE_ROOT / "input").resolve()
        video_path = (input_root / video).resolve()
        mask_video_path = (input_root / mask_video).resolve()
        if input_root not in video_path.parents or input_root not in mask_video_path.parents:
            raise RuntimeError("Input path escapes the package input directory")
        safe_folder = Path(output_subfolder).name or "video_splat"
        config = PipelineConfig(
            mask_threshold=mask_threshold,
            erosion_pixels=erosion_pixels,
            defringe_pixels=defringe_pixels,
        )
        report = run_pipeline(
            video_path, mask_video_path, PACKAGE_ROOT / "output" / safe_folder,
            run_name="__".join(Path(video).parent.parts) if Path(video).parent.parts else video_path.stem,
            colmap_path=colmap_path, config=config,
        )
        sheet, representative = make_contact_sheet(Path(report["dataset_dir"]))
        dataset = VideoSplatDataset(report)
        return dataset, _image_tensor(sheet), _mask_tensor(representative), json.dumps(report, ensure_ascii=False, indent=2)


class VideoSplatPreviewDataset:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"dataset": ("VIDEO_SPLAT_DATASET",)}}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("contact_sheet", "representative_mask", "metadata_json")
    FUNCTION = "preview"
    CATEGORY = "video_splat/Preview"

    def preview(self, dataset: VideoSplatDataset):
        sheet, representative = make_contact_sheet(Path(dataset["dataset_dir"]))
        return _image_tensor(sheet), _mask_tensor(representative), json.dumps(dict(dataset), ensure_ascii=False, indent=2)


NODE_CLASS_MAPPINGS = {
    "VideoSplatBuildDataset": VideoSplatBuildDataset,
    "VideoSplatPreviewDataset": VideoSplatPreviewDataset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoSplatBuildDataset": "Video Splat Build Dataset",
    "VideoSplatPreviewDataset": "Video Splat Preview Dataset",
}
