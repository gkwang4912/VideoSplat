from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STAGE3_DIR = REPOSITORY_ROOT / "src" / "3_virtual_camera"
STAGE4_DIR = REPOSITORY_ROOT / "src" / "4_360brush"


def load_stage(directory: Path, prefix: str):
    config_spec = importlib.util.spec_from_file_location(
        f"{prefix}_config", directory / "config.py"
    )
    if config_spec is None or config_spec.loader is None:
        raise RuntimeError(f"Cannot load config from {directory}")
    stage_config = importlib.util.module_from_spec(config_spec)
    sys.modules[config_spec.name] = stage_config
    config_spec.loader.exec_module(stage_config)

    main_spec = importlib.util.spec_from_file_location(
        f"{prefix}_main", directory / "main.py"
    )
    if main_spec is None or main_spec.loader is None:
        raise RuntimeError(f"Cannot load main module from {directory}")
    stage_main = importlib.util.module_from_spec(main_spec)
    sys.modules[main_spec.name] = stage_main

    previous_config = sys.modules.get("config")
    sys.modules["config"] = stage_config
    try:
        main_spec.loader.exec_module(stage_main)
    finally:
        if previous_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = previous_config
    return stage_config, stage_main


STAGE3_CONFIG, STAGE3 = load_stage(STAGE3_DIR, "videosplat_stage3")
STAGE4_CONFIG, STAGE4 = load_stage(STAGE4_DIR, "videosplat_stage4")


class RgbaHandoffTest(unittest.TestCase):
    def test_stage3_rgba_is_accepted_by_stage4(self) -> None:
        with tempfile.TemporaryDirectory(prefix="videosplat_rgba_test_") as temp_name:
            dataset_dir = Path(temp_name) / "brush_dataset"
            images_dir = dataset_dir / "images"
            masks_dir = dataset_dir / "masks"
            sparse_dir = dataset_dir / "sparse" / "0"
            images_dir.mkdir(parents=True)
            masks_dir.mkdir(parents=True)
            sparse_dir.mkdir(parents=True)

            image_path = images_dir / "frame000.png"
            mask_path = masks_dir / "frame000.png"
            Image.new("RGB", (4, 3), (10, 20, 30)).save(image_path)
            mask = Image.new("L", (4, 3), 0)
            mask.putpixel((1, 1), 255)
            mask.save(mask_path)

            for filename in ("cameras.txt", "images.txt", "points3D.txt"):
                (sparse_dir / filename).write_text("", encoding="utf-8")

            poses = [
                STAGE3.ImagePose(
                    image_id=1,
                    qvec=(1.0, 0.0, 0.0, 0.0),
                    tvec=(0.0, 0.0, 0.0),
                    camera_id=1,
                    source_name="frame000.png",
                )
            ]
            STAGE3.compose_rgba_images(images_dir, masks_dir, poses)
            self.assertEqual(
                STAGE3.validate_dataset(images_dir, masks_dir, sparse_dir, poses),
                (1, 1),
            )

            with Image.open(image_path) as rgba, Image.open(mask_path) as saved_mask:
                self.assertEqual(rgba.mode, "RGBA")
                self.assertEqual(
                    rgba.getchannel("A").tobytes(),
                    saved_mask.tobytes(),
                )

            valid, reason = STAGE4.validate_dataset(dataset_dir)
            self.assertTrue(valid, reason)

    def test_stage4_defaults_to_stage3_output(self) -> None:
        expected = (REPOSITORY_ROOT / "src" / "3_virtual_camera" / "output").resolve()
        self.assertEqual(STAGE4_CONFIG.INPUT_DIR.resolve(), expected)


if __name__ == "__main__":
    unittest.main()
