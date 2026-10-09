from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from core import PipelineConfig, run_pipeline


ROOT = Path(__file__).resolve().parent
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


@dataclass(frozen=True)
class InputPair:
    name: str
    video: Path
    mask_video: Path


def discover_input_pairs(input_dir: Path) -> list[InputPair]:
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")
    pairs: list[InputPair] = []
    invalid: list[str] = []
    candidate_dirs = sorted(
        {path.parent for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS},
        key=lambda path: path.relative_to(input_dir).as_posix(),
    )
    for directory in candidate_dirs:
        videos = sorted(path for path in directory.iterdir()
                        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
        masks = [path for path in videos if "mask" in path.stem.lower()]
        sources = [path for path in videos if path not in masks]
        relative = directory.relative_to(input_dir)
        display_name = relative.as_posix() if relative.parts else sources[0].stem if sources else "root"
        if len(sources) != 1 or len(masks) != 1:
            invalid.append(
                f"{display_name}: sources={[path.name for path in sources]}, "
                f"masks={[path.name for path in masks]}"
            )
            continue
        run_name = "__".join(relative.parts) if relative.parts else sources[0].stem
        pairs.append(InputPair(run_name, sources[0], masks[0]))
    if invalid:
        raise RuntimeError("Invalid input groups:\n- " + "\n- ".join(invalid))
    if not pairs:
        raise RuntimeError(f"No source/mask video pairs found below {input_dir}")
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Brush-ready clean RGBA + COLMAP dataset")
    parser.add_argument("video", type=Path, nargs="?", help="Source video; defaults to input/ auto-discovery")
    parser.add_argument("--mask-video", type=Path, help="Mask video; defaults to input/*mask*.mp4")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "input")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    parser.add_argument(
        "--colmap",
        default=os.environ.get("VIDEOSPLAT_COLMAP", ""),
        help="COLMAP executable/directory; defaults to VIDEOSPLAT_COLMAP or PATH",
    )
    parser.add_argument("--ffmpeg", default="")
    parser.add_argument("--ffprobe", default="")
    parser.add_argument("--mask-threshold", type=int, default=128)
    parser.add_argument("--erosion", type=int, default=1)
    parser.add_argument("--defringe", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.video is None and args.mask_video is None:
        pairs = discover_input_pairs(args.input_dir)
    elif args.video is not None and args.mask_video is not None:
        pairs = [InputPair(args.video.stem, args.video, args.mask_video)]
    else:
        raise RuntimeError("Provide both video and --mask-video, or provide neither to auto-discover input/")
    config = PipelineConfig(
        mask_threshold=args.mask_threshold,
        erosion_pixels=args.erosion,
        defringe_pixels=args.defringe,
    )
    summaries: list[dict[str, object]] = []
    failures = 0
    for index, pair in enumerate(pairs, start=1):
        print(f"[{index}/{len(pairs)}] Processing {pair.name}: {pair.video.name} + {pair.mask_video.name}", flush=True)
        try:
            report = run_pipeline(
                pair.video, pair.mask_video, args.output, run_name=pair.name,
                colmap_path=args.colmap, ffmpeg_path=args.ffmpeg, ffprobe_path=args.ffprobe,
                config=config,
            )
            summaries.append({
                "name": pair.name,
                "status": report["status"],
                "dataset_dir": report["dataset_dir"],
                "images": report["validation"]["image_count"],
                "registered_images": report["geometry"]["registered_images"],
                "registration_ratio": report["geometry"]["registration_ratio"],
                "points3d": report["geometry"]["points3d"],
                "camera_arc_degrees": report["geometry"]["camera_arc_degrees"],
                "hidden_rgb_pixels": report["validation"]["hidden_rgb_pixels"],
            })
        except Exception as error:
            failures += 1
            summaries.append({"name": pair.name, "status": "FAIL", "error": str(error)})
            print(f"[{index}/{len(pairs)}] FAILED {pair.name}: {error}", file=sys.stderr, flush=True)
    batch_report = {
        "status": "PASS" if failures == 0 else "PARTIAL" if failures < len(pairs) else "FAIL",
        "total": len(pairs),
        "passed": len(pairs) - failures,
        "failed": failures,
        "results": summaries,
    }
    (ROOT / "batch_report.json").write_text(
        json.dumps(batch_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(batch_report, ensure_ascii=False, indent=2))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
