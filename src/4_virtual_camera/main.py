from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import config
from PIL import Image


CAMERA_MODELS = {
    "SIMPLE_PINHOLE": (0, 3),
    "PINHOLE": (1, 4),
    "SIMPLE_RADIAL": (2, 4),
    "RADIAL": (3, 5),
    "OPENCV": (4, 8),
    "OPENCV_FISHEYE": (5, 8),
    "FULL_OPENCV": (6, 12),
    "FOV": (7, 5),
    "SIMPLE_RADIAL_FISHEYE": (8, 4),
    "RADIAL_FISHEYE": (9, 5),
    "THIN_PRISM_FISHEYE": (10, 12),
}


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ImagePose:
    image_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    source_name: str


@dataclass(frozen=True)
class Point3D:
    point_id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error: float
    track: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class InputGroup:
    name: str
    root: Path
    color_video: Path
    mask_video: Path
    source_colmap: Path


@dataclass(frozen=True)
class BuildResult:
    group: InputGroup
    dataset_dir: Path
    zip_path: Path | None
    image_count: int
    camera_count: int
    point_count: int


def require_invariants() -> None:
    required_true = {
        "REUSE_ORIGINAL_CAMERA_POSES": config.REUSE_ORIGINAL_CAMERA_POSES,
        "COMPOSE_RGBA_IMAGES": config.COMPOSE_RGBA_IMAGES,
    }
    required_false = {
        "ESTIMATE_CAMERAS": config.ESTIMATE_CAMERAS,
        "OPTIMIZE_CAMERAS": config.OPTIMIZE_CAMERAS,
        "MATCH_FEATURES": config.MATCH_FEATURES,
        "EXPORT_SEPARATE_MASKS": config.EXPORT_SEPARATE_MASKS,
    }
    invalid = [name for name, value in required_true.items() if value is not True]
    invalid += [name for name, value in required_false.items() if value is not False]
    if invalid:
        raise RuntimeError(
            "Experiment invariants are disabled or contradicted in config.py: "
            + ", ".join(invalid)
        )


def require_media_tools() -> None:
    for executable in (config.FFMPEG_EXECUTABLE, config.FFPROBE_EXECUTABLE):
        if shutil.which(executable) is None:
            raise FileNotFoundError(f"Required executable not found: {executable!r}")


def mask_likeness(video_path: Path) -> tuple[float, float]:
    sample_count = max(2, config.MASK_SAMPLE_FRAMES)
    last_frame = config.EXPECTED_FRAMES - 1
    sample_indices = sorted(
        {
            round(index * last_frame / (sample_count - 1))
            for index in range(sample_count)
        }
    )
    select_expression = "+".join(
        f"eq(n\\,{frame_index})" for frame_index in sample_indices
    )
    size = config.MASK_ANALYSIS_SIZE
    command = [
        config.FFMPEG_EXECUTABLE,
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select={select_expression},"
        f"scale={size}:{size}:flags=neighbor,format=rgb24",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    try:
        raw = subprocess.run(command, check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Cannot sample video {video_path}: {detail}") from exc
    if not raw or len(raw) % 3:
        raise ValueError(f"No usable RGB samples decoded from {video_path}")

    pixel_count = len(raw) // 3
    grayscale_count = 0
    binary_count = 0
    for offset in range(0, len(raw), 3):
        red, green, blue = raw[offset : offset + 3]
        is_grayscale = (
            max(red, green, blue) - min(red, green, blue)
            <= config.MASK_GRAYSCALE_TOLERANCE
        )
        if is_grayscale:
            grayscale_count += 1
            level = (red + green + blue) / 3
            if level <= config.MASK_BINARY_LOW or level >= config.MASK_BINARY_HIGH:
                binary_count += 1
    return grayscale_count / pixel_count, binary_count / pixel_count


def video_signature(video_path: Path) -> tuple[int, int, int]:
    command = [
        config.FFPROBE_EXECUTABLE,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_read_frames,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        payload = subprocess.run(
            command, check=True, capture_output=True, text=True, encoding="utf-8"
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Cannot inspect video stream: {video_path}") from exc
    streams = json.loads(payload).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream in {video_path}")
    stream = streams[0]
    frame_value = stream.get("nb_read_frames") or stream.get("nb_frames")
    if frame_value in (None, "N/A"):
        raise ValueError(f"Cannot determine frame count for {video_path}")
    return int(stream["width"]), int(stream["height"]), int(frame_value)


def detect_video_pair(folder: Path) -> tuple[Path, Path]:
    videos = sorted(folder.glob(config.VIDEO_GLOB), key=lambda path: path.name.casefold())
    if len(videos) < 2:
        raise ValueError(
            f"Input group {folder.name!r} requires at least two MP4 files; "
            f"found {len(videos)}"
        )

    likeness = {video: mask_likeness(video) for video in videos}
    mask_candidates = [
        video
        for video, (grayscale_fraction, binary_fraction) in likeness.items()
        if grayscale_fraction >= config.MASK_MIN_GRAYSCALE_FRACTION
        and binary_fraction >= config.MASK_MIN_BINARY_FRACTION
    ]
    if len(mask_candidates) != 1:
        scores = "; ".join(
            f"{video.name}: gray={gray:.3f}, binary={binary:.3f}"
            for video, (gray, binary) in likeness.items()
        )
        raise ValueError(
            f"Input group {folder.name!r} must contain exactly one black/white "
            f"mask video; found {len(mask_candidates)}. {scores}"
        )

    mask_video = mask_candidates[0]
    mask_signature = video_signature(mask_video)
    color_candidates = [
        video
        for video in videos
        if video != mask_video and video_signature(video) == mask_signature
    ]
    if len(color_candidates) != 1:
        matches = ", ".join(video.name for video in color_candidates) or "none"
        raise ValueError(
            f"Input group {folder.name!r} must contain exactly one non-mask video "
            f"with the mask's width, height, and frame count {mask_signature}; "
            f"found {len(color_candidates)}: {matches}"
        )
    return color_candidates[0], mask_video


def discover_input_groups() -> list[InputGroup]:
    require_media_tools()
    if not config.INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Input directory not found: {config.INPUT_DIR}")

    candidates = [
        folder
        for folder in config.INPUT_DIR.iterdir()
        if folder.is_dir()
        and (
            any(folder.glob("*.mp4"))
            or (folder / config.COLMAP_RELATIVE_PATH).exists()
        )
    ]
    candidates.sort(key=lambda path: path.name.casefold())
    if not candidates:
        raise FileNotFoundError(
            f"No input groups found directly under {config.INPUT_DIR}"
        )

    groups: list[InputGroup] = []
    for folder in candidates:
        color_video, mask_video = detect_video_pair(folder)
        groups.append(
            InputGroup(
                name=folder.name,
                root=folder,
                color_video=color_video,
                mask_video=mask_video,
                source_colmap=folder / config.COLMAP_RELATIVE_PATH,
            )
        )
    return groups


def require_inputs(group: InputGroup) -> None:
    required = [
        group.color_video,
        group.mask_video,
        group.source_colmap / "cameras.txt",
        group.source_colmap / "images.txt",
        group.source_colmap / "points3D.txt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required input file(s) missing:\n" + "\n".join(missing))
    if shutil.which(config.FFMPEG_EXECUTABLE) is None:
        raise FileNotFoundError(
            f"ffmpeg executable not found: {config.FFMPEG_EXECUTABLE!r}"
        )


def data_lines(path: Path) -> list[str]:
    try:
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    except OSError as exc:
        raise OSError(f"Cannot read {path}: {exc}") from exc


def read_cameras(path: Path) -> list[Camera]:
    cameras: list[Camera] = []
    for line_number, line in enumerate(data_lines(path), start=1):
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"Invalid camera record {line_number} in {path}")
        camera_id = int(fields[0])
        model = fields[1]
        if model not in CAMERA_MODELS:
            raise ValueError(f"Unsupported COLMAP camera model: {model}")
        expected_params = CAMERA_MODELS[model][1]
        params = tuple(float(value) for value in fields[4:])
        if len(params) != expected_params:
            raise ValueError(
                f"Camera {camera_id} model {model} requires {expected_params} "
                f"parameters, found {len(params)}"
            )
        cameras.append(
            Camera(camera_id, model, int(fields[2]), int(fields[3]), params)
        )
    if not cameras:
        raise ValueError(f"No cameras found in {path}")
    if len({camera.camera_id for camera in cameras}) != len(cameras):
        raise ValueError(f"Duplicate camera IDs in {path}")
    return cameras


def read_image_poses(path: Path) -> list[ImagePose]:
    poses: list[ImagePose] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        # Pose rows have exactly ten fields. Empty POINTS2D rows are blank; a
        # populated POINTS2D row has a field count divisible by three.
        if len(fields) == 10:
            try:
                pose = ImagePose(
                    image_id=int(fields[0]),
                    qvec=tuple(float(value) for value in fields[1:5]),
                    tvec=tuple(float(value) for value in fields[5:8]),
                    camera_id=int(fields[8]),
                    source_name=fields[9],
                )
            except ValueError as exc:
                raise ValueError(f"Invalid image pose at {path}:{line_number}") from exc
            poses.append(pose)
        elif len(fields) % 3 != 0:
            raise ValueError(f"Invalid images.txt row at {path}:{line_number}")
    if not poses:
        raise ValueError(f"No image poses found in {path}")
    if len({pose.image_id for pose in poses}) != len(poses):
        raise ValueError(f"Duplicate image IDs in {path}")
    return poses


def read_points3d(path: Path) -> list[Point3D]:
    points: list[Point3D] = []
    for line_number, line in enumerate(data_lines(path), start=1):
        fields = line.split()
        if len(fields) < 8 or (len(fields) - 8) % 2 != 0:
            raise ValueError(f"Invalid point record {line_number} in {path}")
        track = tuple(
            (int(fields[index]), int(fields[index + 1]))
            for index in range(8, len(fields), 2)
        )
        points.append(
            Point3D(
                point_id=int(fields[0]),
                xyz=tuple(float(value) for value in fields[1:4]),
                rgb=tuple(int(value) for value in fields[4:7]),
                error=float(fields[7]),
                track=track,
            )
        )
    return points


def extract_video_frames(video_path: Path, images_dir: Path, prefix: str) -> list[Path]:
    pattern = images_dir / f"{prefix}%06d.{config.IMAGE_FORMAT}"
    command = [
        config.FFMPEG_EXECUTABLE,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-fps_mode",
        "passthrough",
        "-start_number",
        "0",
        str(pattern),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg frame extraction failed with exit code {exc.returncode}") from exc

    frames = sorted(images_dir.glob(f"{prefix}*.{config.IMAGE_FORMAT}"))
    if len(frames) != config.EXPECTED_FRAMES:
        raise ValueError(
            f"Expected exactly {config.EXPECTED_FRAMES} decoded frames, found {len(frames)}"
        )
    expected_names = [
        f"{prefix}{index:06d}.{config.IMAGE_FORMAT}"
        for index in range(config.EXPECTED_FRAMES)
    ]
    actual_names = [path.name for path in frames]
    if actual_names != expected_names:
        raise ValueError("Decoded frame names are not a contiguous zero-based sequence")
    return frames


def safe_colmap_image_name(name: str) -> Path:
    normalized = Path(name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"Unsafe COLMAP image path: {name!r}")
    if normalized.suffix.lower() != f".{config.IMAGE_FORMAT.lower()}":
        raise ValueError(
            f"COLMAP image {name!r} is not a {config.IMAGE_FORMAT.upper()} file; "
            "renaming it would modify the registered image name"
        )
    return normalized


def write_rgba_images(
    color_frames: list[Path],
    mask_frames: list[Path],
    poses: list[ImagePose],
    images_dir: Path,
) -> list[Path]:
    if len(color_frames) != len(mask_frames) or len(color_frames) != len(poses):
        raise ValueError(
            f"Cannot pair {len(color_frames)} color frames, "
            f"{len(mask_frames)} mask frames, and {len(poses)} poses"
        )

    output_images: list[Path] = []
    for color_frame, mask_frame, pose in zip(
        color_frames, mask_frames, poses, strict=True
    ):
        relative_name = safe_colmap_image_name(pose.source_name)
        destination = images_dir / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(color_frame) as source_image, Image.open(mask_frame) as source_mask:
            rgb = source_image.convert("RGB")
            alpha = source_mask.convert("L")
            if rgb.size != alpha.size:
                alpha = alpha.resize(rgb.size, resample=Image.Resampling.NEAREST)
            rgba = rgb.copy()
            rgba.putalpha(alpha)
            rgba.save(destination, format="PNG")
        output_images.append(destination)
    return output_images


def copy_colmap_text_model(source_dir: Path, sparse_dir: Path) -> None:
    for filename in ("cameras.txt", "images.txt", "points3D.txt"):
        shutil.copy2(source_dir / filename, sparse_dir / filename)


def validate_dataset(
    images_dir: Path,
    sparse_dir: Path,
    poses: list[ImagePose],
) -> int:
    registered_names = [safe_colmap_image_name(pose.source_name) for pose in poses]
    if len(set(registered_names)) != len(registered_names):
        raise ValueError("COLMAP contains duplicate registered image names")

    image_files = sorted(path for path in images_dir.rglob("*") if path.is_file())
    if len(image_files) != len(poses):
        raise ValueError(
            f"Registered/output count mismatch: {len(poses)} registered, "
            f"{len(image_files)} RGBA images"
        )

    for relative_name in registered_names:
        image_path = images_dir / relative_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Registered RGBA image missing: {image_path}")
        with Image.open(image_path) as image:
            if image.mode != "RGBA":
                raise ValueError(
                    f"Brush image must contain RGBA channels: {image_path} "
                    f"has mode {image.mode}"
                )
            alpha_min, alpha_max = image.getchannel("A").getextrema()
            if alpha_min == alpha_max:
                raise ValueError(f"RGBA alpha channel is constant: {image_path}")

    expected_model_files = {"cameras.txt", "images.txt", "points3D.txt"}
    actual_model_files = {path.name for path in sparse_dir.iterdir() if path.is_file()}
    if actual_model_files != expected_model_files:
        raise ValueError(
            f"COLMAP model files conflict or are incomplete: {sorted(actual_model_files)}"
        )
    return len(image_files)


def make_zip(dataset_dir: Path, zip_path: Path) -> None:
    temporary_zip = zip_path.with_suffix(zip_path.suffix + ".tmp")
    if temporary_zip.exists():
        temporary_zip.unlink()
    try:
        with zipfile.ZipFile(
            temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in sorted(dataset_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(dataset_dir).as_posix())
        temporary_zip.replace(zip_path)
    finally:
        if temporary_zip.exists():
            temporary_zip.unlink()


def build_dataset(group: InputGroup) -> BuildResult:
    require_invariants()
    require_inputs(group)

    cameras = read_cameras(group.source_colmap / "cameras.txt")
    poses = read_image_poses(group.source_colmap / "images.txt")
    points = read_points3d(group.source_colmap / "points3D.txt")

    if len(poses) != config.EXPECTED_FRAMES:
        raise ValueError(
            f"Expected {config.EXPECTED_FRAMES} source camera poses, found {len(poses)}"
        )
    camera_ids = {camera.camera_id for camera in cameras}
    missing_camera_ids = sorted({pose.camera_id for pose in poses} - camera_ids)
    if missing_camera_ids:
        raise ValueError(f"Image poses reference missing camera IDs: {missing_camera_ids}")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    group_output_dir = config.OUTPUT_DIR / group.name
    group_output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = group_output_dir / config.DATASET_FOLDER_NAME
    zip_path = group_output_dir / config.ZIP_FILE_NAME
    if not config.OVERWRITE_OUTPUT and (dataset_dir.exists() or zip_path.exists()):
        raise FileExistsError("Output exists and OVERWRITE_OUTPUT is False")

    with tempfile.TemporaryDirectory(
        prefix=f".brush_build_{group.name}_", dir=config.OUTPUT_DIR
    ) as temp:
        staged_dataset = Path(temp) / "brush_dataset"
        images_dir = staged_dataset / "images"
        sparse_dir = staged_dataset / "sparse" / "0"
        images_dir.mkdir(parents=True)
        sparse_dir.mkdir(parents=True)

        raw_color_dir = Path(temp) / "color_frames"
        raw_mask_dir = Path(temp) / "mask_frames"
        raw_color_dir.mkdir()
        raw_mask_dir.mkdir()
        color_frames = extract_video_frames(
            group.color_video, raw_color_dir, config.IMAGE_PREFIX
        )
        mask_frames = extract_video_frames(
            group.mask_video, raw_mask_dir, config.MASK_PREFIX
        )
        write_rgba_images(color_frames, mask_frames, poses, images_dir)
        copy_colmap_text_model(group.source_colmap, sparse_dir)
        image_count = validate_dataset(images_dir, sparse_dir, poses)

        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        staged_dataset.replace(dataset_dir)

    if config.EXPORT_ZIP:
        make_zip(dataset_dir, zip_path)
        exported_zip: Path | None = zip_path
    else:
        exported_zip = None

    return BuildResult(
        group=group,
        dataset_dir=dataset_dir,
        zip_path=exported_zip,
        image_count=image_count,
        camera_count=len(cameras),
        point_count=len(points),
    )


def main() -> int:
    try:
        groups = discover_input_groups()
        print(f"Discovered {len(groups)} input group(s).")
        results: list[BuildResult] = []
        for group in groups:
            print(f"[{group.name}] Color:  {group.color_video.name}")
            print(f"[{group.name}] Mask:   {group.mask_video.name}")
            print(f"[{group.name}] COLMAP: {group.source_colmap}")
            results.append(build_dataset(group))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Created {len(results)} Brush dataset(s) successfully.")
    for result in results:
        print(f"[{result.group.name}] Dataset: {result.dataset_dir}")
        if result.zip_path is not None:
            print(f"[{result.group.name}] ZIP:     {result.zip_path}")
        print(f"[{result.group.name}] Images:  {result.image_count}")
        print(f"[{result.group.name}] Cameras: {result.camera_count}")
        print(f"[{result.group.name}] Points:  {result.point_count}")
        print(f"[{result.group.name}] Camera poses: reused unchanged in source file order")
        if config.BRUSH_EXECUTABLE.is_file():
            print(f"[{result.group.name}] Brush command:")
            print(
                f'  & "{config.BRUSH_EXECUTABLE}" "{result.dataset_dir}"'
                f" --with-viewer --alpha-mode {config.BRUSH_ALPHA_MODE}"
            )
            if result.point_count == 0:
                print(
                    f"[{result.group.name}] Initialization: Brush random initialization "
                    "(source point cloud is empty)"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
