from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_video(images: Path, output: Path, fps: int, digits: int = 4) -> None:
    ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", str(images / f"%0{digits}d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(output)]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError("ffmpeg failed:\n" + completed.stderr[-3000:])


def make_side_by_side_video(images: Path, masks: Path, output: Path, fps: int, digits: int = 4) -> None:
    ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    pattern = f"%0{digits}d.png"
    cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", str(images / pattern),
           "-framerate", str(fps), "-i", str(masks / pattern),
           "-filter_complex", "[0:v]format=rgb24[left];[1:v]format=rgb24[right];[left][right]hstack=inputs=2[v]",
           "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(output)]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError("ffmpeg side-by-side failed:\n" + completed.stderr[-3000:])


def collect_and_combine_previews(output: Path, path_names: list[str], combine_sequence: list[dict] | None = None,
                                 fps: int = 30) -> list[Path]:
    """Copy path previews into output/All and concatenate them in the supplied order."""
    ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    all_dir = output / "All"
    all_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    sources_by_name: dict[str, Path] = {}
    for path_name in path_names:
        source = output / "paths" / path_name / "preview.mp4"
        if not source.is_file():
            raise FileNotFoundError(f"Path preview is missing: {source}")
        destination = all_dir / f"{path_name}.mp4"
        shutil.copy2(source, destination)
        sources_by_name[path_name] = source
        copied.append(destination)

    if combine_sequence is None:
        combine_sequence = [{"path": path_name, "reverse": False} for path_name in path_names]
    sources: list[Path] = []
    temporary_files: list[Path] = []
    for index, entry in enumerate(combine_sequence, 1):
        path_name = entry["path"]
        if path_name not in sources_by_name:
            raise KeyError(f"Unknown combine path: {path_name}")
        source = sources_by_name[path_name]
        if not entry.get("reverse", False):
            sources.append(source)
            continue
        reversed_path = all_dir / f".combine_{index:02d}_{path_name}_reverse.mp4"
        cmd = [ffmpeg, "-y", "-i", str(source), "-vf", "reverse,setpts=N/FRAME_RATE/TB",
               "-an", "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(reversed_path)]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError("ffmpeg reverse failed:\n" + completed.stderr[-3000:])
        sources.append(reversed_path)
        temporary_files.append(reversed_path)

    concat_list = all_dir / ".combine_inputs.txt"
    # The concat demuxer accepts forward-slash absolute Windows paths. Single
    # quotes in a path are escaped according to ffmpeg's concat-file syntax.
    lines = ["file '" + source.resolve().as_posix().replace("'", "'\\''") + "'" for source in sources]
    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
               "-movflags", "+faststart", str(output / "combine.mp4")]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError("ffmpeg combine failed:\n" + completed.stderr[-3000:])
    finally:
        concat_list.unlink(missing_ok=True)
        for temporary_file in temporary_files:
            temporary_file.unlink(missing_ok=True)
    return copied


def concat_videos(sources: list[Path], output: Path) -> None:
    ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    output.parent.mkdir(parents=True, exist_ok=True)
    concat_list = output.parent / f".{output.stem}_inputs.txt"
    lines = ["file '" + source.resolve().as_posix().replace("'", "'\\''") + "'" for source in sources]
    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
               "-movflags", "+faststart", str(output)]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError("ffmpeg concat failed:\n" + completed.stderr[-3000:])
    finally:
        concat_list.unlink(missing_ok=True)


def probe_video_dimensions(path: Path) -> tuple[int, int]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required to validate video dimensions")
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
           "-of", "csv=s=x:p=0", str(path)]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError("ffprobe failed:\n" + completed.stderr[-2000:])
    width, height = completed.stdout.strip().split("x")
    return int(width), int(height)


def contact_sheet(paths: list[Path], labels: list[str], output: Path, columns: int = 4) -> None:
    opened = [Image.open(p).convert("RGB") for p in paths]
    thumb = 320
    rows = (len(opened) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb, rows * (thumb + 36)), "#202020")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=20)
    for i, (im, label) in enumerate(zip(opened, labels)):
        im.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        x = (i % columns) * thumb + (thumb - im.width) // 2
        y = (i // columns) * (thumb + 36)
        sheet.paste(im, (x, y))
        draw.text((i % columns * thumb + 8, y + thumb + 7), label, fill="white", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def write_colmap(root: Path, cameras: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    first = cameras[0]
    (root / "cameras.txt").write_text(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {first['width']} {first['height']} {first['fx']} {first['fy']} {first['cx']} {first['cy']}\n",
        encoding="utf-8")
    lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME", "# POINTS2D[] intentionally empty"]
    for i, cam in enumerate(cameras, 1):
        q = rotmat_to_quat(np.array(cam["rotation_matrix"], dtype=float))
        t = cam["translation"]
        lines.extend([f"{i} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {cam['image']}", ""])
    (root / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "points3D.txt").write_text("# No reconstructed points; known cameras only.\n", encoding="utf-8")


def rotmat_to_quat(m: np.ndarray) -> np.ndarray:
    # Stable matrix to wxyz quaternion conversion.
    q = np.empty(4)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; q[:] = [0.25*s, (m[2,1]-m[1,2])/s, (m[0,2]-m[2,0])/s, (m[1,0]-m[0,1])/s]
    else:
        i = int(np.argmax(np.diag(m))); j, k = (i+1)%3, (i+2)%3
        s = np.sqrt(1.0 + m[i,i] - m[j,j] - m[k,k]) * 2
        q[0] = (m[k,j]-m[j,k])/s; q[i+1] = 0.25*s; q[j+1] = (m[j,i]+m[i,j])/s; q[k+1] = (m[k,i]+m[i,k])/s
    return q / np.linalg.norm(q)
