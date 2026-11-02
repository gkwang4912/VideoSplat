#!/usr/bin/env python3
"""Batch ComfyUI orbit-video generation plus COLMAP and Brush datasets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

if os.name == "nt":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class PipelineError(RuntimeError):
    pass


def log(message: str) -> None:
    print(message, flush=True)


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"無法讀取 JSON：{path}\n{exc}") from exc


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp, path)


def resolve_from_root(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def ensure_command(name: str) -> str:
    command = shutil.which(name)
    if not command:
        raise PipelineError(f"找不到必要程式：{name}。請先安裝並加入 PATH。")
    return command


def run_command(command: list[str], description: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise PipelineError(f"{description}失敗（exit {result.returncode}）：\n{details}")
    return result


def safe_remove_tree(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve()
    root = allowed_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise PipelineError(f"拒絕刪除輸出範圍外的資料夾：{resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


@dataclass(frozen=True)
class Job:
    name: str
    source_video: Path
    camera_path: Path


@dataclass
class Settings:
    root: Path
    raw: dict[str, Any]

    @property
    def comfy_url(self) -> str:
        return str(self.raw.get("comfyui_url", "http://127.0.0.1:8188")).rstrip("/")

    def path(self, key: str) -> Path:
        return resolve_from_root(self.root, str(self.raw[key]))

    @property
    def output_root(self) -> Path:
        return self.path("output_dir")

    @property
    def accepted_frame_counts(self) -> set[int]:
        values = self.raw.get("accepted_output_frame_counts", [90, 91])
        return {int(value) for value in values}


class ComfyUIClient:
    def __init__(self, base_url: str, timeout_seconds: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.client_id = str(uuid.uuid4())

    def _request_json(self, route: str, method: str = "GET", payload: Any = None) -> Any:
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base_url + route, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise PipelineError(f"無法連線 ComfyUI：{self.base_url}{route}\n{exc}") from exc
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PipelineError(f"ComfyUI 回傳非 JSON：{route}") from exc

    def system_stats(self) -> Any:
        return self._request_json("/system_stats")

    def object_info(self) -> Any:
        return self._request_json("/object_info")

    def upload_file(self, source: Path, subfolder: str) -> str:
        boundary = "----OrbitPipeline" + uuid.uuid4().hex
        filename = source.name
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        parts = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"subfolder\"\r\n\r\n{subfolder}\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\n"
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8"),
            source.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
        request = urllib.request.Request(
            self.base_url + "/upload/image",
            data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise PipelineError(f"上傳到 ComfyUI 失敗：{source}\n{exc}") from exc
        uploaded_name = str(result.get("name", filename))
        uploaded_subfolder = str(result.get("subfolder", subfolder)).replace("\\", "/").strip("/")
        return f"{uploaded_subfolder}/{uploaded_name}" if uploaded_subfolder else uploaded_name

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        result = self._request_json(
            "/prompt", method="POST", payload={"prompt": workflow, "client_id": self.client_id}
        )
        if "error" in result:
            raise PipelineError("ComfyUI 拒絕工作流：\n" + json.dumps(result, ensure_ascii=False, indent=2))
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            raise PipelineError("ComfyUI 沒有回傳 prompt_id。")
        return str(prompt_id)

    def wait_for_prompt(self, prompt_id: str, poll_seconds: float, timeout_seconds: int) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        next_status = time.monotonic() + 30
        while time.monotonic() < deadline:
            history = self._request_json("/history/" + urllib.parse.quote(prompt_id))
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error" or not status.get("completed", True):
                    messages = status.get("messages", [])
                    raise PipelineError("ComfyUI 執行失敗：\n" + json.dumps(messages, ensure_ascii=False, indent=2))
                return entry
            if time.monotonic() >= next_status:
                log(f"  ComfyUI 仍在生成（prompt {prompt_id[:8]}…）")
                next_status = time.monotonic() + 30
            time.sleep(poll_seconds)
        raise PipelineError(f"等待 ComfyUI 超時：{timeout_seconds} 秒，prompt_id={prompt_id}")

    def download_output(self, descriptor: dict[str, Any], destination: Path) -> None:
        query = urllib.parse.urlencode(
            {
                "filename": descriptor["filename"],
                "subfolder": descriptor.get("subfolder", ""),
                "type": descriptor.get("type", "output"),
            }
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(destination.suffix + ".part")
        try:
            with urllib.request.urlopen(self.base_url + "/view?" + query, timeout=self.timeout_seconds) as response:
                with temp.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            os.replace(temp, destination)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            temp.unlink(missing_ok=True)
            raise PipelineError(f"下載 ComfyUI 輸出失敗：{descriptor}\n{exc}") from exc


def load_settings(config_path: Path) -> Settings:
    raw = load_json(config_path)
    required = ["video_dir", "paths_dir", "workflow", "reference_image", "output_dir"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise PipelineError("設定檔缺少欄位：" + ", ".join(missing))
    return Settings(config_path.parent.resolve(), raw)


def discover_jobs(settings: Settings, only: list[str] | None = None) -> list[Job]:
    video_dir = settings.path("video_dir")
    paths_dir = settings.path("paths_dir")
    if not video_dir.is_dir():
        raise PipelineError(f"找不到影片資料夾：{video_dir}")
    if not paths_dir.is_dir():
        raise PipelineError(f"找不到 camera paths 資料夾：{paths_dir}")
    filters = {item.lower() for item in (only or [])}
    videos = sorted(
        (path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: natural_key(path.name),
    )
    if filters:
        videos = [video for video in videos if video.stem.lower() in filters or video.name.lower() in filters]
    if not videos:
        raise PipelineError(f"沒有找到待處理影片：{video_dir}")
    jobs: list[Job] = []
    for video in videos:
        camera_path = paths_dir / video.stem
        if not camera_path.is_dir():
            raise PipelineError(f"影片缺少同名 camera path：{video.name} -> {camera_path}")
        jobs.append(Job(video.stem, video, camera_path))
    return jobs


def discover_consolidated_videos(settings: Settings) -> list[Path]:
    video_dir = resolve_from_root(
        settings.root, str(settings.raw.get("consolidated_video_dir", "input"))
    )
    if not video_dir.is_dir():
        return []
    return sorted(
        (path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: natural_key(path.name),
    )


def camera_files(camera_path: Path) -> list[Path]:
    folder = camera_path / "cameras"
    files = sorted(folder.glob("*.json"), key=lambda path: natural_key(path.name))
    if not files:
        raise PipelineError(f"找不到 camera JSON：{folder}")
    return files


def probe_video(video: Path) -> dict[str, Any]:
    ffprobe = ensure_command("ffprobe")
    result = run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames,nb_frames",
            "-of",
            "json",
            str(video),
        ],
        f"讀取影片資訊 {video.name}",
    )
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise PipelineError(f"影片沒有影像軌：{video}")
    stream = streams[0]
    count = stream.get("nb_read_frames") or stream.get("nb_frames")
    stream["frame_count"] = int(count) if count and str(count).isdigit() else None
    return stream


def validate_workflow(settings: Settings, check_server: bool = True) -> dict[str, Any]:
    workflow_path = settings.path("workflow")
    workflow = load_json(workflow_path)
    nodes = settings.raw.get("workflow_nodes", {})
    required_nodes = {"video": "1", "reference_image": "3", "save_video": "16"}
    required_nodes.update({key: str(value) for key, value in nodes.items()})
    expected = {"video": "LoadVideo", "reference_image": "LoadImage", "save_video": "SaveVideo"}
    for role, expected_class in expected.items():
        node_id = required_nodes[role]
        node = workflow.get(node_id)
        if not node:
            raise PipelineError(f"工作流缺少 {role} 節點 ID {node_id}")
        if node.get("class_type") != expected_class:
            raise PipelineError(
                f"工作流節點 {node_id} 應為 {expected_class}，實際是 {node.get('class_type')}"
            )
    if check_server:
        client = ComfyUIClient(settings.comfy_url)
        client.system_stats()
        object_info = client.object_info()
        missing_classes = sorted(
            {
                str(node.get("class_type"))
                for node in workflow.values()
                if str(node.get("class_type")) not in object_info
            }
        )
        if missing_classes:
            raise PipelineError("目前 ComfyUI 缺少工作流節點：\n- " + "\n- ".join(missing_classes))
    return workflow


def find_video_descriptor(history_entry: dict[str, Any], save_node_id: str) -> dict[str, Any]:
    outputs = history_entry.get("outputs", {})
    preferred = outputs.get(save_node_id, {})
    candidates = [preferred] + [value for key, value in outputs.items() if key != save_node_id]
    for output in candidates:
        for key in ("videos", "gifs", "images"):
            for descriptor in output.get(key, []):
                if Path(str(descriptor.get("filename", ""))).suffix.lower() in VIDEO_EXTENSIONS:
                    return descriptor
    raise PipelineError("ComfyUI 已完成，但 history 中找不到輸出影片。")


def make_job_workflow(
    template: dict[str, Any], settings: Settings, uploaded_video: str, uploaded_reference: str, job: Job
) -> dict[str, Any]:
    workflow = copy.deepcopy(template)
    nodes = settings.raw.get("workflow_nodes", {})
    video_node = str(nodes.get("video", "1"))
    reference_node = str(nodes.get("reference_image", "3"))
    save_node = str(nodes.get("save_video", "16"))
    workflow[video_node]["inputs"]["file"] = uploaded_video
    workflow[reference_node]["inputs"]["image"] = uploaded_reference
    prefix = str(settings.raw.get("comfy_output_prefix", "XYZ_Orbit_VideoModel"))
    workflow[save_node]["inputs"]["filename_prefix"] = f"{prefix}/{job.name}"
    seed_node = nodes.get("seed")
    if seed_node is not None and str(seed_node) in workflow:
        seed_input = str(settings.raw.get("seed_input", "seed"))
        base_seed = int(settings.raw.get("base_seed", 26083001))
        digest = int(hashlib.sha256(job.name.encode("utf-8")).hexdigest()[:8], 16)
        workflow[str(seed_node)]["inputs"][seed_input] = (base_seed + digest) % (2**63 - 1)
    return workflow


def generate_video(
    client: ComfyUIClient,
    template: dict[str, Any],
    settings: Settings,
    job: Job,
    uploaded_reference: str,
    destination: Path,
) -> None:
    upload_root = "xyz_orbit_video_model/" + time.strftime("%Y%m%d")
    log(f"  上傳來源影片：{job.source_video.name}")
    uploaded_video = client.upload_file(job.source_video, upload_root + "/videos")
    workflow = make_job_workflow(template, settings, uploaded_video, uploaded_reference, job)
    prompt_id = client.queue_prompt(workflow)
    log(f"  ComfyUI 已排程：{prompt_id}")
    history = client.wait_for_prompt(
        prompt_id,
        float(settings.raw.get("poll_seconds", 3)),
        int(settings.raw.get("generation_timeout_seconds", 21600)),
    )
    save_node = str(settings.raw.get("workflow_nodes", {}).get("save_video", "16"))
    descriptor = find_video_descriptor(history, save_node)
    client.download_output(descriptor, destination)
    log(f"  已下載：{destination}")


def extract_frames(video: Path, destination: Path) -> list[Path]:
    ffmpeg = ensure_command("ffmpeg")
    destination.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-vsync",
            "0",
            "-start_number",
            "0",
            str(destination / "%03d.png"),
        ],
        f"拆解影片 {video.name}",
    )
    return sorted(destination.glob("*.png"), key=lambda path: natural_key(path.name))


def encode_video_segment(source: Path, destination: Path, start_frame: int, frame_count: int) -> None:
    ffmpeg = ensure_command("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    video_filter = (
        f"trim=start_frame={start_frame}:end_frame={start_frame + frame_count},"
        "setpts=PTS-STARTPTS"
    )
    run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            video_filter,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        f"切割片段 {destination.name}",
    )


def is_path_output_complete(path_output: Path, accepted_counts: set[int]) -> bool:
    images = list((path_output / "images").glob("*.png")) if (path_output / "images").is_dir() else []
    cameras = list((path_output / "cameras").glob("*.json")) if (path_output / "cameras").is_dir() else []
    return len(images) in accepted_counts and len(images) == len(cameras) and (path_output / "cameras.json").is_file()


def build_path_output(settings: Settings, job: Job, generated_video: Path, force: bool) -> Path:
    paths_output_root = settings.output_root / "paths"
    final = paths_output_root / job.name
    if final.exists() and not force:
        if is_path_output_complete(final, settings.accepted_frame_counts):
            log(f"  已有完整逐幀輸出，略過：{final}")
            return final
        raise PipelineError(f"逐幀輸出不完整：{final}\n使用 --force 重建。")

    paths_output_root.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{job.name}_", dir=paths_output_root))
    try:
        frames = extract_frames(generated_video, temp / "images")
        frame_count = len(frames)
        if frame_count not in settings.accepted_frame_counts:
            accepted = ", ".join(str(value) for value in sorted(settings.accepted_frame_counts))
            raise PipelineError(f"{job.name} 輸出 {frame_count} 幀；允許值為 {accepted}。")
        source_cameras = camera_files(job.camera_path)
        if frame_count > len(source_cameras):
            raise PipelineError(f"{job.name} 有 {frame_count} 幀，但只有 {len(source_cameras)} 個 camera。")

        camera_output = temp / "cameras"
        camera_output.mkdir(parents=True, exist_ok=True)
        combined_frames: list[dict[str, Any]] = []
        for index in range(frame_count):
            camera = load_json(source_cameras[index])
            image_name = f"{index:03d}.png"
            camera["image"] = image_name
            write_json(camera_output / f"{index:03d}.json", camera)
            combined_frames.append(camera)
        combined_source = job.camera_path / "cameras.json"
        combined = load_json(combined_source) if combined_source.is_file() else {}
        combined["path"] = job.name
        combined["frame_count"] = frame_count
        combined["frames"] = combined_frames
        write_json(temp / "cameras.json", combined)
        shutil.copy2(generated_video, temp / "generated.mp4")

        if final.exists():
            safe_remove_tree(final, settings.output_root)
        os.replace(temp, final)
        log(f"  逐幀完成：{frame_count} 張影像 + {frame_count} 個 camera")
        return final
    except Exception:
        safe_remove_tree(temp, settings.output_root)
        raise


def rotation_matrix_to_qvec(matrix: list[list[float]]) -> tuple[float, float, float, float]:
    r00, r01, r02 = matrix[0]
    r10, r11, r12 = matrix[1]
    r20, r21, r22 = matrix[2]
    trace = r00 + r11 + r22
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        qw = 0.25 * scale
        qx = (r21 - r12) / scale
        qy = (r02 - r20) / scale
        qz = (r10 - r01) / scale
    elif r00 > r11 and r00 > r22:
        scale = math.sqrt(1.0 + r00 - r11 - r22) * 2
        qw = (r21 - r12) / scale
        qx = 0.25 * scale
        qy = (r01 + r10) / scale
        qz = (r02 + r20) / scale
    elif r11 > r22:
        scale = math.sqrt(1.0 + r11 - r00 - r22) * 2
        qw = (r02 - r20) / scale
        qx = (r01 + r10) / scale
        qy = 0.25 * scale
        qz = (r12 + r21) / scale
    else:
        scale = math.sqrt(1.0 + r22 - r00 - r11) * 2
        qw = (r10 - r01) / scale
        qx = (r02 + r20) / scale
        qy = (r12 + r21) / scale
        qz = 0.25 * scale
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    values = (qw / norm, qx / norm, qy / norm, qz / norm)
    return tuple(-value for value in values) if values[0] < 0 else values


def camera_signature(camera: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(camera["width"]),
        int(camera["height"]),
        round(float(camera["fx"]), 9),
        round(float(camera["fy"]), 9),
        round(float(camera["cx"]), 9),
        round(float(camera["cy"]), 9),
    )


def format_number(value: float) -> str:
    return f"{float(value):.17g}"


def opencv_c2w_to_opengl(matrix: list[list[float]]) -> list[list[float]]:
    """Convert OpenCV c2w (+Y down, +Z forward) to OpenGL c2w (+Y up, +Z back)."""
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise PipelineError("camera_to_world 必須是 4x4 矩陣。")
    converted = [[float(value) for value in row] for row in matrix]
    for row in range(4):
        converted[row][1] *= -1.0
        converted[row][2] *= -1.0
    return converted


def completed_path_outputs(settings: Settings) -> list[Path]:
    root = settings.output_root / "paths"
    if not root.is_dir():
        return []
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and is_path_output_complete(path, settings.accepted_frame_counts)
        ),
        key=lambda path: natural_key(path.name),
    )


def build_colmap_dataset(settings: Settings, path_outputs: Iterable[Path], force: bool) -> Path:
    dataset = settings.output_root / "colmap_dataset"
    if dataset.exists() and not force:
        images_txt = dataset / "sparse" / "0" / "images.txt"
        if images_txt.is_file():
            log(f"COLMAP 資料集已存在，重新整理：{dataset}")
        else:
            raise PipelineError(f"COLMAP 資料夾不完整：{dataset}\n使用 --force 重建。")

    settings.output_root.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".colmap_dataset_", dir=settings.output_root))
    try:
        image_root = temp / "images"
        sparse_root = temp / "sparse" / "0"
        metadata_root = temp / "metadata"
        image_root.mkdir(parents=True)
        sparse_root.mkdir(parents=True)
        metadata_root.mkdir(parents=True)
        metadata_cameras = metadata_root / "cameras"
        metadata_cameras.mkdir()

        camera_ids: dict[tuple[Any, ...], int] = {}
        cameras_by_id: dict[int, tuple[Any, ...]] = {}
        image_lines: list[str] = []
        manifest_images: list[dict[str, Any]] = []
        image_id = 1

        outputs = sorted(path_outputs, key=lambda path: natural_key(path.name))
        if not outputs:
            raise PipelineError("沒有可加入 COLMAP 的逐幀輸出。")
        for path_output in outputs:
            destination_folder = image_root / path_output.name
            destination_folder.mkdir(parents=True)
            shutil.copy2(path_output / "cameras.json", metadata_cameras / f"{path_output.name}.json")
            cameras = sorted((path_output / "cameras").glob("*.json"), key=lambda path: natural_key(path.name))
            for camera_file in cameras:
                camera = load_json(camera_file)
                source_image = path_output / "images" / str(camera["image"])
                if not source_image.is_file():
                    raise PipelineError(f"camera 找不到對應影像：{camera_file} -> {source_image}")
                destination_image = destination_folder / source_image.name
                shutil.copy2(source_image, destination_image)
                signature = camera_signature(camera)
                if signature not in camera_ids:
                    camera_id = len(camera_ids) + 1
                    camera_ids[signature] = camera_id
                    cameras_by_id[camera_id] = signature
                camera_id = camera_ids[signature]
                rotation = camera.get("rotation_matrix")
                translation = camera.get("translation")
                if rotation is None or translation is None:
                    world_to_camera = camera["world_to_camera"]
                    rotation = [row[:3] for row in world_to_camera[:3]]
                    translation = [row[3] for row in world_to_camera[:3]]
                qw, qx, qy, qz = rotation_matrix_to_qvec(rotation)
                relative_name = destination_image.relative_to(image_root).as_posix()
                values = [qw, qx, qy, qz, *translation]
                image_lines.append(
                    f"{image_id} {' '.join(format_number(value) for value in values)} {camera_id} {relative_name}\n\n"
                )
                manifest_images.append(
                    {"image_id": image_id, "camera_id": camera_id, "name": relative_name, "source_path": path_output.name}
                )
                image_id += 1

        cameras_header = (
            "# Camera list with one line of data per camera:\n"
            "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
            f"# Number of cameras: {len(cameras_by_id)}\n"
        )
        camera_lines = []
        for camera_id, signature in sorted(cameras_by_id.items()):
            width, height, fx, fy, cx, cy = signature
            params = " ".join(format_number(value) for value in (fx, fy, cx, cy))
            camera_lines.append(f"{camera_id} PINHOLE {width} {height} {params}\n")
        (sparse_root / "cameras.txt").write_text(cameras_header + "".join(camera_lines), encoding="utf-8", newline="\n")

        images_header = (
            "# Image list with two lines of data per image:\n"
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
            "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
            f"# Number of images: {len(manifest_images)}, mean observations per image: 0\n"
        )
        (sparse_root / "images.txt").write_text(images_header + "".join(image_lines), encoding="utf-8", newline="\n")
        (sparse_root / "points3D.txt").write_text(
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
            "# Number of points: 0, mean track length: 0\n",
            encoding="utf-8",
            newline="\n",
        )

        reference = settings.path("reference_image")
        shutil.copy2(reference, metadata_root / ("reference" + reference.suffix.lower()))
        source_videos = temp / "source_videos"
        source_videos.mkdir()
        for path_output in outputs:
            generated = path_output / "generated.mp4"
            if generated.is_file():
                shutil.copy2(generated, source_videos / f"{path_output.name}.mp4")
        write_json(
            metadata_root / "manifest.json",
            {
                "format": "COLMAP text model",
                "coordinate_system": "+X right, +Y down, +Z forward; world_to_camera poses",
                "image_count": len(manifest_images),
                "camera_count": len(cameras_by_id),
                "paths": [path.name for path in outputs],
                "images": manifest_images,
            },
        )

        if dataset.exists():
            safe_remove_tree(dataset, settings.output_root)
        os.replace(temp, dataset)
        log(f"COLMAP 完成：{len(manifest_images)} 張影像、{len(cameras_by_id)} 組內參")
        return dataset
    except Exception:
        safe_remove_tree(temp, settings.output_root)
        raise


def build_brush_dataset(settings: Settings, path_outputs: Iterable[Path], force: bool) -> Path:
    """Build a Brush-native Nerfstudio dataset from completed path outputs."""
    dataset = settings.output_root / "brush_dataset"
    if dataset.exists() and not force:
        transforms = dataset / "transforms.json"
        if transforms.is_file():
            log(f"Brush 資料集已存在，重新整理：{dataset}")
        else:
            raise PipelineError(f"Brush 資料夾不完整：{dataset}\n使用 --force 重建。")

    settings.output_root.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".brush_dataset_", dir=settings.output_root))
    try:
        image_root = temp / "images"
        image_root.mkdir(parents=True)
        copy_masks = bool(settings.raw.get("brush_copy_source_masks", True))
        mask_root = temp / "masks"
        if copy_masks:
            mask_root.mkdir()

        outputs = sorted(path_outputs, key=lambda path: natural_key(path.name))
        if not outputs:
            raise PipelineError("沒有可加入 Brush 的逐幀輸出。")

        frames: list[tuple[dict[str, Any], tuple[Any, ...]]] = []
        signatures: set[tuple[Any, ...]] = set()
        copied_masks = 0
        for path_output in outputs:
            destination_folder = image_root / path_output.name
            destination_folder.mkdir(parents=True)
            destination_masks = mask_root / path_output.name
            if copy_masks:
                destination_masks.mkdir(parents=True)
            cameras = sorted((path_output / "cameras").glob("*.json"), key=lambda path: natural_key(path.name))
            for camera_file in cameras:
                camera = load_json(camera_file)
                source_image = path_output / "images" / str(camera["image"])
                if not source_image.is_file():
                    raise PipelineError(f"camera 找不到對應影像：{camera_file} -> {source_image}")
                destination_image = destination_folder / source_image.name
                shutil.copy2(source_image, destination_image)
                signature = camera_signature(camera)
                signatures.add(signature)
                frame = {
                    "file_path": destination_image.relative_to(temp).as_posix(),
                    "transform_matrix": opencv_c2w_to_opengl(camera["camera_to_world"]),
                }
                frames.append((frame, signature))

                if copy_masks:
                    source_mask = settings.path("paths_dir") / path_output.name / "mask" / source_image.name
                    if source_mask.is_file():
                        shutil.copy2(source_mask, destination_masks / source_image.name)
                        copied_masks += 1

        transforms: dict[str, Any] = {
            "camera_model": "PERSPECTIVE",
            "frames": [],
        }
        if len(signatures) == 1:
            width, height, fx, fy, cx, cy = next(iter(signatures))
            transforms.update({"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "w": width, "h": height})
            transforms["frames"] = [frame for frame, _ in frames]
        else:
            for frame, signature in frames:
                width, height, fx, fy, cx, cy = signature
                frame.update({"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "w": width, "h": height})
                transforms["frames"].append(frame)
        write_json(temp / "transforms.json", transforms)

        mask_note = (
            f"Included {copied_masks} source masks in masks/.\n"
            "These masks use white for background and black for the subject.\n"
            "Enable Brush's invert-masks option so the subject is kept.\n"
            if copied_masks
            else "No masks were included.\n"
        )
        (temp / "README.txt").write_text(
            "Brush Nerfstudio dataset generated by the XYZ orbit video pipeline.\n"
            f"Images: {len(frames)}\n"
            f"Paths: {len(outputs)}\n"
            "Open this folder (or transforms.json) in Brush.\n"
            + mask_note,
            encoding="utf-8",
            newline="\n",
        )

        if dataset.exists():
            safe_remove_tree(dataset, settings.output_root)
        os.replace(temp, dataset)
        log(f"Brush 完成：{len(frames)} 張影像、{copied_masks} 張遮罩、{len(outputs)} 條路徑")
        return dataset
    except Exception:
        safe_remove_tree(temp, settings.output_root)
        raise


def scaled_camera_for_image(camera: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    source_width = int(camera["width"])
    source_height = int(camera["height"])
    scale_x = width / source_width
    scale_y = height / source_height
    scaled = copy.deepcopy(camera)
    scaled.update(
        {
            "width": width,
            "height": height,
            "fx": float(camera["fx"]) * scale_x,
            "fy": float(camera["fy"]) * scale_y,
            "cx": float(camera["cx"]) * scale_x,
            "cy": float(camera["cy"]) * scale_y,
        }
    )
    scaled["intrinsic_matrix"] = [
        [scaled["fx"], 0.0, scaled["cx"]],
        [0.0, scaled["fy"], scaled["cy"]],
        [0.0, 0.0, 1.0],
    ]
    return scaled


def consolidated_layout(settings: Settings) -> list[tuple[Path, list[Path]]]:
    paths_dir = settings.path("paths_dir")
    layout: list[tuple[Path, list[Path]]] = []
    for path_dir in sorted((path for path in paths_dir.iterdir() if path.is_dir()), key=lambda path: natural_key(path.name)):
        layout.append((path_dir, camera_files(path_dir)))
    if not layout:
        raise PipelineError(f"找不到分段用 camera paths：{paths_dir}")
    return layout


def check_consolidated_videos(settings: Settings, videos: Iterable[Path]) -> None:
    layout = consolidated_layout(settings)
    expected = sum(len(cameras) for _, cameras in layout)
    first_camera = load_json(layout[0][1][0])
    for video in videos:
        info = probe_video(video)
        count = info.get("frame_count")
        if count != expected:
            raise PipelineError(
                f"整合影片 {video.name} 有 {count or '?'} 幀，但 12 條 camera path 共需要 {expected} 幀。"
            )
        width, height = int(info["width"]), int(info["height"])
        scale_x = width / int(first_camera["width"])
        scale_y = height / int(first_camera["height"])
        log(
            f"  整合影片 {video.name}: {count} 幀，{width}x{height}，"
            f"camera scale={scale_x:g}x{scale_y:g}"
        )


def build_brush_from_consolidated_video(settings: Settings, video: Path, force: bool) -> Path:
    """Split one ordered all-view video into a self-contained Brush dataset."""
    output_parent = settings.output_root / str(settings.raw.get("consolidated_brush_output", "brush_videos"))
    dataset = output_parent / video.stem
    layout = consolidated_layout(settings)
    expected_count = sum(len(cameras) for _, cameras in layout)
    info = probe_video(video)
    frame_count = info.get("frame_count")
    width, height = int(info["width"]), int(info["height"])
    if frame_count != expected_count:
        raise PipelineError(
            f"{video.name} 有 {frame_count or '?'} 幀；camera paths 共需要 {expected_count} 幀，無法安全自動切段。"
        )

    if dataset.exists() and not force:
        transforms_path = dataset / "transforms.json"
        if transforms_path.is_file():
            transforms = load_json(transforms_path)
            if len(transforms.get("frames", [])) == expected_count:
                log(f"整合影片 Brush 資料集已完成，略過：{dataset}")
                return dataset
        raise PipelineError(f"整合影片輸出不完整：{dataset}\n使用 --force 重建。")

    output_parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{video.stem}_", dir=output_parent))
    try:
        raw_frames_dir = temp / ".raw_frames"
        raw_frames = extract_frames(video, raw_frames_dir)
        if len(raw_frames) != expected_count:
            raise PipelineError(f"{video.name} 實際拆出 {len(raw_frames)} 幀，預期 {expected_count} 幀。")

        image_root = temp / "images"
        segment_root = temp / "segments"
        frames: list[tuple[dict[str, Any], tuple[Any, ...]]] = []
        global_index = 0
        for path_index, (path_dir, cameras) in enumerate(layout, start=1):
            log(f"  [{path_index}/{len(layout)}] 分段 {path_dir.name}: {len(cameras)} 幀")
            path_images = image_root / path_dir.name
            path_images.mkdir(parents=True)
            segment_start = global_index
            for local_index, camera_file in enumerate(cameras):
                image_name = f"{local_index:03d}.png"
                destination_image = path_images / image_name
                os.replace(raw_frames[global_index], destination_image)
                camera = scaled_camera_for_image(load_json(camera_file), width, height)
                frame = {
                    "file_path": destination_image.relative_to(temp).as_posix(),
                    "transform_matrix": opencv_c2w_to_opengl(camera["camera_to_world"]),
                    "path": path_dir.name,
                    "angle_degrees": camera.get("local_path_angle_degrees", float(local_index)),
                }
                frames.append((frame, camera_signature(camera)))
                global_index += 1
            encode_video_segment(
                video,
                segment_root / f"{path_dir.name}.mp4",
                start_frame=segment_start,
                frame_count=len(cameras),
            )
        safe_remove_tree(raw_frames_dir, settings.output_root)

        signatures = {signature for _, signature in frames}
        transforms: dict[str, Any] = {"camera_model": "PERSPECTIVE", "frames": []}
        if len(signatures) == 1:
            dataset_width, dataset_height, fx, fy, cx, cy = next(iter(signatures))
            transforms.update(
                {"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "w": dataset_width, "h": dataset_height}
            )
            transforms["frames"] = [frame for frame, _ in frames]
        else:
            for frame, signature in frames:
                dataset_width, dataset_height, fx, fy, cx, cy = signature
                frame.update(
                    {"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "w": dataset_width, "h": dataset_height}
                )
                transforms["frames"].append(frame)
        write_json(temp / "transforms.json", transforms)
        shutil.copy2(video, temp / video.name)
        (temp / "README.txt").write_text(
            "Brush Nerfstudio dataset generated from one processed all-view video.\n"
            f"Source: {video.name}\n"
            f"Images: {expected_count}\n"
            f"Segments: {len(layout)}\n"
            f"Resolution: {width}x{height}\n"
            "All images were extracted from the source video. No previous output images or masks were used.\n"
            "Open this folder or transforms.json directly in Brush.\n",
            encoding="utf-8",
            newline="\n",
        )

        if dataset.exists():
            safe_remove_tree(dataset, settings.output_root)
        os.replace(temp, dataset)
        log(f"整合影片 Brush 完成：{dataset}（{expected_count} 張、{len(layout)} 段）")
        return dataset
    except Exception:
        safe_remove_tree(temp, settings.output_root)
        raise


def check(settings: Settings, jobs: list[Job], check_server: bool = True) -> None:
    ensure_command("ffmpeg")
    ensure_command("ffprobe")
    workflow = validate_workflow(settings, check_server=check_server)
    reference = settings.path("reference_image")
    if not reference.is_file() or reference.suffix.lower() not in IMAGE_EXTENSIONS:
        raise PipelineError(f"人物參考圖片無效：{reference}")
    log(f"工作流節點：{len(workflow)}")
    log(f"人物參考圖：{reference.name}")
    for job in jobs:
        video_info = probe_video(job.source_video)
        count = video_info.get("frame_count")
        cameras = camera_files(job.camera_path)
        if count is not None and count > len(cameras):
            raise PipelineError(f"{job.name}: 影片 {count} 幀，但 camera 只有 {len(cameras)} 個。")
        log(f"  {job.name}: input={count or '?'} 幀，camera={len(cameras)}")
    server_text = "已連線" if check_server else "略過"
    log(f"檢查通過：{len(jobs)} 部影片；ComfyUI {server_text}。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ComfyUI orbit video -> frames/cameras -> COLMAP + Brush datasets")
    parser.add_argument("--config", default="config.json", help="設定檔路徑（預設 config.json）")
    parser.add_argument("--check", action="store_true", help="只檢查輸入、工作流、工具與 ComfyUI")
    parser.add_argument("--offline-check", action="store_true", help="檢查時不連線 ComfyUI")
    parser.add_argument("--prepare-only", action="store_true", help="不呼叫 ComfyUI，只整理 output/generated_videos 既有影片")
    parser.add_argument("--datasets-only", action="store_true", help="只從 output/paths 重建 COLMAP 與 Brush 資料集")
    parser.add_argument("--consolidated-only", action="store_true", help="只處理 input 根目錄的整合全視角影片")
    parser.add_argument("--only", action="append", help="只處理指定檔名（可重複使用）")
    parser.add_argument("--force", action="store_true", help="重建已存在的逐幀及資料集輸出")
    parser.add_argument("--skip-colmap", action="store_true", help="完成逐幀後不建立 COLMAP 資料集")
    parser.add_argument("--skip-brush", action="store_true", help="完成逐幀後不建立 Brush 資料集")
    parser.add_argument("--skip-consolidated", action="store_true", help="不處理 input 根目錄的整合全視角影片")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    settings = load_settings(config_path)
    consolidated_videos = discover_consolidated_videos(settings)
    if args.consolidated_only:
        if not consolidated_videos:
            raise PipelineError("input 根目錄沒有整合全視角影片。")
        check_consolidated_videos(settings, consolidated_videos)
        for video in consolidated_videos:
            build_brush_from_consolidated_video(settings, video, force=args.force)
        log("整合影片全部完成。")
        return 0

    jobs = discover_jobs(settings, args.only)
    if args.check or args.offline_check:
        check(settings, jobs, check_server=not args.offline_check)
        if consolidated_videos:
            check_consolidated_videos(settings, consolidated_videos)
        return 0

    if args.datasets_only:
        all_path_outputs = completed_path_outputs(settings)
        if not all_path_outputs:
            raise PipelineError(f"找不到完整逐幀輸出：{settings.output_root / 'paths'}")
        if not args.skip_colmap:
            build_colmap_dataset(settings, all_path_outputs, force=args.force)
        if not args.skip_brush:
            build_brush_dataset(settings, all_path_outputs, force=args.force)
        log("資料集整理完成。")
        return 0

    check(settings, jobs, check_server=not args.prepare_only)
    generated_root = settings.output_root / "generated_videos"
    generated_root.mkdir(parents=True, exist_ok=True)
    template: dict[str, Any] | None = None
    client: ComfyUIClient | None = None
    uploaded_reference = ""
    if not args.prepare_only:
        template = validate_workflow(settings, check_server=False)
        client = ComfyUIClient(settings.comfy_url)
        upload_root = "xyz_orbit_video_model/" + time.strftime("%Y%m%d")
        log("上傳人物參考圖片…")
        uploaded_reference = client.upload_file(settings.path("reference_image"), upload_root + "/reference")

    path_outputs: list[Path] = []
    for index, job in enumerate(jobs, start=1):
        log(f"[{index}/{len(jobs)}] {job.name}")
        generated_video = generated_root / f"{job.name}.mp4"
        if generated_video.is_file() and not args.force:
            log("  已有生成影片，從斷點繼續。")
        elif args.prepare_only:
            raise PipelineError(f"prepare-only 找不到影片：{generated_video}")
        else:
            assert client is not None and template is not None
            generate_video(client, template, settings, job, uploaded_reference, generated_video)
        output = build_path_output(settings, job, generated_video, force=args.force)
        path_outputs.append(output)

    all_path_outputs = completed_path_outputs(settings)
    if not args.skip_colmap:
        build_colmap_dataset(settings, all_path_outputs, force=args.force)
    if not args.skip_brush:
        build_brush_dataset(settings, all_path_outputs, force=args.force)
    if not args.skip_consolidated:
        for video in consolidated_videos:
            build_brush_from_consolidated_video(settings, video, force=args.force)
    log("全部完成。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已中止；已完成的影片可在下次執行時續跑。", file=sys.stderr)
        raise SystemExit(130)
    except PipelineError as exc:
        print(f"\n錯誤：{exc}", file=sys.stderr)
        raise SystemExit(1)
