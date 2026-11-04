"""Submit or validate the local ComfyUI workflow.

Submit every discovered input job:
    python main.py

Validate only (does not generate):
    python main.py --check
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import mimetypes
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import config


@dataclass(frozen=True)
class Job:
    name: str
    image: Path
    video: Path


def api_request(
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    content_type: str | None = None,
) -> Any:
    payload = api_request_bytes(
        path, method=method, body=body, content_type=content_type
    )
    if not payload:
        return None
    return json.loads(payload.decode("utf-8"))


def api_request_bytes(
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    content_type: str | None = None,
) -> bytes:
    headers = {"Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = content_type
    request = Request(
        f"{config.COMFYUI_URL.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=config.REQUEST_TIMEOUT_SECONDS) as response:
            payload = response.read()
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI API {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"無法連線到 {config.COMFYUI_URL}；請先啟動 ComfyUI。原因：{exc.reason}"
        ) from exc
    return payload


def load_workflow() -> dict[str, Any]:
    if not config.WORKFLOW_PATH.is_file():
        raise RuntimeError(f"找不到 API 工作流：{config.WORKFLOW_PATH}")
    try:
        workflow = json.loads(config.WORKFLOW_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"工作流 JSON 無法讀取：{exc}") from exc
    if not isinstance(workflow, dict) or not workflow:
        raise RuntimeError("API 工作流必須是非空的 JSON 物件。")
    return workflow


def discover_jobs() -> list[Job]:
    if not config.INPUT_DIR.is_dir():
        raise RuntimeError(f"找不到 input 資料夾：{config.INPUT_DIR}")
    image_filename = getattr(config, "INPUT_IMAGE_FILENAME", "")
    video_filename = getattr(config, "INPUT_VIDEO_FILENAME", "")
    if not isinstance(image_filename, str) or Path(image_filename).name != image_filename:
        raise RuntimeError("config.py 的 INPUT_IMAGE_FILENAME 必須是單一檔名。")
    if not isinstance(video_filename, str) or Path(video_filename).name != video_filename:
        raise RuntimeError("config.py 的 INPUT_VIDEO_FILENAME 必須是單一檔名。")

    jobs: list[Job] = []
    incomplete: list[str] = []
    folders = sorted(
        (path for path in config.INPUT_DIR.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    for folder in folders:
        image = folder / image_filename
        video = folder / video_filename
        has_image = image.is_file()
        has_video = video.is_file()
        if has_image and has_video:
            jobs.append(Job(name=folder.name, image=image, video=video))
        elif has_image or has_video:
            missing = video_filename if has_image else image_filename
            incomplete.append(f"input/{folder.name} 缺少 {missing}")

    if incomplete:
        raise RuntimeError("輸入資料不完整：\n" + "\n".join(incomplete))
    if not jobs:
        raise RuntimeError(
            "input 內找不到完整工作；每個子資料夾需同時包含 "
            f"{image_filename} 與 {video_filename}。"
        )
    return jobs


def workflow_nodes(workflow: dict[str, Any], class_type: str) -> list[dict[str, Any]]:
    return [node for node in workflow.values() if node.get("class_type") == class_type]


def validate_workflow(workflow: dict[str, Any]) -> set[str]:
    required_counts = {"LoadImage": 1, "LoadVideo": 1}
    for class_type, expected in required_counts.items():
        actual = len(workflow_nodes(workflow, class_type))
        if actual != expected:
            raise RuntimeError(
                f"工作流需有 {expected} 個 {class_type} 節點，目前找到 {actual} 個。"
            )
    if not workflow_nodes(workflow, "SaveVideo"):
        raise RuntimeError("工作流至少需要 1 個 SaveVideo 節點。")
    return {
        node["class_type"]
        for node in workflow.values()
        if isinstance(node, dict) and isinstance(node.get("class_type"), str)
    }


def check_server(required_classes: set[str]) -> None:
    api_request("/system_stats")
    object_info = api_request("/object_info")
    if not isinstance(object_info, dict):
        raise RuntimeError("ComfyUI /object_info 回傳格式不正確。")
    missing = sorted(required_classes - set(object_info))
    if missing:
        raise RuntimeError("ComfyUI 缺少工作流節點：" + ", ".join(missing))


def multipart_file(path: Path) -> tuple[bytes, str]:
    boundary = f"----ComfyUIBoundary{uuid.uuid4().hex}"
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts = [
        f"--{boundary}\r\n".encode(),
        (
            f'Content-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode("utf-8"),
        path.read_bytes(),
        b"\r\n",
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="type"\r\n\r\ninput\r\n',
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="overwrite"\r\n\r\nfalse\r\n',
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), boundary


def upload_input(path: Path) -> str:
    body, boundary = multipart_file(path)
    result = api_request(
        "/upload/image",
        method="POST",
        body=body,
        content_type=f"multipart/form-data; boundary={boundary}",
    )
    if not isinstance(result, dict) or not result.get("name"):
        raise RuntimeError(f"上傳 {path.name} 時收到非預期回應：{result!r}")
    subfolder = str(result.get("subfolder", "")).strip("/\\")
    return f"{subfolder}/{result['name']}" if subfolder else str(result["name"])


def prepare_workflow(base: dict[str, Any], job: Job) -> tuple[dict[str, Any], set[str]]:
    workflow = copy.deepcopy(base)
    image_name = upload_input(job.image)
    video_name = upload_input(job.video)
    workflow_nodes(workflow, "LoadImage")[0]["inputs"]["image"] = image_name
    workflow_nodes(workflow, "LoadVideo")[0]["inputs"]["file"] = video_name
    output_node_ids: set[str] = set()
    output_index = 0
    for node_id, node in workflow.items():
        if node.get("class_type") != "SaveVideo":
            continue
        output_index += 1
        output_node_ids.add(str(node_id))
        node["inputs"]["filename_prefix"] = (
            f"{config.OUTPUT_PREFIX}_{job.name}_output_{output_index}"
        )
    return workflow, output_node_ids


def queue_workflow(workflow: dict[str, Any]) -> str:
    body = json.dumps({"prompt": workflow}).encode("utf-8")
    result = api_request(
        "/prompt",
        method="POST",
        body=body,
        content_type="application/json",
    )
    if not isinstance(result, dict) or not result.get("prompt_id"):
        raise RuntimeError(f"工作流未成功排入佇列：{result!r}")
    return str(result["prompt_id"])


def wait_for_history(prompt_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + config.GENERATION_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        history = api_request(f"/history/{prompt_id}")
        if isinstance(history, dict) and prompt_id in history:
            record = history[prompt_id]
            status = record.get("status", {})
            if status.get("completed") is False:
                messages = status.get("messages", [])
                raise RuntimeError(f"ComfyUI 生成失敗：{messages!r}")
            return record
        time.sleep(config.POLL_INTERVAL_SECONDS)
    raise RuntimeError(
        f"等待生成超過 {config.GENERATION_TIMEOUT_SECONDS} 秒，prompt_id={prompt_id}"
    )


def find_output_files(record: dict[str, Any], node_ids: set[str]) -> list[dict[str, str]]:
    outputs = record.get("outputs", {})
    found: list[dict[str, str]] = []
    for node_id in node_ids:
        node_output = outputs.get(node_id, {})
        if not isinstance(node_output, dict):
            continue
        for value in node_output.values():
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, dict) and item.get("filename"):
                    found.append(
                        {
                            "filename": str(item["filename"]),
                            "subfolder": str(item.get("subfolder", "")),
                            "type": str(item.get("type", "output")),
                        }
                    )
    if not found:
        raise RuntimeError("生成已完成，但歷史紀錄中找不到 SaveVideo 輸出檔案。")
    return found


def unique_destination(folder: Path, filename: str) -> Path:
    destination = folder / Path(filename).name
    if not destination.exists():
        return destination
    stem, suffix = destination.stem, destination.suffix
    index = 2
    while True:
        candidate = folder / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def job_signature(workflow: dict[str, Any], job: Job) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(workflow, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(file_digest(job.image).encode("ascii"))
    digest.update(file_digest(job.video).encode("ascii"))
    return digest.hexdigest()


def completion_marker(job: Job) -> Path:
    return config.OUTPUT_DIR / job.name / ".completed.json"


def job_is_complete(job: Job, signature: str) -> bool:
    """Return True only for the same workflow and unchanged input files."""
    job_output_dir = config.OUTPUT_DIR / job.name
    marker = completion_marker(job)
    if not marker.is_file():
        return False
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if record.get("signature") != signature:
        return False
    output_names = record.get("outputs")
    return bool(output_names) and all(
        isinstance(name, str) and (job_output_dir / name).is_file()
        for name in output_names
    )


def mark_job_complete(job: Job, signature: str, saved: list[Path]) -> None:
    marker = completion_marker(job)
    marker.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "signature": signature,
        "outputs": [path.name for path in saved],
    }
    marker.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def download_outputs(
    prompt_id: str, job: Job, output_node_ids: set[str]
) -> list[Path]:
    record = wait_for_history(prompt_id)
    files = find_output_files(record, output_node_ids)
    job_output_dir = config.OUTPUT_DIR / job.name
    job_output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for item in files:
        query = urlencode(item)
        content = api_request_bytes(f"/view?{query}")
        destination = unique_destination(job_output_dir, item["filename"])
        destination.write_bytes(content)
        saved.append(destination)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="驗證或執行本地 ComfyUI 工作流")
    parser.add_argument(
        "--check",
        action="store_true",
        help="只驗證設定、輸入檔案與 ComfyUI 節點，不執行生成",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        workflow = load_workflow()
        jobs = discover_jobs()
        required_classes = validate_workflow(workflow)
        check_server(required_classes)
        print(f"驗證通過：ComfyUI 可連線，節點完整，共找到 {len(jobs)} 組輸入。")
        signatures = {job.name: job_signature(workflow, job) for job in jobs}
        for job in jobs:
            state = "已完成，將跳過" if job_is_complete(job, signatures[job.name]) else "待處理"
            print(f"  - {job.name}: {job.image.name} + {job.video.name} [{state}]")

        if args.check:
            print("僅完成驗證，未執行生成。")
            return 0

        for job in jobs:
            signature = signatures[job.name]
            if job_is_complete(job, signature):
                print(f"跳過已完成工作：{job.name}")
                continue
            prepared, output_node_ids = prepare_workflow(workflow, job)
            prompt_id = queue_workflow(prepared)
            print(f"已排入佇列：{job.name}，prompt_id={prompt_id}；等待完成……")
            saved = download_outputs(prompt_id, job, output_node_ids)
            for path in saved:
                print(f"已儲存：{path}")
            mark_job_complete(job, signature, saved)
        return 0
    except (RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
