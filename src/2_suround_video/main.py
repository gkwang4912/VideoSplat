"""Submit or validate the local ComfyUI workflow.

Submit every configured input job:
    python main.py

Validate only (does not generate):
    python main.py --check
"""

from __future__ import annotations

import argparse
import copy
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
    input_root = config.INPUT_DIR.resolve()
    configured_jobs = getattr(config, "INPUT_JOBS", None)
    if not isinstance(configured_jobs, list) or not configured_jobs:
        raise RuntimeError("config.py 的 INPUT_JOBS 必須至少設定一組輸入。")

    def resolve_input(relative_path: object, label: str) -> Path:
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise RuntimeError(f"{label} 必須是相對於 input 的非空路徑字串。")
        configured_path = Path(relative_path)
        if configured_path.is_absolute():
            raise RuntimeError(f"{label} 不可使用絕對路徑：{relative_path}")
        resolved = (input_root / configured_path).resolve()
        if not resolved.is_relative_to(input_root):
            raise RuntimeError(f"{label} 不可指向 input 資料夾外：{relative_path}")
        if not resolved.is_file():
            raise RuntimeError(f"{label} 找不到檔案：input/{relative_path}")
        return resolved

    jobs: list[Job] = []
    names: set[str] = set()
    for index, item in enumerate(configured_jobs, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(f"INPUT_JOBS 第 {index} 組必須是字典設定。")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise RuntimeError(f"INPUT_JOBS 第 {index} 組缺少有效的 name。")
        name = name.strip()
        if name in names:
            raise RuntimeError(f"INPUT_JOBS 的 name 不可重複：{name}")
        if Path(name).name != name:
            raise RuntimeError(f"INPUT_JOBS 的 name 不可包含路徑字元：{name}")

        image = resolve_input(item.get("image"), f"INPUT_JOBS[{index}] image")
        video = resolve_input(item.get("video"), f"INPUT_JOBS[{index}] video")
        if image.suffix.lower() not in config.IMAGE_EXTENSIONS:
            raise RuntimeError(f"指定的圖片副檔名不支援：{image.name}")
        if video.suffix.lower() not in config.VIDEO_EXTENSIONS:
            raise RuntimeError(f"指定的影片副檔名不支援：{video.name}")
        names.add(name)
        jobs.append(Job(name=name, image=image, video=video))
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


def job_is_complete(job: Job, expected_output_count: int) -> bool:
    """Return True only when every expected numbered output already exists."""
    job_output_dir = config.OUTPUT_DIR / job.name
    if not job_output_dir.is_dir():
        return False
    prefix = Path(config.OUTPUT_PREFIX).name
    for output_index in range(1, expected_output_count + 1):
        pattern = f"{prefix}_{job.name}_output_{output_index}_*"
        if not any(path.is_file() for path in job_output_dir.glob(pattern)):
            return False
    return True


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
        expected_output_count = len(workflow_nodes(workflow, "SaveVideo"))
        check_server(required_classes)
        print(f"驗證通過：ComfyUI 可連線，節點完整，共找到 {len(jobs)} 組輸入。")
        for job in jobs:
            state = "已完成，將跳過" if job_is_complete(job, expected_output_count) else "待處理"
            print(f"  - {job.name}: {job.image.name} + {job.video.name} [{state}]")

        if args.check:
            print("僅完成驗證，未執行生成。")
            return 0

        for job in jobs:
            if job_is_complete(job, expected_output_count):
                print(f"跳過已完成工作：{job.name}")
                continue
            prepared, output_node_ids = prepare_workflow(workflow, job)
            prompt_id = queue_workflow(prepared)
            print(f"已排入佇列：{job.name}，prompt_id={prompt_id}；等待完成……")
            saved = download_outputs(prompt_id, job, output_node_ids)
            for path in saved:
                print(f"已儲存：{path}")
        return 0
    except (RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
