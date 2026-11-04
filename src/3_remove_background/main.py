"""批次呼叫 ComfyUI 去除 input 影片背景。

執行全部工作：
    python main.py

只驗證設定、輸入與 ComfyUI：
    python main.py --check

重新產生已存在的輸出：
    python main.py --force
"""

from __future__ import annotations

import argparse
import copy
import http.client
import json
import mimetypes
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

import config


@dataclass(frozen=True)
class Job:
    source: Path
    relative_path: Path

    @property
    def output_folder(self) -> Path:
        return config.OUTPUT_DIR / self.relative_path.parent


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
            return response.read()
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI API {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"無法連線到 {config.COMFYUI_URL}；請先啟動 ComfyUI。原因：{exc.reason}"
        ) from exc


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
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ComfyUI 回傳無效 JSON：{exc}") from exc


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
    config.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extensions = {extension.casefold() for extension in config.VIDEO_EXTENSIONS}
    videos = sorted(
        (
            path
            for path in config.INPUT_DIR.rglob("*")
            if path.is_file() and path.suffix.casefold() in extensions
        ),
        key=lambda path: str(path.relative_to(config.INPUT_DIR)).casefold(),
    )
    if not videos:
        raise RuntimeError(
            f"{config.INPUT_DIR} 內找不到影片。支援格式："
            + ", ".join(sorted(extensions))
        )
    return [
        Job(source=path, relative_path=path.relative_to(config.INPUT_DIR))
        for path in videos
    ]


def workflow_nodes(
    workflow: dict[str, Any], class_type: str
) -> list[tuple[str, dict[str, Any]]]:
    return [
        (str(node_id), node)
        for node_id, node in workflow.items()
        if isinstance(node, dict) and node.get("class_type") == class_type
    ]


def validate_workflow(workflow: dict[str, Any]) -> set[str]:
    load_nodes = workflow_nodes(workflow, "VHS_LoadVideo")
    if len(load_nodes) != 1:
        raise RuntimeError(
            "工作流需有 1 個 VHS_LoadVideo 節點，"
            f"目前找到 {len(load_nodes)} 個。"
        )
    if "video" not in load_nodes[0][1].get("inputs", {}):
        raise RuntimeError("VHS_LoadVideo 節點缺少 video 輸入欄位。")
    save_nodes = workflow_nodes(workflow, "SaveVideo")
    if not save_nodes:
        raise RuntimeError("工作流至少需要 1 個 SaveVideo 節點。")
    for node_id, node in save_nodes:
        if "filename_prefix" not in node.get("inputs", {}):
            raise RuntimeError(f"SaveVideo 節點 {node_id} 缺少 filename_prefix。")
    return {
        str(node["class_type"])
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


def _connection() -> tuple[http.client.HTTPConnection, str]:
    parsed = urlsplit(config.COMFYUI_URL)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError(f"COMFYUI_URL 格式不正確：{config.COMFYUI_URL}")
    connection_class = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_class(
        parsed.hostname,
        parsed.port,
        timeout=config.UPLOAD_TIMEOUT_SECONDS,
    )
    base_path = parsed.path.rstrip("/")
    return connection, f"{base_path}/upload/image"


def upload_input(path: Path) -> str:
    """以串流 multipart 上傳，避免把大型影片整個載入記憶體。"""
    boundary = f"----ComfyUIBoundary{uuid.uuid4().hex}"
    safe_name = path.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{safe_name}"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode("utf-8")
    suffix = (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="type"\r\n\r\n'
        "input\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
        "false\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    content_length = len(prefix) + path.stat().st_size + len(suffix)
    connection, endpoint = _connection()
    try:
        connection.putrequest("POST", endpoint)
        connection.putheader("Accept", "application/json")
        connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        connection.putheader("Content-Length", str(content_length))
        connection.endheaders()
        connection.send(prefix)
        with path.open("rb") as source:
            while chunk := source.read(config.UPLOAD_CHUNK_SIZE):
                connection.send(chunk)
        connection.send(suffix)
        response = connection.getresponse()
        payload = response.read()
    except (OSError, http.client.HTTPException) as exc:
        raise RuntimeError(f"上傳 {path} 失敗：{exc}") from exc
    finally:
        connection.close()
    if not 200 <= response.status < 300:
        details = payload.decode("utf-8", errors="replace")
        raise RuntimeError(f"上傳 {path.name} 失敗，HTTP {response.status}: {details}")
    try:
        result = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"上傳 {path.name} 時收到無效 JSON。") from exc
    if not isinstance(result, dict) or not result.get("name"):
        raise RuntimeError(f"上傳 {path.name} 時收到非預期回應：{result!r}")
    subfolder = str(result.get("subfolder", "")).strip("/\\")
    return f"{subfolder}/{result['name']}" if subfolder else str(result["name"])


def prepare_workflow(
    base: dict[str, Any], job: Job
) -> tuple[dict[str, Any], set[str]]:
    workflow = copy.deepcopy(base)
    uploaded_name = upload_input(job.source)
    workflow_nodes(workflow, "VHS_LoadVideo")[0][1]["inputs"]["video"] = uploaded_name
    output_node_ids: set[str] = set()
    run_id = uuid.uuid4().hex
    for output_index, (node_id, node) in enumerate(
        workflow_nodes(workflow, "SaveVideo"), start=1
    ):
        output_node_ids.add(node_id)
        node["inputs"]["filename_prefix"] = (
            f"{config.COMFYUI_OUTPUT_PREFIX}/{run_id}_{output_index}"
        )
    return workflow, output_node_ids


def queue_workflow(workflow: dict[str, Any]) -> str:
    body = json.dumps({"prompt": workflow}).encode("utf-8")
    result = api_request(
        "/prompt", method="POST", body=body, content_type="application/json"
    )
    if not isinstance(result, dict) or not result.get("prompt_id"):
        raise RuntimeError(f"工作流未成功排入佇列：{result!r}")
    return str(result["prompt_id"])


def wait_for_history(prompt_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + config.GENERATION_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        history = api_request(f"/history/{quote(prompt_id, safe='')}")
        if isinstance(history, dict) and prompt_id in history:
            record = history[prompt_id]
            status = record.get("status", {})
            if status.get("completed") is False or status.get("status_str") == "error":
                raise RuntimeError(
                    f"ComfyUI 生成失敗：{status.get('messages', [])!r}"
                )
            return record
        time.sleep(config.POLL_INTERVAL_SECONDS)
    raise RuntimeError(
        f"等待生成超過 {config.GENERATION_TIMEOUT_SECONDS} 秒，prompt_id={prompt_id}"
    )


def find_output_files(
    record: dict[str, Any], node_ids: set[str]
) -> list[dict[str, str]]:
    outputs = record.get("outputs", {})
    found: list[dict[str, str]] = []
    for node_id in sorted(node_ids):
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


def output_candidates(job: Job) -> list[Path]:
    if not job.output_folder.is_dir():
        return []
    extensions = {extension.casefold() for extension in config.VIDEO_EXTENSIONS}
    return sorted(
        path
        for path in job.output_folder.glob(f"{job.relative_path.stem}.*")
        if path.is_file() and path.suffix.casefold() in extensions
    )


def download_outputs(
    prompt_id: str, job: Job, output_node_ids: set[str]
) -> list[Path]:
    record = wait_for_history(prompt_id)
    files = find_output_files(record, output_node_ids)
    job.output_folder.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for index, item in enumerate(files, start=1):
        content = api_request_bytes(f"/view?{urlencode(item)}")
        generated_suffix = Path(item["filename"]).suffix or job.source.suffix
        name = job.relative_path.stem
        if len(files) > 1:
            name += f"_output_{index}"
        destination = job.output_folder / f"{name}{generated_suffix}"
        destination.write_bytes(content)
        saved.append(destination)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批次執行 ComfyUI 去背工作流")
    parser.add_argument(
        "--check",
        action="store_true",
        help="只驗證設定、輸入檔案與 ComfyUI 節點，不執行生成",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使 output 已有同名影片也重新生成並覆寫",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        workflow = load_workflow()
        jobs = discover_jobs()
        required_classes = validate_workflow(workflow)
        check_server(required_classes)
        print(f"驗證通過：ComfyUI 可連線，節點完整，共找到 {len(jobs)} 部影片。")
        for job in jobs:
            state = "已完成，將跳過" if output_candidates(job) and not args.force else "待處理"
            print(f"  - {job.relative_path} [{state}]")

        if args.check:
            print("僅完成驗證，未執行生成。")
            return 0

        for index, job in enumerate(jobs, start=1):
            if output_candidates(job) and not args.force:
                print(f"[{index}/{len(jobs)}] 跳過已完成：{job.relative_path}")
                continue
            print(f"[{index}/{len(jobs)}] 上傳：{job.relative_path}")
            prepared, output_node_ids = prepare_workflow(workflow, job)
            prompt_id = queue_workflow(prepared)
            print(f"[{index}/{len(jobs)}] 已排入佇列：prompt_id={prompt_id}；等待完成……")
            saved = download_outputs(prompt_id, job, output_node_ids)
            for path in saved:
                print(f"[{index}/{len(jobs)}] 已儲存：{path}")
        return 0
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
