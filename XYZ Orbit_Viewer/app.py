from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import re
import threading
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT_DIR / "input"
DEFAULT_DIST_DIR = ROOT_DIR / "frontend" / "dist"
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.json"
CHUNK_SIZE = 1024 * 1024

DEFAULT_VIEWER_CONFIG: dict[str, Any] = {
    "path_playback": {"smart_reverse": True},
    "image_mode": {
        "drag_degrees_per_pixel": 0.24,
        "keyboard_step_degrees": 1.0,
    },
    "ply_mode": {
        "camera_point_size_px": 3.0,
        "camera_point_opacity": 0.8,
        "camera_orbit_scale": 1.0,
        "path_opacity": 0.42,
        "model_scale": 1.0,
        "model_scale_min": 0.25,
        "model_scale_max": 4.0,
        "model_scale_step": 0.05,
        "axis_length": 1.8726598549,
        "axis_offset": [0.0, 0.0, 0.0],
        "axis_opacity": 0.9,
        "selected_camera_marker_size_px": 26.0,
        "selected_ray_opacity": 0.48,
        "overview_distance_multiplier": 2.55,
        "overview_fov_degrees": 54.0,
        "focus_distance_multiplier": 1.18,
        "focus_fov_degrees": 30.0,
        "min_zoom_distance_multiplier": 0.48,
        "max_zoom_distance_multiplier": 2.6,
        "camera_point_color": "#737d89",
        "path_color": "#59626e",
        "selected_camera_color": "#ffc857",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_viewer_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        return copy.deepcopy(DEFAULT_VIEWER_CONFIG)
    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"config.json 最外層必須是 JSON 物件：{config_path}")
    config = _deep_merge(DEFAULT_VIEWER_CONFIG, loaded)
    axis_offset = config["ply_mode"].get("axis_offset")
    if (
        not isinstance(axis_offset, list)
        or len(axis_offset) != 3
        or not all(isinstance(value, (int, float)) for value in axis_offset)
    ):
        raise ValueError("ply_mode.axis_offset 必須是三個數字，例如 [0, 0, 0]")
    return config


def _asset_url(relative_path: Path) -> str:
    return "/input/" + "/".join(relative_path.as_posix().split("/"))


def _read_ply_metadata(ply_path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "format": "unknown",
        "vertex_count": None,
        "vertical_axis": None,
        "sh_degree": None,
    }
    with ply_path.open("rb") as stream:
        for _ in range(512):
            raw_line = stream.readline(4096)
            if not raw_line:
                break
            line = raw_line.decode("ascii", errors="replace").strip()
            if line.startswith("format "):
                metadata["format"] = line.split()[1]
            elif line.startswith("element vertex "):
                metadata["vertex_count"] = int(line.split()[-1])
            elif line.lower().startswith("comment vertical axis:"):
                metadata["vertical_axis"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("comment sh degree:"):
                metadata["sh_degree"] = int(line.split(":", 1)[1].strip())
            elif line == "end_header":
                break
    return metadata


def _compact_frame(
    frame: dict[str, Any],
    edge_id: str,
    path_directory: Path,
    frame_index: int,
) -> dict[str, Any]:
    image_name = str(frame.get("image", f"{frame_index:03d}.png"))
    image_relative = path_directory / "images" / image_name
    return {
        "id": f"{edge_id}:{frame_index:03d}",
        "frame_index": frame_index,
        "angle_degrees": float(frame.get("local_path_angle_degrees", frame_index)),
        "image_url": _asset_url(image_relative),
        "camera_position": frame["camera_position"],
        "camera_forward": frame.get("camera_forward"),
        "camera_up": frame.get("camera_up"),
    }


@dataclass(slots=True)
class DatasetIndex:
    input_dir: Path
    config_path: Path
    manifest: dict[str, Any]
    manifest_bytes: bytes
    config_lock: threading.RLock = field(default_factory=threading.RLock)

    @classmethod
    def load(cls, input_dir: Path, config_path: Path | None = None) -> "DatasetIndex":
        input_dir = input_dir.resolve()
        config_path = (config_path or input_dir.parent / "config.json").resolve()
        graph_path = input_dir / "path_graph.json"
        if not graph_path.is_file():
            raise FileNotFoundError(f"找不到路徑定義：{graph_path}")

        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        edges = graph.get("edges", [])
        if not edges:
            raise ValueError("path_graph.json 沒有任何路徑")

        paths: list[dict[str, Any]] = []
        warnings: list[str] = []
        subject_center: list[float] | None = None
        sphere_radius: float | None = None
        image_size: list[int] | None = None

        for edge in edges:
            edge_id = edge["id"]
            path_directory = Path(edge["directory"])
            cameras_path = input_dir / path_directory / "cameras.json"
            if not cameras_path.is_file():
                raise FileNotFoundError(f"缺少相機資料：{cameras_path}")

            camera_bundle = json.loads(cameras_path.read_text(encoding="utf-8"))
            camera_frames = camera_bundle.get("frames", [])
            if not camera_frames:
                raise ValueError(f"{edge_id} 沒有相機影格")

            compact_frames: list[dict[str, Any]] = []
            for index, frame in enumerate(camera_frames):
                compact = _compact_frame(frame, edge_id, path_directory, index)
                image_path = input_dir / compact["image_url"].removeprefix("/input/")
                if not image_path.is_file():
                    warnings.append(f"缺少影像：{compact['image_url']}")
                compact_frames.append(compact)

                if subject_center is None:
                    subject_center = [float(value) for value in frame["subject_center"]]
                    sphere_radius = float(frame["sphere_radius"])
                    image_size = [int(frame["width"]), int(frame["height"])]

            declared_count = int(edge.get("frame_count", len(compact_frames)))
            if declared_count != len(compact_frames):
                warnings.append(
                    f"{edge_id} 宣告 {declared_count} 張，但相機資料有 {len(compact_frames)} 張"
                )

            paths.append(
                {
                    "id": edge_id,
                    "start_anchor": edge["start_anchor"],
                    "end_anchor": edge["end_anchor"],
                    "degrees_per_frame": float(edge.get("degrees_per_frame", 1.0)),
                    "frame_count": len(compact_frames),
                    "frames": compact_frames,
                }
            )

        if subject_center is None or sphere_radius is None:
            raise ValueError("無法從相機資料取得主體中心與軌道半徑")

        anchors: list[dict[str, Any]] = []
        for node in graph.get("nodes", []):
            camera_path = input_dir / Path(node["camera"])
            if not camera_path.is_file():
                warnings.append(f"缺少錨點相機：{node['camera']}")
                continue
            camera = json.loads(camera_path.read_text(encoding="utf-8"))
            anchors.append(
                {
                    "id": node["id"],
                    "camera_position": camera["camera_position"],
                    "camera_forward": camera.get("camera_forward"),
                    "camera_up": camera.get("camera_up"),
                    "image_url": _asset_url(Path(node["image"])),
                }
            )

        ply_files = sorted(input_dir.glob("*.ply"), key=lambda path: path.name.lower())
        if not ply_files:
            ply_files = sorted(input_dir.rglob("*.ply"), key=lambda path: path.name.lower())
        if not ply_files:
            raise FileNotFoundError(f"{input_dir} 內找不到 PLY 檔案")
        ply_path = max(ply_files, key=lambda path: path.stat().st_size)
        ply_relative = ply_path.relative_to(input_dir)
        ply_metadata = _read_ply_metadata(ply_path)

        total_samples = sum(path["frame_count"] for path in paths)
        frame_counts = sorted({path["frame_count"] for path in paths})
        manifest: dict[str, Any] = {
            "dataset_name": input_dir.parent.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "subject_center": subject_center,
            "sphere_radius": sphere_radius,
            "image_size": image_size,
            "model": {
                "name": ply_path.name,
                "url": _asset_url(ply_relative),
                "size_bytes": ply_path.stat().st_size,
                **ply_metadata,
            },
            "anchors": anchors,
            "paths": paths,
            "stats": {
                "path_count": len(paths),
                "total_camera_samples": total_samples,
                "frame_counts": frame_counts,
            },
            "warnings": warnings[:100],
            "config": load_viewer_config(config_path),
        }
        manifest_bytes = json.dumps(
            manifest, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return cls(
            input_dir=input_dir,
            config_path=config_path,
            manifest=manifest,
            manifest_bytes=manifest_bytes,
        )

    def refresh_config(self) -> None:
        with self.config_lock:
            self.manifest["config"] = load_viewer_config(self.config_path)
            self.manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
            self.manifest_bytes = json.dumps(
                self.manifest, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")


def _safe_child(root: Path, request_path: str) -> Path | None:
    candidate = (root / request_path.lstrip("/")).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


class ViewerRequestHandler(BaseHTTPRequestHandler):
    server_version = "XYZOrbitViewer/1.0"

    @property
    def viewer_server(self) -> "ViewerHTTPServer":
        return self.server  # type: ignore[return-value]

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}")

    def do_HEAD(self) -> None:
        self._route(send_body=False)

    def do_GET(self) -> None:
        self._route(send_body=True)

    def _route(self, send_body: bool) -> None:
        request_path = unquote(urlparse(self.path).path)

        if request_path == "/api/health":
            self._send_json(
                {
                    "status": "ok",
                    "paths": self.viewer_server.dataset.manifest["stats"]["path_count"],
                    "samples": self.viewer_server.dataset.manifest["stats"][
                        "total_camera_samples"
                    ],
                },
                send_body,
            )
            return
        if request_path == "/api/manifest":
            try:
                self.viewer_server.dataset.refresh_config()
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self._send_json(
                    {"error": f"config.json 無法讀取：{error}"},
                    send_body,
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            self._send_bytes(
                self.viewer_server.dataset.manifest_bytes,
                "application/json; charset=utf-8",
                send_body,
                cache_control="no-cache",
            )
            return
        if request_path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return

        if request_path.startswith("/input/"):
            relative = request_path.removeprefix("/input/")
            file_path = _safe_child(self.viewer_server.dataset.input_dir, relative)
            if file_path is None:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self._send_file(file_path, send_body, cache_control="public, max-age=3600")
            return

        if request_path.startswith("/static/"):
            relative = request_path.removeprefix("/static/")
            file_path = _safe_child(self.viewer_server.dist_dir, relative)
            if file_path is None:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self._send_file(
                file_path,
                send_body,
                cache_control="public, max-age=31536000, immutable",
            )
            return

        if request_path in {"/", "/index.html"}:
            index_path = self.viewer_server.dist_dir / "index.html"
            if not index_path.is_file():
                message = (
                    "前端尚未建置。請在專案資料夾執行 npm install 與 npm run build。"
                ).encode("utf-8")
                self._send_bytes(
                    message,
                    "text/plain; charset=utf-8",
                    send_body,
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            self._send_file(index_path, send_body, cache_control="no-cache")
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def _send_json(
        self,
        payload: dict[str, Any],
        send_body: bool,
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(
            body,
            "application/json; charset=utf-8",
            send_body,
            cache_control="no-cache",
            status=status,
        )

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        send_body: bool,
        *,
        cache_control: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def _send_file(self, file_path: Path, send_body: bool, *, cache_control: str) -> None:
        if not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        stat = file_path.stat()
        etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", cache_control)
            self.end_headers()
            return

        start = 0
        end = stat.st_size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            start_text, end_text = match.groups()
            if start_text:
                start = int(start_text)
                end = int(end_text) if end_text else end
            elif end_text:
                suffix_length = int(end_text)
                start = max(0, stat.st_size - suffix_length)
            if start > end or start >= stat.st_size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{stat.st_size}")
                self.end_headers()
                return
            end = min(end, stat.st_size - 1)
            status = HTTPStatus.PARTIAL_CONTENT

        content_length = end - start + 1
        guessed_type, guessed_encoding = mimetypes.guess_type(file_path.name)
        content_type = guessed_type or "application/octet-stream"

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if guessed_encoding:
            self.send_header("Content-Encoding", guessed_encoding)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", cache_control)
        self.send_header("ETag", etag)
        self.send_header("X-Content-Type-Options", "nosniff")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{stat.st_size}")
        self.end_headers()

        if not send_body:
            return
        try:
            with file_path.open("rb") as stream:
                stream.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk = stream.read(min(CHUNK_SIZE, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


class ViewerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        dataset: DatasetIndex,
        dist_dir: Path,
    ) -> None:
        super().__init__(address, ViewerRequestHandler)
        self.dataset = dataset
        self.dist_dir = dist_dir.resolve()


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    dist_dir: Path = DEFAULT_DIST_DIR,
    config_path: Path | None = None,
) -> ViewerHTTPServer:
    dataset = DatasetIndex.load(input_dir, config_path)
    return ViewerHTTPServer((host, port), dataset, dist_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XYZ Orbit Viewer 本機伺服器")
    parser.add_argument("--host", default="127.0.0.1", help="監聽位址")
    parser.add_argument("--port", type=int, default=8000, help="監聽連接埠")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Viewer 顯示設定 JSON",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="資料集 input 資料夾",
    )
    parser.add_argument(
        "--open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="啟動後開啟瀏覽器",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = create_server(
        args.host,
        args.port,
        input_dir=args.input,
        config_path=args.config,
    )
    actual_host, actual_port = server.server_address[:2]
    browser_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
    url = f"http://{browser_host}:{actual_port}/"
    stats = server.dataset.manifest["stats"]
    model = server.dataset.manifest["model"]

    print("XYZ Orbit Viewer 已就緒")
    print(f"資料：{stats['path_count']} 條路徑，{stats['total_camera_samples']} 個相機位置")
    print(f"PLY：{model['name']}，{model.get('vertex_count') or '?'} 個 splats")
    print(f"網址：{url}")
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer 已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
