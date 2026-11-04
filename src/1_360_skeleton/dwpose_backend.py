"""DWPose 模型介面。模型與第三方程式都放在專案內，不污染系統 Python。"""

from __future__ import annotations

import os
import hashlib
import importlib
import sys
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent
VENDOR_SRC = ROOT / "third_party" / "comfyui_controlnet_aux" / "src"
MODEL_DIR = ROOT / "models" / "yzd-v" / "DWPose"
DET_MODEL = MODEL_DIR / "yolox_l.onnx"
POSE_MODEL = MODEL_DIR / "dw-ll_ucoco_384.onnx"

MODEL_SPECS = {
    DET_MODEL: {
        "url": "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx",
        "size": 216_746_733,
        "sha256": "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411",
    },
    POSE_MODEL: {
        "url": "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
        "size": 134_399_116,
        "sha256": "724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843",
    },
}

# CPU OpenCV DNN 在這兩張圖只需約 0.25 秒，且不依賴特定 CUDA/cuDNN 版本。
os.environ.setdefault("AUX_ORT_PROVIDERS", "CPUExecutionProvider")
if str(VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(VENDOR_SRC))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_model(path: Path, spec: dict) -> bool:
    if not path.is_file() or path.stat().st_size != spec["size"]:
        return False
    return _sha256(path) == spec["sha256"]


def _download_once(path: Path, spec: dict) -> None:
    """下載到 .part；伺服器支援 Range 時會從中斷位置續傳。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    downloaded = part.stat().st_size if part.is_file() else 0
    if downloaded > spec["size"]:
        part.unlink()
        downloaded = 0

    headers = {"User-Agent": "VideoSplat-DWPose/1.0"}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"
    request = urllib.request.Request(spec["url"], headers=headers)

    with urllib.request.urlopen(request, timeout=60) as response:
        append = downloaded > 0 and response.status == 206
        if not append:
            downloaded = 0
        mode = "ab" if append else "wb"
        last_report = -1
        with part.open(mode) as stream:
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                downloaded += len(chunk)
                percent = min(100, int(downloaded * 100 / spec["size"]))
                if percent // 5 != last_report:
                    last_report = percent // 5
                    print(f"  {path.name}: {percent}% ({downloaded / 1024 / 1024:.1f} MB)", flush=True)

    if part.stat().st_size != spec["size"]:
        raise RuntimeError(
            f"{path.name} 大小不正確：{part.stat().st_size}，預期 {spec['size']} bytes")
    if _sha256(part) != spec["sha256"]:
        raise RuntimeError(f"{path.name} SHA-256 驗證失敗；保留 .part 供下次重新下載")
    os.replace(part, path)


def ensure_models() -> None:
    """缺少或損壞時自動下載模型；失敗會重試並保留續傳檔。"""
    for path, spec in MODEL_SPECS.items():
        if _valid_model(path, spec):
            continue
        print(f"[DWPose] 缺少或模型不完整，開始下載 {path.name}")
        last_error = None
        for attempt in range(1, 4):
            try:
                _download_once(path, spec)
                print(f"[DWPose] {path.name} 下載與驗證完成")
                break
            except (OSError, RuntimeError, urllib.error.URLError) as error:
                last_error = error
                if attempt < 3:
                    wait_seconds = attempt * 2
                    print(f"[DWPose] 第 {attempt} 次下載失敗：{error}；{wait_seconds} 秒後續傳")
                    time.sleep(wait_seconds)
        else:
            raise RuntimeError(
                f"無法下載 DWPose 模型 {path.name}。請檢查網路後重新執行；"
                f"已下載部分會自動續傳。最後錯誤：{last_error}")


class DWPoseEstimator:
    """載入一次 DWPose，對每張圖回傳 OpenPose 18 點格式。"""

    KEYPOINT_NAMES = (
        "nose", "neck", "r_shoulder", "r_elbow", "r_wrist",
        "l_shoulder", "l_elbow", "l_wrist", "r_hip", "r_knee",
        "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye",
        "l_eye", "r_ear", "l_ear",
    )

    def __init__(self) -> None:
        ensure_models()

        # 避免執行第三方 dwpose/__init__.py；該檔會載入 ONNX 路徑用不到的
        # torchvision 訓練元件。建立輕量 package 後只載入 Wholebody 推論器。
        package_name = "custom_controlnet_aux.dwpose"
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(VENDOR_SRC / "custom_controlnet_aux" / "dwpose")]
            package.__package__ = package_name
            sys.modules[package_name] = package
        Wholebody = importlib.import_module(f"{package_name}.wholebody").Wholebody

        self.model = Wholebody(str(DET_MODEL), str(POSE_MODEL), torchscript_device="cpu")

    def detect(self, image_bgr) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        raw = self.model(image_rgb)
        if raw is None or len(raw) == 0:
            raise ValueError("DWPose 找不到人物")

        # 單人圖仍可能回傳多個候選，取身體關節總信心最高者。
        person = max(raw, key=lambda item: float(item[:18, 2].sum()))
        points: dict[str, tuple[float, float]] = {}
        scores: dict[str, float] = {}
        for index, name in enumerate(self.KEYPOINT_NAMES):
            x, y, score = [float(value) for value in person[index, :3]]
            if score >= 0.30:
                points[name] = (x, y)
                scores[name] = score

        # COCO-WholeBody：92:113 左手、113:134 右手，每手 21 點。
        for side, start in (("l", 92), ("r", 113)):
            for hand_index in range(21):
                x, y, score = [float(value) for value in person[start + hand_index, :3]]
                if score >= 0.30:
                    name = f"{side}_hand_{hand_index}"
                    points[name] = (x, y)
                    scores[name] = score
        return points, scores
