"""DWPose 模型介面。模型與第三方程式都放在專案內，不污染系統 Python。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent
VENDOR_SRC = ROOT / "third_party" / "comfyui_controlnet_aux" / "src"
MODEL_DIR = ROOT / "models" / "yzd-v" / "DWPose"
DET_MODEL = MODEL_DIR / "yolox_l.onnx"
POSE_MODEL = MODEL_DIR / "dw-ll_ucoco_384.onnx"

# CPU OpenCV DNN 在這兩張圖只需約 0.25 秒，且不依賴特定 CUDA/cuDNN 版本。
os.environ.setdefault("AUX_ORT_PROVIDERS", "CPUExecutionProvider")
if str(VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(VENDOR_SRC))


class DWPoseEstimator:
    """載入一次 DWPose，對每張圖回傳 OpenPose 18 點格式。"""

    KEYPOINT_NAMES = (
        "nose", "neck", "r_shoulder", "r_elbow", "r_wrist",
        "l_shoulder", "l_elbow", "l_wrist", "r_hip", "r_knee",
        "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye",
        "l_eye", "r_ear", "l_ear",
    )

    def __init__(self) -> None:
        missing = [path for path in (DET_MODEL, POSE_MODEL) if not path.is_file()]
        if missing:
            raise FileNotFoundError("缺少 DWPose 模型：" + ", ".join(str(path) for path in missing))

        from custom_controlnet_aux.dwpose.wholebody import Wholebody

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
