"""骨架環繞動畫的唯一設定檔。修改此檔後執行 main.py。"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"

# input/ 內每張圖片都視為一位人物。支援子資料夾；輸出名稱會使用圖片檔名。
INPUT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# 批次輸出結構：output/default 保留原始固定骨架，其餘資料夾各對應一張人物圖。
DEFAULT_OUTPUT_NAME = "default"
SAVE_ALIGNMENT_PREVIEW = True

# 優先使用 DWPose 模型；只有模型真的失敗時才回退到輪廓估計。
USE_DWPOSE = True
POSE_PYTHON = ROOT / ".pose_env" / "python.exe"
POSE_MIN_CONFIDENCE = 0.30

# 白色或透明背景人物圖的前景偵測參數。
# 數值越小越容易把接近白色的陰影也視為人物；通常不需要修改。
BACKGROUND_COLOR_DISTANCE = 24.0
FOREGROUND_MIN_AREA_RATIO = 0.02

# 影片規格
WIDTH = 720
HEIGHT = 1280
FPS = 24
FRAME_COUNT = 175
VIDEO_FILENAME = "pose_control_video_175.mp4"
VIDEO_CRF = 12

# 固定半徑、固定高度、固定焦距的水平 360° 攝影機環繞。
# Frame 0 與最後一幀皆為正面；角度公式為 360 * frame / (FRAME_COUNT - 1)。
CAMERA_RADIUS = 4.8
CAMERA_HEIGHT = 1.45
CAMERA_TARGET = (0.0, 1.05, 0.0)
FOCAL_LENGTH_PX = 1850.0
PRINCIPAL_POINT = (WIDTH / 2.0, HEIGHT * 0.53)

# 是否保留與影片逐幀對應的 PNG。COLMAP images.txt 會引用這些檔名。
SAVE_FRAME_IMAGES = True
FRAME_NAME_PATTERN = "frame{frame:03d}.png"

# 除錯圖顯示的關鍵幀；文字只會出現在 debug 圖，不會進入正式控制影片。
DEBUG_FRAMES = (0, 44, 87, 131, 174)

# 固定人物姿勢。世界座標：+Y 向上，+Z 為人物正面。
JOINTS = {
    "pelvis": (0.00, 1.02, 0.00), "spine": (0.00, 1.30, 0.00),
    "neck": (0.00, 1.60, 0.00), "head_center": (0.00, 1.79, 0.00),
    "crown": (0.00, 1.96, 0.00), "nose": (0.00, 1.83, 0.105),
    "r_eye": (-0.045, 1.87, 0.120), "l_eye": (0.045, 1.87, 0.120),
    "r_ear": (-0.095, 1.84, 0.065), "l_ear": (0.095, 1.84, 0.065),
    "r_shoulder": (-0.22, 1.56, 0.00), "r_elbow": (-0.48, 1.39, 0.055),
    "r_wrist": (-0.67, 1.20, 0.10), "l_shoulder": (0.22, 1.56, 0.00),
    "l_elbow": (0.50, 1.47, 0.04), "l_wrist": (0.72, 1.38, 0.09),
    "r_hip": (-0.13, 1.00, 0.00), "r_knee": (-0.16, 0.57, 0.035),
    "r_ankle": (-0.18, 0.10, 0.015), "r_toe": (-0.18, 0.03, 0.20),
    "r_heel": (-0.18, 0.03, -0.08), "l_hip": (0.13, 1.00, 0.00),
    "l_knee": (0.18, 0.57, -0.025), "l_ankle": (0.22, 0.10, 0.00),
    "l_toe": (0.22, 0.03, 0.20), "l_heel": (0.22, 0.03, -0.08),
}

# SDPose / OpenPose 風格的 RGB 骨架顏色。
LIMBS = [
    ("neck", "r_shoulder", (255, 85, 0)), ("r_shoulder", "r_elbow", (255, 170, 0)),
    ("r_elbow", "r_wrist", (255, 255, 0)), ("neck", "l_shoulder", (170, 255, 0)),
    ("l_shoulder", "l_elbow", (85, 255, 0)), ("l_elbow", "l_wrist", (0, 255, 0)),
    ("neck", "spine", (0, 255, 85)), ("spine", "pelvis", (0, 255, 125)),
    ("pelvis", "r_hip", (0, 255, 170)), ("r_hip", "r_knee", (0, 255, 255)),
    ("r_knee", "r_ankle", (0, 170, 255)), ("pelvis", "l_hip", (0, 85, 255)),
    ("l_hip", "l_knee", (0, 0, 255)), ("l_knee", "l_ankle", (85, 0, 255)),
    ("neck", "head_center", (170, 0, 255)), ("head_center", "crown", (220, 0, 255)),
    ("head_center", "nose", (200, 0, 255)), ("nose", "r_eye", (255, 0, 255)),
    ("r_eye", "r_ear", (255, 0, 170)), ("nose", "l_eye", (255, 0, 85)),
    ("l_eye", "l_ear", (255, 0, 0)), ("r_ankle", "r_toe", (0, 255, 255)),
    ("r_ankle", "r_heel", (0, 210, 255)), ("l_ankle", "l_toe", (110, 0, 255)),
    ("l_ankle", "l_heel", (165, 0, 255)),
]

