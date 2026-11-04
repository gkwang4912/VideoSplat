from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

# ComfyUI HTTP API. Change this if ComfyUI uses another host or port.
COMFYUI_URL = "http://127.0.0.1:8188"

WORKFLOW_PATH = PROJECT_DIR / "workflows" / "VideoSplat_2_api.json"
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"

# 程式會自動掃描 input 的每個第一層子資料夾，不需設定 4、5、6 等編號。
# 子資料夾同時包含以下兩個檔名時，就會自動建立一組工作。
INPUT_IMAGE_FILENAME = "source_aligned.png"
INPUT_VIDEO_FILENAME = "pose_control_video_175.mp4"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

REQUEST_TIMEOUT_SECONDS = 30
OUTPUT_PREFIX = "video/VideoSplat_2"
POLL_INTERVAL_SECONDS = 2
GENERATION_TIMEOUT_SECONDS = 60 * 60 * 6

