from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

# ComfyUI HTTP API. Change this if ComfyUI uses another host or port.
COMFYUI_URL = "http://127.0.0.1:8188"

WORKFLOW_PATH = PROJECT_DIR / "workflows" / "VideoSplat_2_api.json"
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"

# 明確指定要使用的檔案；image 與 video 都是相對於 input 資料夾的路徑。
# 即使 input 內有其他圖片、影片或子資料夾，程式也不會自動選取它們。
# 需要處理多組資料時，複製一組設定並修改 name、image、video 即可。
INPUT_JOBS = [
    {
        "name": "4",
        "image": "4/source_aligned.png",
        "video": "4/pose_control_video_175.mp4",
    },
    {
        "name": "5",
        "image": "5/source_aligned.png",
        "video": "5/pose_control_video_175.mp4",
    },
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

REQUEST_TIMEOUT_SECONDS = 30
OUTPUT_PREFIX = "video/VideoSplat_2"
POLL_INTERVAL_SECONDS = 2
GENERATION_TIMEOUT_SECONDS = 60 * 60 * 6

