from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

# ComfyUI HTTP API；若 ComfyUI 使用其他主機或連接埠，請修改此處。
COMFYUI_URL = "http://127.0.0.1:8188"

WORKFLOW_PATH = PROJECT_DIR / "workflows" / "Sam3_remove_background_api.json"
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"

# input 下的影片會被遞迴搜尋；output 會保留相同的相對資料夾結構。
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

REQUEST_TIMEOUT_SECONDS = 60
UPLOAD_TIMEOUT_SECONDS = 60 * 30
POLL_INTERVAL_SECONDS = 2
GENERATION_TIMEOUT_SECONDS = 60 * 60 * 6
UPLOAD_CHUNK_SIZE = 1024 * 1024

# ComfyUI 伺服器上的暫存輸出前綴；下載後仍會依 input 的結構命名。
COMFYUI_OUTPUT_PREFIX = "video/remove_background"
