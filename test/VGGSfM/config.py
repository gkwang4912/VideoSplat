from pathlib import Path

# ============================================================
# 路徑
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent
INPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "output"
VGGSFM_REPO = ROOT_DIR / "third_party" / "vggsfm"

# ============================================================
# 輸入影片 / 抽幀
# ============================================================
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
FRAME_STEP = 1               # 1 = 每一幀都使用
MAX_FRAMES = None            # None = 不限制
IMAGE_FORMAT = "jpg"         # jpg / png
JPEG_QUALITY = 100
CLEAR_OUTPUT_BEFORE_RUN = True

# ============================================================
# VGGSfM 執行模式
# auto: <= GLOBAL_MAX_FRAMES 用 demo.py 全局重建；更多幀用 video_demo.py
# global: 強制 demo.py
# video: 強制 video_demo.py sliding-window
# ============================================================
MODE = "auto"
GLOBAL_MAX_FRAMES = 400      # 官方 README 表示一般 sparse mode 約可處理至 400 frames

# 影片通常由同一台相機、同一焦段拍攝
SHARED_CAMERA = True
CAMERA_TYPE = "SIMPLE_RADIAL"
MIXED_PRECISION = "fp16"
QUERY_METHOD = "aliked"
FINE_TRACKING = True
USE_POSELIB = True

# 你已經先篩過影片，因此預設不主動剔除 frame
FILTER_INVALID_FRAME = False
COMPLEMENT_NON_VISIBLE = True

# ============================================================
# 全局重建 demo.py 參數
# ============================================================
GLOBAL_MAX_QUERY_PTS = 2048
GLOBAL_QUERY_FRAME_NUM = 8
# 360 度環繞影片依時間通常就是依角度排列，因此均勻取 query frame
GLOBAL_QUERY_BY_INTERVAL = True
GLOBAL_QUERY_BY_MIDPOINT = False
GLOBAL_BA_ITERS = 2
GLOBAL_ROBUST_REFINE = 2

# ============================================================
# 長影片 video_demo.py 參數
# 官方預設：32 / 16 / 6。這裡把 Joint BA 稍微加密到 4。
# ============================================================
VIDEO_MAX_QUERY_PTS = 1024
VIDEO_QUERY_FRAME_NUM = 6
INIT_WINDOW_SIZE = 32
WINDOW_SIZE = 16
JOINT_BA_INTERVAL = 4

# ============================================================
# 診斷輸出
# ============================================================
MAKE_REPROJECTION_VIDEO = False
VISUAL_TRACKS = False
GR_VISUALIZE = False
VIZ_VISUALIZE = False

# 只用於結果檢查，不會硬改相機位置
TARGET_REGISTERED_RATIO = 0.80
TARGET_ANGULAR_COVERAGE_DEG = 330.0
MAX_LOOP_CLOSURE_STEP_RATIO = 6.0
