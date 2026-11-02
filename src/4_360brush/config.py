import os
import shutil
from pathlib import Path

# ============================================================
# 基本路徑
# ============================================================

# main.py / config.py 所在資料夾
BASE_DIR = Path(__file__).resolve().parent

# 預設直接讀取 Stage 3 建立的 RGBA Brush dataset。若要改用其他資料，
# 可用 VIDEOSPLAT_BRUSH_INPUT_DIR 指定包含各 group 的目錄。
INPUT_DIR = Path(
    os.environ.get("VIDEOSPLAT_BRUSH_INPUT_DIR")
    or BASE_DIR.parent / "3_virtual_camera" / "output"
)
OUTPUT_DIR = BASE_DIR / "output"

# 每一組資料中真正要送進 Brush 的資料夾名稱
# 例如：
# 3_virtual_camera/output/
#   1/
#     brush_dataset/
#       images/
#       sparse/
DATASET_SUBDIR = "brush_dataset"

# 使用手動 GUI 訓練結果良好的 Brush 0.3.0；同一執行檔也支援 CLI 批次模式。
BRUSH_EXE = Path(
    os.environ.get("VIDEOSPLAT_BRUSH_EXE")
    or shutil.which("brush-cli")
    or shutil.which("brush-cli.exe")
    or "brush-cli.exe"
)


# ============================================================
# Brush 高品質訓練參數
# ============================================================
# 對齊 Brush 0.3.0 GUI 中已驗證效果良好的設定。

# Brush 0.3.0：--total-steps
# 新版 main branch：--total-train-iters
# main.py 會從 brush-cli.exe --help 自動判斷。
AUTO_DETECT_STEPS_FLAG = True
MANUAL_STEPS_FLAG = "--total-steps"

# 與 GUI 截圖一致。
TOTAL_STEPS = 50_000

# 最大 Gaussian / splat 數量上限。
# 這只是 cap，不代表一定會長到這麼多。
MAX_SPLATS = 10_000_000

# 原始 RGBA 影像最大載入解析度。
# 目前資料是人物多視角影像，2048 是品質 / VRAM 比較合理的高品質設定。
MAX_RESOLUTION = 2048

# Spherical Harmonics degree
SH_DEGREE = 3

# 隨機種子
SEED = 42


# ============================================================
# RGBA / 去背訓練設定
# ============================================================

# Brush 0.3.0 會自動把 RGBA PNG 的 alpha 當透明度；此版本沒有 --alpha-mode。
ALPHA_MODE = None

# 與 Brush 0.3.0 GUI/CLI 原生預設一致。
MATCH_ALPHA_WEIGHT = 0.1

# Brush 0.3.0 沒有 Mip-Splatting scale floor；不傳入此參數。
MIN_SCALE_FACTOR = None

# None = 完全使用 Brush 0.3.0 GUI 預設（目前為 15000）。
GROWTH_STOP_ITER = None

# None = 使用 Brush 預設值。
GROWTH_START_ITER = None
GROWTH_GRAD_THRESHOLD = None
GROWTH_SELECT_FRACTION = None
REFINE_EVERY = None
SPLIT_AT_SCREEN_SIZE = None

# Brush 對透明區背後使用的背景設定。
# 保持官方/目前成功流程的基準值，不額外強行改動。
BACKGROUND_COLOR = None
BACKGROUND_NOISE_STRENGTH = None


# ============================================================
# Dataset 相關參數
# ============================================================

# None = 不限制
MAX_FRAMES = None

# None = 不做 eval split
# 例如 8 = 每 8 張取一張作為 eval
EVAL_SPLIT_EVERY = None

# None = 不抽幀
# 例如 2 = 每 2 張取 1 張
SUBSAMPLE_FRAMES = None

# None = 不抽 sparse point
SUBSAMPLE_POINTS = None

# 啟動前檢查 PNG 是否真的帶 alpha channel。
# 只做快速 header 檢查，不會修改任何圖片。
CHECK_PNG_ALPHA = True


# ============================================================
# Eval / 匯出設定
# ============================================================

EVAL_EVERY = 1000
EVAL_SAVE_TO_DISK = False

# None = 只需要最終輸出。
# main.py 會把 --export-every 設成 TOTAL_STEPS，
# Brush 在最後一步也會輸出最終結果。
#
# 如果改成 5000，則每 5000 steps 匯出一次 checkpoint。
EXPORT_EVERY = None

# False：不開 Brush 視窗，適合批次跑
# True：每組都開 viewer
WITH_VIEWER = False


# ============================================================
# Brush 版本檢查
# ============================================================

# 啟動前確認執行檔支援這套 Brush 0.3.0 GUI 等效流程。
REQUIRE_BRUSH_FLAGS = True

REQUIRED_FLAGS = [
    "--match-alpha-weight",
    "--growth-stop-iter",
    "--max-resolution",
    "--max-splats",
    "--export-every",
    "--export-path",
    "--export-name",
]


# ============================================================
# 批次處理行為
# ============================================================

# 這次要用新的 GLB-safe 參數重新訓練，因此不要沿用舊 PLY。
# 確認新版本都產生完成後，可再改回 True。
SKIP_COMPLETED = False

# 某一組失敗後是否繼續下一組
CONTINUE_ON_ERROR = True

# 只列出將執行的命令，不真的執行 Brush
DRY_RUN = False

# 不設定 timeout。Brush 可以一直跑到自己完成。
NO_TIMEOUT = True


# ============================================================
# 額外 Brush CLI 參數
# ============================================================

# 這裡可以加入 Brush 支援但上面沒有單獨列出的參數。
# 例如：
# EXTRA_ARGS = [
#     "--opac-decay", "0.004",
# ]
EXTRA_ARGS = []


# ============================================================
# 環境變數
# ============================================================

# 例如：
# EXTRA_ENV = {
#     "CUBECL_DEFAULT_DEVICE": "0",
# }
EXTRA_ENV = {}
