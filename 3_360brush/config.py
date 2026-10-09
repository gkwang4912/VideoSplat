import os
import shutil
from pathlib import Path

# ============================================================
# 基本路徑
# ============================================================

# main.py / config.py 所在資料夾
BASE_DIR = Path(__file__).resolve().parent

# 輸入與輸出資料夾
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

# 每一組資料中真正要送進 Brush 的資料夾名稱
# 例如：
# input/
#   1/
#     brush_dataset_rgba_clean/
#       images/
#       sparse/
DATASET_SUBDIR = "brush_dataset_rgba_clean"

# 使用支援 --min-scale-factor / --alpha-mode 的 Brush CLI。
# 優先讀取 VIDEOSPLAT_BRUSH_EXE，否則從 PATH 尋找。
BRUSH_EXE = Path(
    os.environ.get("VIDEOSPLAT_BRUSH_EXE")
    or shutil.which("brush-cli")
    or shutil.which("brush-cli.exe")
    or "brush-cli.exe"
)


# ============================================================
# Brush 高品質訓練參數
# ============================================================
# 目前採「GLB-safe 高品質」設定：
#   - 訓練可跑久（100k）
#   - densification 只到 15k
#   - min-scale-factor 固定 1.0
# 目標是優先確保輸出的 Gaussian PLY 能穩定交給 ComfyUI SplatToMesh。

# Brush 0.3.0：--total-steps
# 新版 main branch：--total-train-iters
# main.py 會從 brush-cli.exe --help 自動判斷。
AUTO_DETECT_STEPS_FLAG = True
MANUAL_STEPS_FLAG = "--total-train-iters"

# GLB-safe 高品質版本：保留 100k steps 做後期收斂；densification 會提早停止。
TOTAL_STEPS = 100_000

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

# 強制把 PNG alpha 當真正透明度使用，不把透明背景當黑色 RGB 背景學進去。
ALPHA_MODE = "transparent"

# 加強 predicted alpha 對齊輸入 PNG alpha。
# 先前已驗證 1.0 可明顯抑制人物周圍的黑色浮雲。
MATCH_ALPHA_WEIGHT = 1.0

# Mip-Splatting 3D scale floor。
# 這個專案的已驗證 GLB-safe 基準是 1.0。
# 之前改成 0.5 後，雖然可能保留更薄的細節，但也更容易重新產生
# 對 ComfyUI SplatToMesh 不友善的極薄 Gaussian。
# 若優先目標是穩定 PLY -> SplatToMesh -> GLB，維持 1.0。
MIN_SCALE_FACTOR = 1.0

# GLB-safe 基準：在 15k 停止 densification / growth。
# 後面的 steps 仍會繼續優化既有 Gaussian，因此 TOTAL_STEPS 可維持 100k。
# 這也回到 Brush 官方目前的 growth_stop_iter 預設值，避免後期持續產生
# 大量更小、更薄的 splats，增加 SplatToMesh 數值不穩定風險。
GROWTH_STOP_ITER = 15_000

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

# 若為 True，main.py 會要求目前 Brush 必須支援我們這套新流程的 CLI flags。
# 可防止不小心又跑到舊的 brush_app.exe。
REQUIRE_LATEST_BRUSH_FLAGS = True

REQUIRED_FLAGS = [
    "--alpha-mode",
    "--match-alpha-weight",
    "--min-scale-factor",
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
