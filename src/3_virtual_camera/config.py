import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent

INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"

# Each immediate child directory of INPUT_DIR is one input group. Files are
# discovered from this stable layout instead of hard-coded group or file names.
COLOR_VIDEO_GLOB = "*_output_2_*.mp4"
MASK_VIDEO_GLOB = "*_output_1_*.mp4"
COLMAP_RELATIVE_PATH = Path("colmap") / "sparse" / "0"

DATASET_FOLDER_NAME = "brush_dataset"
ZIP_FILE_NAME = "brush_dataset.zip"

EXPECTED_FRAMES = 175

IMAGE_FORMAT = "png"
IMAGE_PREFIX = "frame_"
MASK_PREFIX = "mask_"

EXPORT_SEPARATE_MASKS = True
COMPOSE_RGBA_IMAGES = True
MASK_THRESHOLD = 127
MASK_EROSION_PIXELS = 0

REUSE_ORIGINAL_CAMERA_POSES = True
ESTIMATE_CAMERAS = False
OPTIMIZE_CAMERAS = False
MATCH_FEATURES = False

EXPORT_ZIP = True

OVERWRITE_OUTPUT = True

# ffmpeg is used only to decode the source video into its original frame order.
FFMPEG_EXECUTABLE = "ffmpeg"

# Optional Brush path used only when printing the next command. The dataset
# builder never starts training. Set VIDEOSPLAT_BRUSH_EXE or place brush-cli
# on PATH; the fallback remains a non-existing local name when neither is set.
BRUSH_EXECUTABLE = Path(
    os.environ.get("VIDEOSPLAT_BRUSH_EXE")
    or shutil.which("brush-cli")
    or shutil.which("brush-cli.exe")
    or "brush-cli.exe"
)
