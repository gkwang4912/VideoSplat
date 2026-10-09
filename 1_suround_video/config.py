from pathlib import Path

# ============================================================
# 基本路徑
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# 每一張人物圖片都會各跑一次完整 VideoSplat_360 workflow
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"
}

# 0 = 處理 input/ 內全部圖片
MAX_IMAGES = 0

# True  = 重新生成並覆蓋 output/<人物名>/video.*、mask.*
# False = 若該人物資料夾已同時存在 video 與 mask，就直接跳過
OVERWRITE_OUTPUT = False

# 是否把原始人物圖也複製進 output/<人物名>/source.<ext>
COPY_SOURCE_IMAGE = True


# ============================================================
# ComfyUI
# ============================================================

COMFYUI_URL = "http://127.0.0.1:8188"
POLL_INTERVAL = 2.0

# 上傳到 ComfyUI input/ 下的暫存子資料夾
COMFY_UPLOAD_SUBFOLDER = "videosplat_360_batch"

# ComfyUI output/ 內的暫存輸出前綴。
# Python 下載完成後會整理到本專案 output/<人物名>/。
COMFY_OUTPUT_PREFIX = "video/VideoSplat_360_batch"


# ============================================================
# 影片模型
# 對應 VideoSplat_360.json 中的影片生成工作流
# ============================================================

UNET_NAME = "minimax_h3_ref2va_int8_convrot.safetensors"
UNET_WEIGHT_DTYPE = "default"

CLIP_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
CLIP_TYPE = "minimax"
CLIP_DEVICE = "default"

VIDEO_VAE_NAME = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE_NAME = "minimax_h3_audio_vae_fp32.safetensors"


# ============================================================
# Lightning / Turbo LoRA
# ============================================================

ENABLE_LIGHTNING_LORA = False

LORA_NAME = (
    "minimax_h3_ref2v_turbo_4step_v0.1_"
    "comfyui_resized_avg_rank_21_bf16.safetensors"
)
LORA_STRENGTH = 1.0

STEPS_FULL = 20
STEPS_LIGHTNING = 4


# ============================================================
# Sage Attention
# 對應 PathchSageAttentionKJ
# ============================================================

ENABLE_SAGE_ATTENTION = True
SAGE_ATTENTION = "auto"
SAGE_ALLOW_COMPILE = True


# ============================================================
# Sampling
# ============================================================

SAMPLER_NAME = "res_multistep"
SCHEDULER = "simple"
DENOISE = 1.0

# -1 = 每張圖片隨機 seed
# >= 0 = 固定 seed；可配合 INCREMENT_SEED_PER_IMAGE 每張 +1
SEED = -1
INCREMENT_SEED_PER_IMAGE = True


# ============================================================
# 影片尺寸
# 對應 workflow Resize 子圖：原圖寬高 * 0.7，再四捨五入到 32 倍數
# ============================================================

# None = 依輸入圖自動計算
# 若 WIDTH / HEIGHT 都填整數，則強制使用固定尺寸
WIDTH = None
HEIGHT = None

RESIZE_SCALE = 0.7
SIZE_MULTIPLE = 32
MIN_WIDTH = 32
MIN_HEIGHT = 32


# ============================================================
# 影片長度與輸出
# ============================================================

# 對應 workflow 的 Float (Duration) = 8
DURATION_SECONDS = 8.0

# 主影片 CreateVideo
VIDEO_FPS = 24
VIDEO_BIT_DEPTH = 8
VIDEO_COLOR_SPACE = "sRGB"

# Mask 影片 CreateVideo；原 workflow 設為 30 fps
MASK_FPS = 30
MASK_BIT_DEPTH = "auto"
MASK_COLOR_SPACE = "sRGB"

VIDEO_FORMAT = "auto"
VIDEO_CODEC = "auto"


# ============================================================
# BiRefNet RMBG
# 只處理生成影片第一幀，做為 SAM3 initial_mask
# ============================================================

BIREFNET_MODEL = "BiRefNet-general"
BIREFNET_SENSITIVITY = 0
BIREFNET_MASK_BLUR = 0
BIREFNET_MASK_OFFSET = False
BIREFNET_INVERT_OUTPUT = False
BIREFNET_REFINE_FOREGROUND = "Alpha"
BIREFNET_BACKGROUND = "Alpha"
BIREFNET_BACKGROUND_COLOR = "#222222"


# ============================================================
# SAM3 Video Track
# ============================================================

SAM3_CHECKPOINT = "sam3.1_multiplex_fp16.safetensors"
SAM3_DETECTION_THRESHOLD = 0.5
SAM3_MAX_OBJECTS = 4
SAM3_DETECT_INTERVAL = 1

# 空字串 = 輸出追蹤到的全部物件，與目前 workflow 相同
SAM3_OBJECT_INDICES = ""


# ============================================================
# Prompt
# 完整對應 VideoSplat_360.json：純黑背景、人物完全不動、攝影機 360° 環繞
# ============================================================

PROMPT = r"""subject_definitions:
<Subject 1> is the exact person shown in <Picture 1>. Preserve the person's identity, face, facial expression, hairstyle, hair shape, body proportions, clothing, accessories, colors, materials, and pose exactly. <Subject 1> is treated as a completely motionless life-size character model for clean visual reference capture, while retaining the natural visual appearance of the original person.

summary:
[reference generation] Generate one continuous character reference shot of <Subject 1>. The subject remains completely stationary in world space for the entire video. Only the physical camera moves, performing one complete horizontal 360-degree orbit around the stationary subject and returning to the original frontal viewpoint.

retention_analysis:
<Subject 1> (appears throughout [Shot 1]): fully_preserved - preserve the exact identity, facial appearance, hairstyle, clothing, accessories, body proportions, expression, and fixed pose of the person in <Picture 1>. The subject has absolutely no self-generated motion. The character does not rotate; only the camera changes position around the character.

detailed_description:
The target video is a clean character reference capture in which the character remains fixed in world coordinates and ONLY THE CAMERA physically moves around the character.

The background must be completely pure black, uniform, flat, and empty in every frame. The entire image background should be solid black only, with no visible environment or spatial context of any kind. There must be no studio, no room, no walls, no floor, no corners, no seams, no horizon line, no backdrop structure, no platform, no pedestal, no props, no furniture, no decorations, no text, no graphics, no reflections, no visible lighting setup, and no additional objects of any kind. The frame should contain only <Subject 1> against a completely pure black background.

[Shot 1] The shot begins from the frontal appearance and viewing direction established by <Picture 1>. <Subject 1> stands completely frozen in exactly the same pose for the entire duration.

The camera performs a smooth, continuous, complete 360-degree Arc Shot around <Subject 1> at a constant radius, constant camera height, and constant speed. The camera moves horizontally clockwise around the stationary subject: beginning at the exact front view, progressively revealing the three-quarter view, exact side profile, rear three-quarter view, full back view, opposite rear three-quarter view, opposite side profile, opposite front three-quarter view, and finally returning smoothly to the exact frontal viewpoint.

Throughout the orbit, the camera continuously points toward the same fixed center of <Subject 1>. Maintain a level horizon and stable framing. Maintain the same focal length, camera distance, subject scale, and vertical framing throughout the entire orbit. Do not zoom, push in, pull out, tilt, roll, pedestal, shake, or change camera height. There are no cuts and no transitions.

CRITICAL MOTION CONSTRAINT: <Subject 1> is completely rigid and motionless in world space during every frame. The body does not rotate with the camera. The head does not turn or track the camera. The torso, shoulders, arms, hands, fingers, legs, and feet never move. The pose remains mathematically identical from beginning to end.

The facial expression is completely frozen. No blinking. No eyelid movement. No eye movement. No gaze tracking. No eyebrow movement. No mouth movement. No lip movement. No swallowing. No breathing motion. No facial muscle movement.

The hairstyle is completely frozen in shape. Individual hairs do not move. No hair flutter, hair sway, wind response, cloth simulation, or secondary motion. Clothing, loose fabric, jewelry, accessories, straps, and every other visible element remain perfectly still.

There is no wind and no environmental force acting on the subject. Do not animate the character in any way. Do not interpret the orbit as the character turning on a turntable. The subject remains fixed; ONLY THE CAMERA travels around the subject.

Maintain absolute pure black background consistency in every frame. The background must remain perfectly solid black and completely empty from beginning to end. Do not introduce any environmental details at any point. Do not generate any studio-like appearance. Do not add floor contact shadows, cast shadows, lighting gradients, background texture, or any visual cue that suggests a physical space.

Preserve temporal consistency throughout the orbit: no identity drift, no facial reconstruction changes, no body deformation, no changing proportions, no costume morphing, no hairstyle changes, no changing expression, no changing pose, and no object appearing or disappearing.

At the end of the shot, the camera completes exactly one full 360-degree horizontal orbit and returns to the same frontal viewing angle, camera height, distance, focal length, and framing as the opening.

overall_soundscape:
N/A

non_diegetic_music:
N/A"""
