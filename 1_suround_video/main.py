from __future__ import annotations

import json
import mimetypes
import random
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image

import config


VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}

# build_prompt() 中兩個 SaveVideo 的固定 node id。
MAIN_VIDEO_SAVE_NODE = "17"
MASK_VIDEO_SAVE_NODE = "25"


def api_url(path: str) -> str:
    return config.COMFYUI_URL.rstrip("/") + "/" + path.lstrip("/")


def http_json(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    headers: dict[str, str] = {}
    data = None

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(api_url(path), data=data, headers=headers, method=method)

    # 不設定 timeout：ComfyUI 可不限時間完成請求。
    with urlopen(req) as resp:
        raw = resp.read()

    if not raw:
        return None

    return json.loads(raw.decode("utf-8"))


def check_comfyui() -> None:
    try:
        info = http_json("GET", "/object_info")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"無法連線 ComfyUI：{config.COMFYUI_URL}\n"
            "請先啟動 ComfyUI，並確認網址與連接埠。"
        ) from exc

    required_nodes = {
        "LoadImage",
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "LoraLoaderModelOnly",
        "MiniMaxH3ImageToVideo",
        "BasicGuider",
        "RandomNoise",
        "KSamplerSelect",
        "BasicScheduler",
        "SamplerCustomAdvanced",
        "VAEDecode",
        "VAEDecodeAudio",
        "CreateVideo",
        "SaveVideo",
        "ImageFromBatch",
        "BiRefNetRMBG",
        "CheckpointLoaderSimple",
        "SAM3_VideoTrack",
        "SAM3_TrackToMask",
        "MaskToImage",
    }

    if config.ENABLE_SAGE_ATTENTION:
        required_nodes.add("PathchSageAttentionKJ")

    if isinstance(info, dict):
        missing = sorted(node for node in required_nodes if node not in info)
        if missing:
            raise RuntimeError(
                "目前 ComfyUI 缺少以下 workflow 節點：\n  - "
                + "\n  - ".join(missing)
                + "\n請先更新 ComfyUI 或安裝對應 custom nodes。"
            )


def make_multipart(
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
    upload_name: str,
) -> tuple[bytes, str]:
    boundary = "----VideoSplat360Batch" + uuid.uuid4().hex
    chunks: list[bytes] = []

    for key, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
        )
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")

    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"

    chunks.append(f"--{boundary}\r\n".encode())
    chunks.append(
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{upload_name}"\r\n'
        ).encode()
    )
    chunks.append(f"Content-Type: {mime}\r\n\r\n".encode())
    chunks.append(file_path.read_bytes())
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())

    return b"".join(chunks), boundary


def upload_image(image_path: Path) -> str:
    remote_name = (
        f"{image_path.stem}_{uuid.uuid4().hex[:10]}{image_path.suffix.lower()}"
    )

    body, boundary = make_multipart(
        fields={
            "type": "input",
            "subfolder": config.COMFY_UPLOAD_SUBFOLDER,
            "overwrite": "true",
        },
        file_field="image",
        file_path=image_path,
        upload_name=remote_name,
    )

    req = Request(
        api_url("/upload/image"),
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    with urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    name = result.get("name", remote_name)
    subfolder = result.get("subfolder", config.COMFY_UPLOAD_SUBFOLDER)

    if subfolder:
        return f"{str(subfolder).rstrip('/')}/{name}"
    return str(name)


def round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple) * multiple))


def get_generation_size(image_path: Path) -> tuple[int, int]:
    if (config.WIDTH is None) ^ (config.HEIGHT is None):
        raise ValueError("WIDTH 與 HEIGHT 必須同時設定，或同時為 None。")

    if config.WIDTH is not None and config.HEIGHT is not None:
        width = int(config.WIDTH)
        height = int(config.HEIGHT)
    else:
        with Image.open(image_path) as im:
            src_width, src_height = im.size

        # 對應 workflow Resize 子圖：round(a*b/32)*32
        width = round_to_multiple(
            src_width * float(config.RESIZE_SCALE),
            int(config.SIZE_MULTIPLE),
        )
        height = round_to_multiple(
            src_height * float(config.RESIZE_SCALE),
            int(config.SIZE_MULTIPLE),
        )

    width = max(int(config.MIN_WIDTH), width)
    height = max(int(config.MIN_HEIGHT), height)

    width = round_to_multiple(width, int(config.SIZE_MULTIPLE))
    height = round_to_multiple(height, int(config.SIZE_MULTIPLE))
    return width, height


def duration_to_length(seconds: float) -> int:
    # 完整對應 workflow：
    # max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17
    raw = max(5, round(float(seconds) * 24))
    return raw + (5 - (raw % 17)) % 17


def choose_seed(index: int) -> int:
    if int(config.SEED) < 0:
        return random.SystemRandom().randrange(0, 2**63 - 1)

    seed = int(config.SEED)
    if bool(config.INCREMENT_SEED_PER_IMAGE):
        seed += index
    return seed


def build_prompt(
    uploaded_image: str,
    width: int,
    height: int,
    seed: int,
    main_output_prefix: str,
    mask_output_prefix: str,
) -> dict[str, Any]:
    """建立與 VideoSplat_360.json 等價的 ComfyUI API prompt。"""

    steps = (
        int(config.STEPS_LIGHTNING)
        if config.ENABLE_LIGHTNING_LORA
        else int(config.STEPS_FULL)
    )
    length = duration_to_length(float(config.DURATION_SECONDS))

    prompt: dict[str, Any] = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": uploaded_image},
        },
        "2": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": config.UNET_NAME,
                "weight_dtype": config.UNET_WEIGHT_DTYPE,
            },
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": config.CLIP_NAME,
                "type": config.CLIP_TYPE,
                "device": config.CLIP_DEVICE,
            },
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": config.VIDEO_VAE_NAME},
        },
        "5": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": config.AUDIO_VAE_NAME},
        },
    }

    # workflow 的 If/Else Switch(model) 等價邏輯。
    if config.ENABLE_LIGHTNING_LORA:
        prompt["6"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["2", 0],
                "lora_name": config.LORA_NAME,
                "strength_model": float(config.LORA_STRENGTH),
            },
        }
        selected_model: list[Any] = ["6", 0]
    else:
        selected_model = ["2", 0]

    # workflow 中 Sage Attention 位於 model switch 之後、BasicGuider 之前。
    if config.ENABLE_SAGE_ATTENTION:
        prompt["7"] = {
            "class_type": "PathchSageAttentionKJ",
            "inputs": {
                "model": selected_model,
                "sage_attention": config.SAGE_ATTENTION,
                "allow_compile": bool(config.SAGE_ALLOW_COMPILE),
            },
        }
        guider_model: list[Any] = ["7", 0]
    else:
        guider_model = selected_model

    prompt.update(
        {
            "8": {
                "class_type": "MiniMaxH3ImageToVideo",
                "inputs": {
                    "clip": ["3", 0],
                    "vae": ["4", 0],
                    # 原 workflow 只有 first_frame，last_frame 未接線。
                    "first_frame": ["1", 0],
                    "prompt": config.PROMPT,
                    "width": int(width),
                    "height": int(height),
                    "length": int(length),
                },
            },
            "9": {
                "class_type": "BasicGuider",
                "inputs": {
                    "model": guider_model,
                    "conditioning": ["8", 0],
                },
            },
            "10": {
                "class_type": "RandomNoise",
                "inputs": {"noise_seed": int(seed)},
            },
            "11": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": config.SAMPLER_NAME},
            },
            "12": {
                "class_type": "BasicScheduler",
                "inputs": {
                    # 原 workflow 的 BasicScheduler 直接吃 UNETLoader 輸出，
                    # 不經 LoRA / Sage。
                    "model": ["2", 0],
                    "scheduler": config.SCHEDULER,
                    "steps": int(steps),
                    "denoise": float(config.DENOISE),
                },
            },
            "13": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["10", 0],
                    "guider": ["9", 0],
                    "sampler": ["11", 0],
                    "sigmas": ["12", 0],
                    "latent_image": ["8", 1],
                },
            },
            "14": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["13", 0],
                    "vae": ["4", 0],
                },
            },
            "15": {
                "class_type": "VAEDecodeAudio",
                "inputs": {
                    "samples": ["13", 0],
                    "vae": ["5", 0],
                },
            },
            "16": {
                "class_type": "CreateVideo",
                "inputs": {
                    "images": ["14", 0],
                    "audio": ["15", 0],
                    "fps": int(config.VIDEO_FPS),
                    "bit_depth": int(config.VIDEO_BIT_DEPTH),
                    "color_space": config.VIDEO_COLOR_SPACE,
                },
            },
            MAIN_VIDEO_SAVE_NODE: {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["16", 0],
                    "filename_prefix": main_output_prefix,
                    "format": config.VIDEO_FORMAT,
                    "codec": config.VIDEO_CODEC,
                },
            },
            # ----------------------------------------------------
            # 第一幀 -> BiRefNet -> SAM3 initial mask
            # ----------------------------------------------------
            "18": {
                "class_type": "ImageFromBatch",
                "inputs": {
                    "image": ["14", 0],
                    "batch_index": 0,
                    "length": 1,
                },
            },
            "19": {
                "class_type": "BiRefNetRMBG",
                "inputs": {
                    "image": ["18", 0],
                    "model": config.BIREFNET_MODEL,
                    "sensitivity": config.BIREFNET_SENSITIVITY,
                    "mask_blur": config.BIREFNET_MASK_BLUR,
                    "mask_offset": config.BIREFNET_MASK_OFFSET,
                    "invert_output": config.BIREFNET_INVERT_OUTPUT,
                    "refine_foreground": config.BIREFNET_REFINE_FOREGROUND,
                    "background": config.BIREFNET_BACKGROUND,
                    "background_color": config.BIREFNET_BACKGROUND_COLOR,
                },
            },
            "20": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": config.SAM3_CHECKPOINT},
            },
            "21": {
                "class_type": "SAM3_VideoTrack",
                "inputs": {
                    "images": ["14", 0],
                    "model": ["20", 0],
                    "initial_mask": ["19", 1],
                    "detection_threshold": float(
                        config.SAM3_DETECTION_THRESHOLD
                    ),
                    "max_objects": int(config.SAM3_MAX_OBJECTS),
                    "detect_interval": int(config.SAM3_DETECT_INTERVAL),
                },
            },
            "22": {
                "class_type": "SAM3_TrackToMask",
                "inputs": {
                    "track_data": ["21", 0],
                    "object_indices": config.SAM3_OBJECT_INDICES,
                },
            },
            "23": {
                "class_type": "MaskToImage",
                "inputs": {"mask": ["22", 0]},
            },
            "24": {
                "class_type": "CreateVideo",
                "inputs": {
                    "images": ["23", 0],
                    "fps": int(config.MASK_FPS),
                    "bit_depth": config.MASK_BIT_DEPTH,
                    "color_space": config.MASK_COLOR_SPACE,
                },
            },
            MASK_VIDEO_SAVE_NODE: {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["24", 0],
                    "filename_prefix": mask_output_prefix,
                    "format": config.VIDEO_FORMAT,
                    "codec": config.VIDEO_CODEC,
                },
            },
        }
    )

    return prompt


def queue_prompt(prompt: dict[str, Any], client_id: str) -> str:
    result = http_json(
        "POST",
        "/prompt",
        {"prompt": prompt, "client_id": client_id},
    )

    if not isinstance(result, dict) or "prompt_id" not in result:
        raise RuntimeError(
            "ComfyUI /prompt 沒有回傳 prompt_id。\n"
            f"回應內容：{result}"
        )

    return str(result["prompt_id"])


def wait_for_history(prompt_id: str) -> dict[str, Any]:
    # 不設定生成 timeout；只要 ComfyUI 尚未完成，就持續輪詢。
    while True:
        try:
            history = http_json("GET", f"/history/{prompt_id}")
        except HTTPError as exc:
            if exc.code == 404:
                history = {}
            else:
                raise

        if isinstance(history, dict) and prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            status_str = str(status.get("status_str", "")).lower()

            if status_str in {"error", "failed"}:
                raise RuntimeError(
                    "ComfyUI 任務失敗：\n"
                    + json.dumps(entry, ensure_ascii=False, indent=2)
                )

            if "outputs" in entry:
                return entry

        time.sleep(float(config.POLL_INTERVAL))


def find_video_refs(value: Any) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            filename = obj.get("filename")
            if isinstance(filename, str):
                ext = Path(filename).suffix.lower()
                if ext in VIDEO_EXTENSIONS:
                    refs.append(
                        {
                            "filename": filename,
                            "subfolder": str(obj.get("subfolder", "")),
                            "type": str(obj.get("type", "output")),
                        }
                    )

            for child in obj.values():
                walk(child)

        elif isinstance(obj, list):
            for child in obj:
                walk(child)

    walk(value)

    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for ref in refs:
        key = (ref["filename"], ref["subfolder"], ref["type"])
        if key not in seen:
            unique.append(ref)
            seen.add(key)

    return unique


def refs_from_save_node(history_entry: dict[str, Any], node_id: str) -> list[dict[str, str]]:
    outputs = history_entry.get("outputs", {})
    if not isinstance(outputs, dict):
        return []
    return find_video_refs(outputs.get(node_id, {}))


def download_output(ref: dict[str, str], target_path: Path) -> Path:
    query = urlencode(
        {
            "filename": ref["filename"],
            "subfolder": ref.get("subfolder", ""),
            "type": ref.get("type", "output"),
        }
    )

    req = Request(api_url(f"/view?{query}"), method="GET")
    with urlopen(req) as resp:
        data = resp.read()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(data)
    return target_path


def discover_input_images() -> list[Path]:
    config.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        (
            p
            for p in config.INPUT_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
        ),
        key=lambda p: p.name.lower(),
    )

    if config.MAX_IMAGES and int(config.MAX_IMAGES) > 0:
        images = images[: int(config.MAX_IMAGES)]

    return images


def sanitize_name(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    cleaned = cleaned.strip(" ._")
    return cleaned or "character"


def character_output_dir(image_path: Path) -> Path:
    return config.OUTPUT_DIR / sanitize_name(image_path.stem)


def find_existing_named_video(folder: Path, basename: str) -> Path | None:
    for ext in VIDEO_EXTENSIONS:
        path = folder / f"{basename}{ext}"
        if path.exists():
            return path
    return None


def character_is_complete(image_path: Path) -> bool:
    folder = character_output_dir(image_path)
    return (
        find_existing_named_video(folder, "video") is not None
        and find_existing_named_video(folder, "mask") is not None
    )


def copy_source_image(image_path: Path, folder: Path) -> None:
    if not config.COPY_SOURCE_IMAGE:
        return

    target = folder / f"source{image_path.suffix.lower()}"
    if target.exists() and not config.OVERWRITE_OUTPUT:
        return
    shutil.copy2(image_path, target)


def save_single_video_ref(
    refs: list[dict[str, str]],
    folder: Path,
    basename: str,
) -> Path:
    if not refs:
        raise RuntimeError(f"找不到 {basename} 的 SaveVideo 輸出。")

    # 每個 SaveVideo 正常只會有一個影片。若意外多個，取第一個，
    # 並把完整 history 留給上層在錯誤時除錯。
    ref = refs[0]
    ext = Path(ref["filename"]).suffix.lower() or ".mp4"
    target = folder / f"{basename}{ext}"

    if target.exists() and not config.OVERWRITE_OUTPUT:
        return target

    return download_output(ref, target)


def process_one(
    image_path: Path,
    index: int,
    total: int,
    client_id: str,
) -> None:
    print()
    print("=" * 72)
    print(f"[{index + 1}/{total}] {image_path.name}")

    output_dir = character_output_dir(image_path)

    if not config.OVERWRITE_OUTPUT and character_is_complete(image_path):
        print(f"跳過：{output_dir} 已同時存在 video 與 mask。")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_source_image(image_path, output_dir)

    width, height = get_generation_size(image_path)
    length = duration_to_length(float(config.DURATION_SECONDS))
    seed = choose_seed(index)

    print(f"輸出資料夾：{output_dir}")
    print(f"尺寸：{width} x {height}")
    print(f"時長：{config.DURATION_SECONDS}s -> length={length}")
    print(f"Seed：{seed}")
    print("上傳人物圖到 ComfyUI...")

    uploaded = upload_image(image_path)
    safe_stem = sanitize_name(image_path.stem)

    comfy_base = (
        f"{config.COMFY_OUTPUT_PREFIX.rstrip('/')}/"
        f"{safe_stem}_{uuid.uuid4().hex[:8]}"
    )
    main_prefix = f"{comfy_base}/video"
    mask_prefix = f"{comfy_base}/mask"

    prompt = build_prompt(
        uploaded_image=uploaded,
        width=width,
        height=height,
        seed=seed,
        main_output_prefix=main_prefix,
        mask_output_prefix=mask_prefix,
    )

    print("送出完整 VideoSplat_360 任務...")
    prompt_id = queue_prompt(prompt, client_id)
    print(f"prompt_id：{prompt_id}")
    print("等待影片模型、去背、追蹤與兩支影片輸出完成...")

    history_entry = wait_for_history(prompt_id)

    main_refs = refs_from_save_node(history_entry, MAIN_VIDEO_SAVE_NODE)
    mask_refs = refs_from_save_node(history_entry, MASK_VIDEO_SAVE_NODE)

    if not main_refs or not mask_refs:
        debug_path = output_dir / "history.json"
        debug_path.write_text(
            json.dumps(history_entry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        missing = []
        if not main_refs:
            missing.append("主影片")
        if not mask_refs:
            missing.append("mask 影片")
        raise RuntimeError(
            "任務完成，但找不到 " + "、".join(missing) + " 的輸出資訊。\n"
            f"完整 history 已寫入：{debug_path}"
        )

    video_path = save_single_video_ref(main_refs, output_dir, "video")
    mask_path = save_single_video_ref(mask_refs, output_dir, "mask")

    # 成功後若之前留有失敗 history，移除避免誤判。
    debug_path = output_dir / "history.json"
    if debug_path.exists():
        debug_path.unlink()

    print(f"主影片：{video_path}")
    print(f"Mask  ：{mask_path}")
    print("完成。")


def main() -> int:
    print("VideoSplat 360 ComfyUI 批次處理")
    print(f"ComfyUI：{config.COMFYUI_URL}")
    print(f"Input ：{config.INPUT_DIR}")
    print(f"Output：{config.OUTPUT_DIR}")

    try:
        check_comfyui()
    except Exception as exc:
        print(f"\n[錯誤] {exc}", file=sys.stderr)
        return 1

    images = discover_input_images()

    if not images:
        print()
        print(
            "input/ 內沒有可處理圖片。\n"
            f"請把人物圖片放到：{config.INPUT_DIR}"
        )
        return 0

    print(f"找到 {len(images)} 位人物。")

    client_id = uuid.uuid4().hex
    failures: list[tuple[Path, str]] = []

    for index, image_path in enumerate(images):
        try:
            process_one(
                image_path=image_path,
                index=index,
                total=len(images),
                client_id=client_id,
            )
        except KeyboardInterrupt:
            print("\n使用者中止。")
            return 130
        except Exception as exc:
            failures.append((image_path, str(exc)))
            print(f"[失敗] {image_path.name}", file=sys.stderr)
            print(str(exc), file=sys.stderr)

    print()
    print("=" * 72)

    if failures:
        print(f"完成，但有 {len(failures)} 位人物失敗：")
        for path, reason in failures:
            print(f"- {path.name}: {reason}")
        return 2

    print("全部人物處理完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
