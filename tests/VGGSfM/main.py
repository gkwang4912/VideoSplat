from __future__ import annotations

"""
單檔版 VGGSfM 人物環繞影片 Pipeline
=================================

專案資料夾只需要：
    main.py
    input/
        your_video.mp4

執行：
    python main.py

Windows 下會自動把工作交給 WSL2。
第一次執行會自動在 WSL 的 ~/.cache/vggsfm_onefile/ 建立 VGGSfM 執行環境，
不會在此專案資料夾產生 third_party、setup.ps1、run.ps1 或 config.py。

最終輸出：
    output/
        images/
        sparse/
            0/
                cameras.bin
                images.bin
                points3D.bin
        reconstruction_report.json
"""

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


# ============================================================
# 使用者可調參數：全部集中在 main.py 最上方
# ============================================================

# 影片
FRAME_STEP = 1                  # 1 = 每一幀都使用
MAX_FRAMES = None               # None = 不限制
IMAGE_FORMAT = "jpg"            # jpg / png
JPEG_QUALITY = 100
CLEAR_OUTPUT_BEFORE_RUN = True

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"
}

# VGGSfM 模式
# auto   : <= GLOBAL_MAX_FRAMES 使用全局 demo.py
#          >  GLOBAL_MAX_FRAMES 使用 video_demo.py
# global : 永遠使用 demo.py
# video  : 永遠使用 video_demo.py
MODE = "auto"
GLOBAL_MAX_FRAMES = 400

# 同一支影片通常是同一台相機、同一焦段
SHARED_CAMERA = True
CAMERA_TYPE = "SIMPLE_RADIAL"

# 一般 VGGSfM
MIXED_PRECISION = "fp16"
QUERY_METHOD = "aliked"
FINE_TRACKING = True
USE_POSELIB = True

# 你的影片已經事先處理過，因此預設不主動丟掉 frame
FILTER_INVALID_FRAME = False
COMPLEMENT_NON_VISIBLE = True

# 全局模式（你的 243 幀影片會走這裡）
GLOBAL_MAX_QUERY_PTS = 2048
GLOBAL_QUERY_FRAME_NUM = 8
GLOBAL_QUERY_BY_INTERVAL = True
GLOBAL_QUERY_BY_MIDPOINT = False
GLOBAL_BA_ITERS = 2
GLOBAL_ROBUST_REFINE = 2

# 長影片模式
VIDEO_MAX_QUERY_PTS = 1024
VIDEO_QUERY_FRAME_NUM = 6
INIT_WINDOW_SIZE = 32
WINDOW_SIZE = 16
JOINT_BA_INTERVAL = 4

# 結果品質檢查
TARGET_REGISTERED_RATIO = 0.80
TARGET_ANGULAR_COVERAGE_DEG = 330.0
MAX_LOOP_CLOSURE_STEP_RATIO = 6.0

# 視覺化（預設全部關閉，節省 VRAM）
MAKE_REPROJECTION_VIDEO = False
VISUAL_TRACKS = False
GR_VISUALIZE = False
VIZ_VISUALIZE = False


# ============================================================
# 固定設定
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
INPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "output"

# 所有大型依賴都放 WSL HOME，不污染專案資料夾
RUNTIME_ROOT = Path.home() / ".cache" / "vggsfm_onefile"
CONDA_DIR = RUNTIME_ROOT / "miniconda3"
ENV_NAME = "vggsfm"
VGGSFM_REPO = RUNTIME_ROOT / "vggsfm"
INSTALL_MARKER = RUNTIME_ROOT / ".installed_v2"

VGGSFM_ZIP = "https://github.com/facebookresearch/vggsfm/archive/refs/heads/main.zip"
LIGHTGLUE_ZIP = "https://github.com/jytime/LightGlue/archive/refs/heads/main.zip"

COLMAP_FILES = ("cameras.bin", "images.bin", "points3D.bin")


# ============================================================
# 共用工具
# ============================================================

def bool_arg(value: bool) -> str:
    return "true" if value else "false"


def run(
    cmd,
    *,
    cwd: Path | None = None,
    check: bool = True,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    printable = cmd if isinstance(cmd, str) else " ".join(map(str, cmd))
    print(f"      $ {printable}", flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        env=env,
    )


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"      下載：{url}", flush=True)

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    with urllib.request.urlopen(request) as response, target.open("wb") as f:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total else 0
        downloaded = 0
        next_report = 0

        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            f.write(block)
            downloaded += len(block)

            if total_bytes > 0:
                pct = downloaded * 100 // total_bytes
                if pct >= next_report:
                    print(f"        {pct:3d}%", flush=True)
                    next_report += 10


def download_and_extract_github_zip(url: str, target_dir: Path) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="vggsfm_dl_") as td:
        temp_dir = Path(td)
        archive = temp_dir / "repo.zip"
        extracted = temp_dir / "extract"

        download_file(url, archive)

        print("      解壓縮...", flush=True)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(extracted)

        dirs = [p for p in extracted.iterdir() if p.is_dir()]
        if len(dirs) != 1:
            raise RuntimeError(f"GitHub ZIP 結構異常：{url}")

        if target_dir.exists():
            shutil.rmtree(target_dir)

        shutil.move(str(dirs[0]), str(target_dir))


# ============================================================
# Windows → WSL 啟動
# ============================================================

def get_wsl_script_path() -> str:
    result = subprocess.run(
        ["wsl.exe", "wslpath", "-a", str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
        check=True,
    )
    path = result.stdout.strip()
    if not path:
        raise RuntimeError("無法把 main.py 路徑轉換成 WSL 路徑。")
    return path


def launch_from_windows() -> int:
    print("[啟動] 偵測到 Windows，將工作自動交給 WSL2。", flush=True)

    try:
        check = subprocess.run(
            ["wsl.exe", "--status"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print(
            "\n錯誤：找不到 WSL。\n"
            "請先以系統管理員 PowerShell 執行：\n"
            "  wsl --install\n"
            "重新開機並完成 Ubuntu 第一次設定後，再執行 python main.py。",
            file=sys.stderr,
        )
        return 1

    if check.returncode != 0:
        print(
            "\n錯誤：WSL 尚未正常啟用。\n"
            "請先以系統管理員 PowerShell 執行：\n"
            "  wsl --install",
            file=sys.stderr,
        )
        return 1

    try:
        script_wsl = get_wsl_script_path()
    except Exception as exc:
        print(f"\n錯誤：無法取得 WSL 路徑：{exc}", file=sys.stderr)
        return 1

    command = (
        f"python3 {shlex.quote(script_wsl)} --bootstrap"
    )

    print("      WSL 執行 VGGSfM pipeline", flush=True)
    proc = subprocess.run(["wsl.exe", "bash", "-lc", command])
    return proc.returncode


# ============================================================
# WSL 自動建立環境
# ============================================================

def conda_exe() -> Path:
    return CONDA_DIR / "bin" / "conda"


def env_python() -> Path:
    return CONDA_DIR / "envs" / ENV_NAME / "bin" / "python"


def install_miniconda() -> None:
    if conda_exe().exists():
        return

    print("[環境] 安裝 Miniconda（只會做一次）", flush=True)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

    installer = RUNTIME_ROOT / "miniconda.sh"
    url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

    download_file(url, installer)
    run(["bash", str(installer), "-b", "-p", str(CONDA_DIR)])

    try:
        installer.unlink()
    except OSError:
        pass


def conda_run(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return run(
        [
            str(conda_exe()),
            "run",
            "--no-capture-output",
            "-n",
            ENV_NAME,
            *args,
        ],
        check=check,
    )


def conda_env_exists() -> bool:
    return env_python().exists()


def install_runtime() -> None:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

    install_miniconda()

    if not conda_env_exists():
        print("[環境] 建立 Python 3.10 環境", flush=True)
        run([
            str(conda_exe()),
            "create",
            "-y",
            "-n",
            ENV_NAME,
            "python=3.10",
        ])

    if not INSTALL_MARKER.exists():
        print("[環境] 安裝 VGGSfM 相依套件", flush=True)

        run([
            str(conda_exe()),
            "install",
            "-y",
            "-n",
            ENV_NAME,
            "pytorch=2.1.0",
            "torchvision",
            "pytorch-cuda=12.1",
            "-c", "pytorch",
            "-c", "nvidia",
        ])

        conda_run([
            "python", "-m", "pip", "install", "--upgrade", "pip"
        ])

        conda_run([
            "python", "-m", "pip", "install",
            "hydra-core",
            "omegaconf",
            "opencv-python",
            "einops",
            "visdom",
            "tqdm",
            "scipy",
            "plotly",
            "scikit-learn",
            "imageio[ffmpeg]",
            "gradio",
            "trimesh",
            "huggingface_hub",
            "fvcore",
            "iopath",
        ])

        # 官方 install.sh 指定的版本
        conda_run([
            "python", "-m", "pip", "install",
            "numpy==1.26.3",
            "pycolmap==3.10.0",
            "pyceres==2.3",
            "poselib==2.0.2",
        ])

    if not (VGGSFM_REPO / "demo.py").exists():
        print("[環境] 下載 VGGSfM", flush=True)
        download_and_extract_github_zip(VGGSFM_ZIP, VGGSFM_REPO)

    lightglue_dir = VGGSFM_REPO / "dependency" / "LightGlue"
    if not (lightglue_dir / "lightglue").exists():
        print("[環境] 下載 LightGlue", flush=True)
        lightglue_dir.parent.mkdir(parents=True, exist_ok=True)
        download_and_extract_github_zip(LIGHTGLUE_ZIP, lightglue_dir)

    if not INSTALL_MARKER.exists():
        print("[環境] 安裝 LightGlue", flush=True)
        conda_run([
            "python", "-m", "pip", "install", "-e", str(lightglue_dir)
        ])

        # 建立完成標記
        INSTALL_MARKER.write_text("ok\n", encoding="utf-8")

    print("[環境] 檢查 CUDA / VGGSfM", flush=True)
    code = (
        "import torch, cv2, pycolmap, pyceres, poselib; "
        "print('PyTorch:', torch.__version__); "
        "print('CUDA available:', torch.cuda.is_available()); "
        "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); "
        "raise SystemExit(0 if torch.cuda.is_available() else 2)"
    )

    result = conda_run(["python", "-c", code], check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "VGGSfM 環境已建立，但 WSL 裡的 PyTorch 看不到 NVIDIA CUDA GPU。\n"
            "請先確認 Windows NVIDIA 驅動與 WSL2 GPU 支援正常。"
        )


def bootstrap() -> int:
    # 如果已經是在真正的 conda worker 裡，就直接工作
    if os.environ.get("VGGSFM_ONEFILE_WORKER") == "1":
        worker()
        return 0

    install_runtime()

    script = Path(__file__).resolve()
    worker_env = os.environ.copy()
    worker_env["VGGSFM_ONEFILE_WORKER"] = "1"

    print("[啟動] 使用 VGGSfM Python 3.10 環境執行 main.py", flush=True)

    proc = subprocess.run(
        [
            str(conda_exe()),
            "run",
            "--no-capture-output",
            "-n",
            ENV_NAME,
            "python",
            str(script),
            "--worker",
        ],
        env=worker_env,
    )
    return proc.returncode


# ============================================================
# Pipeline
# ============================================================

def find_input_video() -> Path:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        raise FileNotFoundError(
            f"找不到輸入影片。\n請把影片放進：{INPUT_DIR}"
        )

    if len(videos) > 1:
        names = "\n  - ".join(p.name for p in videos)
        raise RuntimeError(
            "input 裡有多個影片。為避免拿錯檔案，請只保留一個：\n  - "
            + names
        )

    return videos[0]


def prepare_output() -> Path:
    if CLEAR_OUTPUT_BEFORE_RUN and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    images_dir = OUTPUT_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def extract_frames(video_path: Path, images_dir: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 無法開啟影片：{video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    saved = 0
    decoded = 0
    ext = IMAGE_FORMAT.lower().lstrip(".")

    if ext not in {"jpg", "jpeg", "png"}:
        raise ValueError("IMAGE_FORMAT 只支援 jpg / png")

    print(f"[1/4] 抽幀：{video_path.name}", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if decoded % FRAME_STEP == 0:
            out = images_dir / f"{saved:06d}.{ext}"

            if ext in {"jpg", "jpeg"}:
                params = [cv2.IMWRITE_JPEG_QUALITY, int(JPEG_QUALITY)]
            else:
                params = [cv2.IMWRITE_PNG_COMPRESSION, 1]

            if not cv2.imwrite(str(out), frame, params):
                cap.release()
                raise RuntimeError(f"寫入圖片失敗：{out}")

            saved += 1

            if MAX_FRAMES is not None and saved >= int(MAX_FRAMES):
                break

        decoded += 1

    cap.release()

    if saved < 3:
        raise RuntimeError(f"只抽出 {saved} 幀，無法進行 SfM。")

    print(f"      原始 FPS: {source_fps:.3f}")
    print(f"      原始幀數: {source_frame_count}")
    print(f"      使用幀數: {saved} (FRAME_STEP={FRAME_STEP})")

    return {
        "video": video_path.name,
        "source_fps": source_fps,
        "source_frame_count": source_frame_count,
        "decoded_frames": decoded,
        "used_frames": saved,
        "frame_step": FRAME_STEP,
        "width": width,
        "height": height,
    }


def choose_mode(frame_count: int) -> str:
    mode = MODE.lower().strip()

    if mode not in {"auto", "global", "video"}:
        raise ValueError("MODE 必須是 auto / global / video")

    if mode == "auto":
        return "global" if frame_count <= GLOBAL_MAX_FRAMES else "video"

    return mode


def build_vggsfm_command(mode: str) -> list[str]:
    launcher = VGGSFM_REPO / ("demo.py" if mode == "global" else "video_demo.py")
    scene = str(OUTPUT_DIR.resolve())

    cmd = [
        "python",
        str(launcher),
        f"SCENE_DIR={scene}",
        f"shared_camera={bool_arg(SHARED_CAMERA)}",
        f"camera_type={CAMERA_TYPE}",
        f"mixed_precision={MIXED_PRECISION}",
        f"query_method={QUERY_METHOD}",
        f"fine_tracking={bool_arg(FINE_TRACKING)}",
        f"use_poselib={bool_arg(USE_POSELIB)}",
        f"filter_invalid_frame={bool_arg(FILTER_INVALID_FRAME)}",
        f"comple_nonvis={bool_arg(COMPLEMENT_NON_VISIBLE)}",
        f"make_reproj_video={bool_arg(MAKE_REPROJECTION_VIDEO)}",
        f"visual_tracks={bool_arg(VISUAL_TRACKS)}",
        f"gr_visualize={bool_arg(GR_VISUALIZE)}",
        f"viz_visualize={bool_arg(VIZ_VISUALIZE)}",
        "dense_depth=false",
        "save_to_disk=true",
    ]

    if mode == "global":
        cmd.extend([
            f"max_query_pts={GLOBAL_MAX_QUERY_PTS}",
            f"query_frame_num={GLOBAL_QUERY_FRAME_NUM}",
            f"query_by_interval={bool_arg(GLOBAL_QUERY_BY_INTERVAL)}",
            f"query_by_midpoint={bool_arg(GLOBAL_QUERY_BY_MIDPOINT)}",
            f"BA_iters={GLOBAL_BA_ITERS}",
            f"robust_refine={GLOBAL_ROBUST_REFINE}",
        ])
    else:
        cmd.extend([
            f"max_query_pts={VIDEO_MAX_QUERY_PTS}",
            f"query_frame_num={VIDEO_QUERY_FRAME_NUM}",
            f"init_window_size={INIT_WINDOW_SIZE}",
            f"window_size={WINDOW_SIZE}",
            f"joint_BA_interval={JOINT_BA_INTERVAL}",
        ])

    return cmd


def run_vggsfm(mode: str) -> None:
    launcher = VGGSFM_REPO / ("demo.py" if mode == "global" else "video_demo.py")
    if not launcher.exists():
        raise FileNotFoundError(f"找不到 VGGSfM launcher：{launcher}")

    print(f"[2/4] 執行 VGGSfM（模式：{mode}）", flush=True)

    cmd = build_vggsfm_command(mode)
    result = conda_run(cmd, check=False)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def normalize_sparse_layout() -> Path:
    sparse = OUTPUT_DIR / "sparse"
    sparse0 = sparse / "0"

    # 若新版/其他版本本來就輸出 sparse/0，就直接保留
    if all((sparse0 / name).exists() for name in COLMAP_FILES):
        return sparse0

    sparse0.mkdir(parents=True, exist_ok=True)

    # VGGSfM 官方通常輸出到 sparse/ 根目錄
    for name in COLMAP_FILES:
        src = sparse / name
        dst = sparse0 / name

        if src.exists():
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))

    missing = [
        name for name in COLMAP_FILES
        if not (sparse0 / name).exists()
    ]

    if missing:
        raise RuntimeError(
            "VGGSfM 執行結束，但缺少 COLMAP 輸出："
            + ", ".join(missing)
        )

    return sparse0


def trajectory_metrics(centers):
    import numpy as np

    centers = np.asarray(centers, dtype=np.float64)

    if len(centers) < 4:
        return {}

    center = centers.mean(axis=0)
    x = centers - center

    _, _, vt = np.linalg.svd(x, full_matrices=False)

    plane_xy = x @ vt[:2].T
    plane_z = x @ vt[2]

    radii = np.linalg.norm(plane_xy, axis=1)
    mean_radius = float(radii.mean())

    radius_cv = (
        float(radii.std() / mean_radius)
        if mean_radius > 1e-12 else None
    )

    plane_rmse = float(np.sqrt(np.mean(plane_z ** 2)))
    plane_rmse_ratio = (
        float(plane_rmse / mean_radius)
        if mean_radius > 1e-12 else None
    )

    angles = np.mod(
        np.degrees(np.arctan2(plane_xy[:, 1], plane_xy[:, 0])),
        360.0,
    )
    angles = np.sort(angles)

    gaps = np.diff(np.r_[angles, angles[0] + 360.0])
    max_gap = float(gaps.max())
    angular_coverage = float(360.0 - max_gap)

    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    positive_steps = steps[steps > 1e-12]

    median_step = (
        float(np.median(positive_steps))
        if len(positive_steps) else 0.0
    )

    loop_distance = float(np.linalg.norm(centers[-1] - centers[0]))
    loop_ratio = (
        float(loop_distance / median_step)
        if median_step > 1e-12 else None
    )

    return {
        "angular_coverage_deg": angular_coverage,
        "largest_angular_gap_deg": max_gap,
        "radius_cv": radius_cv,
        "plane_rmse_ratio": plane_rmse_ratio,
        "median_camera_step": median_step,
        "first_last_distance": loop_distance,
        "loop_closure_step_ratio": loop_ratio,
    }


def inspect_reconstruction(
    sparse0: Path,
    extraction_info: dict,
    mode: str,
) -> dict:
    import numpy as np
    import pycolmap

    print("[3/4] 檢查 COLMAP / 相機路徑", flush=True)

    try:
        rec = pycolmap.Reconstruction(str(sparse0))
    except Exception:
        rec = pycolmap.Reconstruction()
        rec.read(str(sparse0))

    images = sorted(rec.images.values(), key=lambda image: image.name)

    registered = len(images)
    used_frames = int(extraction_info["used_frames"])
    ratio = registered / used_frames if used_frames else 0.0

    centers = []

    for image in images:
        try:
            c = np.asarray(
                image.projection_center(),
                dtype=np.float64,
            ).reshape(3)
            centers.append(c)
        except Exception:
            pass

    path_stats = trajectory_metrics(centers) if centers else {}

    warnings = []

    if ratio < TARGET_REGISTERED_RATIO:
        warnings.append(
            f"註冊率 {ratio:.1%} 低於目標 "
            f"{TARGET_REGISTERED_RATIO:.0%}。"
        )

    coverage = path_stats.get("angular_coverage_deg")

    if (
        coverage is not None
        and coverage < TARGET_ANGULAR_COVERAGE_DEG
    ):
        warnings.append(
            f"估計角度覆蓋約 {coverage:.1f}°，低於 "
            f"{TARGET_ANGULAR_COVERAGE_DEG:.0f}°。"
        )

    loop_ratio = path_stats.get("loop_closure_step_ratio")

    if (
        loop_ratio is not None
        and loop_ratio > MAX_LOOP_CLOSURE_STEP_RATIO
    ):
        warnings.append(
            f"首尾相機距離約為典型相鄰步長的 "
            f"{loop_ratio:.2f} 倍；360° 軌跡可能沒有良好閉合。"
        )

    report = {
        "mode": mode,
        "input": extraction_info,
        "reconstruction": {
            "registered_frames": registered,
            "registration_ratio": ratio,
            "camera_models": len(rec.cameras),
            "points3D": len(rec.points3D),
            **path_stats,
        },
        "warnings": warnings,
    }

    report_path = OUTPUT_DIR / "reconstruction_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"      已註冊：{registered}/{used_frames} "
        f"({ratio:.1%})"
    )
    print(f"      3D points：{len(rec.points3D)}")

    if coverage is not None:
        print(f"      估計角度覆蓋：{coverage:.1f}°")

    if loop_ratio is not None:
        print(
            "      首尾距離 / 中位相鄰步長："
            f"{loop_ratio:.2f}"
        )

    for warning in warnings:
        print(f"      [警告] {warning}")

    return report


def worker() -> None:
    # 確認 worker 真的是 Linux/WSL
    if sys.platform == "win32":
        raise RuntimeError("worker 不應在 Windows 原生 Python 執行。")

    # CUDA 再確認一次
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch 看不到 CUDA GPU，無法執行 VGGSfM。"
        )

    print(f"[GPU] {torch.cuda.get_device_name(0)}", flush=True)

    video = find_input_video()
    images_dir = prepare_output()
    extraction_info = extract_frames(video, images_dir)

    mode = choose_mode(extraction_info["used_frames"])

    if mode == "global":
        print(
            f"      {extraction_info['used_frames']} 幀 <= "
            f"{GLOBAL_MAX_FRAMES}，使用全局 VGGSfM。",
            flush=True,
        )
    else:
        print(
            f"      {extraction_info['used_frames']} 幀，"
            "使用 VGGSfM Video Runner。",
            flush=True,
        )

    run_vggsfm(mode)

    sparse0 = normalize_sparse_layout()
    inspect_reconstruction(sparse0, extraction_info, mode)

    print("[4/4] 完成", flush=True)
    print(f"      Images : {OUTPUT_DIR / 'images'}")
    print(f"      COLMAP : {sparse0}")
    print(
        f"      Report : "
        f"{OUTPUT_DIR / 'reconstruction_report.json'}"
    )


# ============================================================
# main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if sys.platform == "win32":
        return launch_from_windows()

    # WSL / Linux
    if args.worker:
        worker()
        return 0

    return bootstrap()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n使用者中止。", file=sys.stderr)
        sys.exit(130)
    except subprocess.CalledProcessError as exc:
        print(
            f"\nVGGSfM 執行失敗，return code = "
            f"{exc.returncode}",
            file=sys.stderr,
        )
        sys.exit(exc.returncode or 1)
    except Exception as exc:
        print(f"\n錯誤：{exc}", file=sys.stderr)
        sys.exit(1)
