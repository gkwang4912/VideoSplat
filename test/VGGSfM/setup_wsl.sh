#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$ROOT_DIR/third_party/vggsfm"
ENV_NAME="vggsfm"

if ! command -v conda >/dev/null 2>&1; then
  echo "找不到 conda。請先在 WSL/Ubuntu 安裝 Miniconda 或 Anaconda。"
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"

# 依照 VGGSfM 官方 install.sh 的主要版本組合。
conda install -y pytorch=2.1.0 torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

python -m pip install --upgrade pip
python -m pip install \
  hydra-core omegaconf opencv-python einops visdom tqdm scipy plotly \
  scikit-learn 'imageio[ffmpeg]' gradio trimesh huggingface_hub

python -m pip install numpy==1.26.3 pycolmap==3.10.0 pyceres==2.3 poselib==2.0.2

mkdir -p "$(dirname "$REPO_DIR")"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone https://github.com/facebookresearch/vggsfm.git "$REPO_DIR"
else
  git -C "$REPO_DIR" pull --ff-only
fi

mkdir -p "$REPO_DIR/dependency"
if [ ! -d "$REPO_DIR/dependency/LightGlue/.git" ]; then
  git clone https://github.com/jytime/LightGlue.git "$REPO_DIR/dependency/LightGlue"
fi
python -m pip install -e "$REPO_DIR/dependency/LightGlue"
python -m pip install -e "$REPO_DIR"

# PyTorch3D 只用於 Visdom 視覺化；本 pipeline 預設關閉 viz_visualize，故不強制安裝。
python - <<'PY'
import torch
import cv2
import pycolmap
import pyceres
import poselib
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("pycolmap:", pycolmap.__version__)
print("VGGSfM 環境基本 import 測試完成。")
PY

echo
echo "安裝完成。"
echo "之後執行："
echo "  conda activate $ENV_NAME"
echo "  cd '$ROOT_DIR'"
echo "  python main.py"
