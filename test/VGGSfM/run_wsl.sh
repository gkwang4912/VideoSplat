#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval "$(conda shell.bash hook)"
conda activate vggsfm
cd "$ROOT_DIR"
python main.py
