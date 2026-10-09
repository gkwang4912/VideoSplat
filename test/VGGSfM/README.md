# VGGSfM 人物 360° 環繞影片 Pipeline

## 資料夾

```text
vggsfm_orbit_pipeline/
├─ input/                 # 放 1 個人物環繞影片
├─ output/                # 執行後自動產生
├─ third_party/vggsfm/    # setup_wsl.sh 自動 clone
├─ config.py
├─ main.py
├─ setup_wsl.sh
└─ run_wsl.sh
```

## 建議平台

建議使用 Ubuntu / WSL2 + NVIDIA CUDA。VGGSfM 官方安裝腳本本身採用 bash/conda，且 Windows native 曾有 pycolmap DLL 問題回報。

## 第一次安裝

```bash
cd /path/to/vggsfm_orbit_pipeline
bash setup_wsl.sh
```

## 執行

1. 把唯一一個影片放進 `input/`。
2. `config.py` 預設 `FRAME_STEP = 1`，因此每一幀都會抽出。
3. 執行：

```bash
bash run_wsl.sh
```

或：

```bash
conda activate vggsfm
python main.py
```

## 模式

`MODE = "auto"`：

- <= 400 幀：使用 `demo.py` 全局重建，較適合完整 360° 環繞、需要首尾一致性的資料。
- > 400 幀：使用 `video_demo.py` sliding-window，可處理更長影片。

## 輸出

```text
output/
├─ images/
├─ sparse/
│  └─ 0/
│     ├─ cameras.bin
│     ├─ images.bin
│     └─ points3D.bin
└─ reconstruction_report.json
```

`output/` 可直接作為多數 COLMAP/3DGS/Brush 資料來源。`reconstruction_report.json` 會記錄註冊率、近似角度覆蓋、首尾閉合程度等檢查結果。

## 重要參數

如果顯存不足，優先降低：

```python
GLOBAL_MAX_QUERY_PTS = 2048
VIDEO_MAX_QUERY_PTS = 1024
```

如果長影片的 video mode 仍 OOM，再降低：

```python
INIT_WINDOW_SIZE = 32
WINDOW_SIZE = 16
```

如果影片確實是固定焦距同一台相機，保留：

```python
SHARED_CAMERA = True
CAMERA_TYPE = "SIMPLE_RADIAL"
```

如果你已確認每一幀都有效，希望不要主動剔除幀：

```python
FILTER_INVALID_FRAME = False
FRAME_STEP = 1
```
