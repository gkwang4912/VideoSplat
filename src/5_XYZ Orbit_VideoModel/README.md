# XYZ Orbit 影片模型批次程式

## 直接執行

1. 啟動 ComfyUI，確認網址是 `http://127.0.0.1:8188`。
2. 在本資料夾執行 `python main.py --check`；看到「檢查通過」才進行下一步。
3. 執行 `python main.py`。12 部影片會依序生成，已完成項目會在重跑時自動略過，最後建立 COLMAP 與 Brush 資料集。

完整生成時間取決於影片模型工作流與 GPU；單部最長等待設定為 6 小時。終端每 30 秒顯示一次仍在生成的狀態。

## 輸入對應

- 軌跡影片：`input/All/*.mp4`
- 人物參考圖：`input/test_avater.jpg`
- Camera：`input/paths/<影片檔名>/cameras/*.json`
- ComfyUI API 工作流：`workflows/video_model_latent_refine_api.json`
- 已處理的整合全視角影片：`input/*.mp4`

影片檔名（不含副檔名）必須與 path 資料夾名稱相同，例如：

```text
input/All/01_front_to_left.mp4
input/paths/01_front_to_left/
```

目前 12 部來源影片與 camera 都是 0° 到 90°，含端點共 91 幀。程式接受 ComfyUI 回傳 90 或 91 幀：每個輸出幀依索引使用同度數 camera；90 幀時使用 `000` 到 `089`，不做視角重採樣。

## 輸出內容

```text
output/
├─ generated_videos/
│  └─ 01_front_to_left.mp4
├─ paths/
│  └─ 01_front_to_left/
│     ├─ images/000.png ...
│     ├─ cameras/000.json ...
│     ├─ cameras.json
│     └─ generated.mp4
├─ colmap_dataset/
│  ├─ images/<path>/<frame>.png
│  ├─ sparse/0/cameras.txt
│  ├─ sparse/0/images.txt
│  ├─ sparse/0/points3D.txt
│  ├─ source_videos/
│  └─ metadata/
│     ├─ cameras/<path>.json
│     ├─ manifest.json
│     └─ reference.jpg
├─ brush_dataset/
   ├─ images/<path>/<frame>.png
   ├─ masks/<path>/<frame>.png
   ├─ transforms.json
   └─ README.txt
└─ brush_videos/
   └─ All/
      ├─ All.mp4
      ├─ images/<path>/000.png ...
      ├─ segments/01_front_to_left.mp4 ...
      ├─ transforms.json
      └─ README.txt
```

`colmap_dataset` 已是 COLMAP 標準 text model 結構。可在 COLMAP GUI 使用 `File > Import model` 選擇 `output/colmap_dataset/sparse/0`，影像根目錄選擇 `output/colmap_dataset/images`。`points3D.txt` 為空，因為此流程提供已知相機姿態與影像，但不會虛構 3D 特徵點。

只處理單一路徑時，COLMAP 資料集仍會自動收集 `output/paths` 中所有已完成路徑，不會把先前完成的路徑移除。

`brush_dataset` 使用 Brush 原生支援的 Nerfstudio 格式。直接在 Brush 開啟 `output/brush_dataset` 或其中的 `transforms.json`。相機已由 OpenCV 座標轉成 Nerfstudio/OpenGL 座標。來源遮罩是背景白、人物黑，因此載入時要啟用 Brush 的 `invert masks`／`--invert-masks`；若不需要遮罩，把 `config.json` 的 `brush_copy_source_masks` 改為 `false`。

`input` 根目錄中的影片會視為「已處理完成、依 12 條 path 順序合併」的全視角影片。程式依各 path 的 camera 數量自動切段，所有圖片都從該影片重新拆出，不使用 `output/paths` 的舊圖片或舊遮罩。影片解析度與 camera 原始解析度不同時，內參會按 X/Y 比例自動縮放。每一部整合影片會得到獨立且可直接載入 Brush 的 `output/brush_videos/<影片名稱>` 資料夾。

## 常用命令

```powershell
# 只檢查，不生成
python main.py --check

# 只跑一條軌跡
python main.py --only 01_front_to_left

# 已把生成影片放入 output/generated_videos，只做拆幀與資料集
python main.py --prepare-only

# 不碰影片，只從現有 output/paths 建立 COLMAP 與 Brush
python main.py --datasets-only

# 只建立 Brush 資料集
python main.py --datasets-only --skip-colmap

# 只處理 input 根目錄的整合全視角影片，不呼叫 ComfyUI
python main.py --consolidated-only

# 強制重建整合影片 Brush 資料集
python main.py --consolidated-only --force

# 強制重建指定軌跡及 COLMAP 輸出
python main.py --only 01_front_to_left --force
```

## 修改工作流後

若 ComfyUI 節點 ID 改變，請更新 `config.json` 的 `workflow_nodes`：

- `video`：`LoadVideo`
- `reference_image`：`LoadImage`
- `seed`：含 `seed` 欄位的節點
- `save_video`：`SaveVideo`

程式使用 ComfyUI 的 `/upload/image`、`/prompt`、`/history` 與 `/view` API，不要求把本機 ComfyUI 安裝路徑寫死在設定檔內。
