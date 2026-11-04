# 360° Skeleton Orbit Generator

這個專案會保留原始固定骨架，並針對 `input/` 內每張人物圖，使用 DWPose 偵測身體、臉部與雙手關節，再各自生成彩色骨架環繞動畫與逐幀對應的 COLMAP 相機資料。模型漏點時才使用人物輪廓備援。

## 專案結構

```text
src/1_360_skeleton/
├─ config.py       # 所有可調參數、固定骨架與顏色
├─ main.py         # 生成影片、逐幀 PNG、COLMAP 相機資料
├─ dwpose_backend.py
├─ .pose_env/      # 可選的本機 Python 3.10 環境，不納入 Git
├─ models/         # 首次執行時自動下載的 DWPose ONNX 模型，不納入 Git
├─ third_party/    # DWPose 推論程式
├─ README.md
├─ input/          # 一張圖視為一位人物；可放 PNG/JPG/WEBP/BMP
└─ output/
   ├─ default/     # 原始固定骨架（永遠保留）
   ├─ 人物檔名/   # 每張輸入圖各自一個輸出資料夾
   └─ batch_summary.json
```

## 執行

需求：Python 3.10、專案 requirements 與 FFmpeg。首次使用 DWPose 時，程式會從 Hugging Face 自動下載約 351 MB 的兩個 ONNX 模型到 `models/yzd-v/DWPose/`。下載支援中斷續傳，並會驗證檔案大小與 SHA-256；後續執行不會重複下載。若本機自行建立 `.pose_env/python.exe`，程式會自動切換到該環境；Git 版本不包含虛擬環境或模型權重。

```powershell
python -m pip install -r requirements.txt
python -B src/1_360_skeleton/main.py
```

一般只需要修改 `config.py`：

- `FRAME_COUNT`：總幀數，預設 175。
- `FPS`：預設 24。
- `WIDTH`、`HEIGHT`：預設 720×1280。
- `CAMERA_RADIUS`、`CAMERA_HEIGHT`：固定環繞半徑與高度。
- `CAMERA_TARGET`：鏡頭持續看向的位置。
- `FOCAL_LENGTH_PX`：固定焦距，單位為像素。
- `JOINTS`：固定 3D 姿勢。

## 輸入圖片

1. 將每位人物的全身正面圖放進 `input/`；例如 `alice.png`、`bob.png`。
2. 圖片以單人、完整頭腳、直立姿勢、白色或透明背景效果最好。
3. 執行 `main.py`。程式會把每張圖視為不同人物，輸出到同名資料夾。

對齊優先使用 DWPose 的人體偵測與 COCO-WholeBody 133 關節模型；肩、肘、腕、髖、膝、踝、臉與手指會直接採用模型結果。只有缺失關節、頭頂與足尖使用輪廓推算。每個人物資料夾的 `pose_detection.json` 會記錄方法、各關節座標與信心分數。

輸出是「與人物比例和正面姿勢對齊的 360° 骨架控制圖」，不是從單張正面圖憑空生成角色背面外觀。

## 輸出

```text
output/人物檔名/
├─ pose_control_video_175.mp4       # 該人物的環繞骨架影片
├─ pose_control_frame000.png        # 正面骨架
├─ pose_control_debug.png           # 五個環繞角度預覽
├─ source_aligned.png               # 置中縮放後的原圖
├─ pose_alignment_preview.png       # 原圖與正面骨架疊合檢查
├─ camera_poses.json
├─ generation_summary.json
└─ colmap/
   ├─ images/
   │  ├─ frame000.png
   │  └─ ...
   └─ sparse/0/
      ├─ cameras.txt
      ├─ images.txt
      ├─ points3D.txt
      ├─ rigs.txt
      └─ frames.txt
```

每個人物資料夾內的 `colmap/images/frameNNN.png` 與 `images.txt`、`frames.txt` 使用相同的一對一幀編號。未來將生成影片拆成同樣的 `frame000.png` 至 `frame174.png`，即可按影像名稱與相機外參對齊。

## COLMAP 座標定義

- 世界座標：`+Y` 向上，`+Z` 為人物正面。
- COLMAP 相機座標：`+X` 向右、`+Y` 向下、`+Z` 向前。
- `images.txt` 與 `frames.txt` 儲存的是 COLMAP 標準 world-to-camera：`qvec` 與 `tvec`。
- 攝影機中心可由 `C = -Rᵀt` 還原。
- 相機內參使用 `SIMPLE_PINHOLE`：`f, cx, cy`。
- `points3D.txt` 為空是正常的；本專案輸出已知相機軌跡，不進行特徵匹配或稀疏重建。

## 環繞規則

- Frame 0：0°，人物正面。
- 約 Frame 44：90°，側面。
- 約 Frame 87：180°，背面。
- 約 Frame 131：270°，另一側面。
- Frame 174：360°，回到正面。

角度為 `360 × frame / (FRAME_COUNT - 1)`。人物的 3D 關節完全固定；每幀只改變攝影機外參。

