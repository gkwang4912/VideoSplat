# video_splat（重新實作版）

將單圈人物影片轉換為 Brush 可直接使用的乾淨 RGBA + COLMAP 資料集。這份程式碼是在空的 `old` 資料夾中重新撰寫，不是從備份還原。

## 資料流

1. FFmpeg 均勻抽取最多 192 張 PNG。
2. 從每組輸入的 `mask.mp4` 按幀讀取 mask；不執行 AI 去背。
3. 移除小連通區、選擇性內縮 1 px，建立硬式 0/255 alpha。
4. 只在前景內側窄帶做 RGB defringe，透明區 RGB 強制歸零。
5. CUDA COLMAP 執行特徵、配對、mapper 與文字模型轉換。
6. 若 mapper 產生多個 sparse model，依註冊影像數、再依 3D 點數自動選最佳模型。
7. 驗證註冊率、相機弧、RGBA、檔名、hidden RGB 與 sparse 結構。

## 最終輸出

```text
output/<組別>/brush_dataset_rgba_clean/
├── images/
│   ├── 0001.png
│   └── ...
└── sparse/0/
    ├── cameras.txt
    ├── images.txt
    └── points3D.txt
```

`images` 全部是 RGBA PNG，alpha 只有 0/255，alpha=0 的 RGB 必須為 0。最終資料夾只有 `images` 與 `sparse`。

## ComfyUI 安裝

將本資料夾以 junction 或目錄方式放入：

```text
ComfyUI/custom_nodes/video_splat
```

使用 ComfyUI 的 Python 安裝相依套件：

```powershell
<ComfyUI Python> -m pip install -r requirements.txt
```

每組輸入放在獨立子資料夾中，每組必須各有一支普通影片與一支檔名包含 `mask` 的影片：

```text
input/
├── 1/
│   ├── video.mp4
│   ├── mask.mp4
│   └── source.png       # 可有可無，不參與處理
├── 2/
│   ├── video.mp4
│   └── mask.mp4
└── ...
```

同組兩支影片必須有相同解析度與相同總幀數。FPS 和時長可以不同，程式會按第 N 幀對第 N 幀配對。不同組可使用不同解析度與幀數。

## 節點

- `Video Splat Build Dataset`：讀取來源影片與 mask 影片，執行 COLMAP、RGBA 與驗證。
- `Video Splat Preview Dataset`：顯示實際輸出 contact sheet、代表 mask 與 JSON metadata。

本流程不載入去背模型，只使用輸入的 mask 影片。

## CLI

直接執行時，程式會遞迴掃描 `input`，自動配對並依資料夾順序逐組處理：

```powershell
python main.py
```

目前 `input/1`～`input/5` 會分別輸出為：

```text
output/1/brush_dataset_rgba_clean/
output/2/brush_dataset_rgba_clean/
output/3/brush_dataset_rgba_clean/
output/4/brush_dataset_rgba_clean/
output/5/brush_dataset_rgba_clean/
```

單組失敗不會阻止後續組別執行。全部結束後，摘要寫入專案根目錄的 `batch_report.json`；只要有失敗，程式結束碼會是 1。

也可以明確指定：

```powershell
python main.py input/video.mp4 --mask-video input/mask.mp4 --colmap "C:\path\to\colmap"
```

## 測試

```powershell
python -m unittest discover -s tests -v
```

完整 Video/COLMAP 實跑需要來源影片、mask 影片、FFmpeg 與 CUDA COLMAP；不需要去背模型。
