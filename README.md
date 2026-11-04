# VideoSplat

**V2.0.0 Release Candidate · Unreleased**

[繁體中文](README.md) | [English](README_EN.md)

## 專案總覽（Project Overview）

VideoSplat 把一張全身角色圖轉成可供 Gaussian Splatting 訓練的多視角資料，再以 Brush 匯出 PLY。這個工作區目前的實作分成五個可獨立執行的 Python 工具：骨架與相機建立、角色影片生成、影片遮罩生成、RGBA 資料集組裝、Gaussian 訓練。

設計重點是先定義相機，再產生受骨架控制的影像。骨架、RGB 影片、mask 影片與相機記錄以逐幀順序配對，避免在生成後另行估計相機。使用者主要是需要檢查控制條件、重建資料契約與 3DGS 結果的研究者及工程師；專案性質為本機研究工具鏈（tool），各階段透過磁碟檔案交接。

### 循環預覽

以下直接以圖片嵌入 GIF，不加額外的連結包裝。三個檔案均為無限循環動畫（`loop=0`）；完整 MP4 另列於下表。

骨架預覽由 `src/4_virtual_camera/input/4/pose_control_video_175.mp4` 轉出，展示用 MP4 也使用這個來源的完整副本。

<table>
  <tr>
    <th>骨架控制／Skeleton control</th>
    <th>歷史角色展示／Legacy character</th>
    <th>歷史高斯展示／Legacy Gaussian render</th>
  </tr>
  <tr>
    <td><img src="docs/assets/demo/v2/pose_control_video_175.gif" width="260" alt="骨架控制 GIF"></td>
    <td><img src="docs/assets/demo/character-2-orbit.gif" width="260" alt="角色 2 的歷史環繞展示 GIF"></td>
    <td><img src="docs/assets/demo/character-2-3d.gif" width="260" alt="角色 2 的歷史 3DGS 模型渲染 GIF"></td>
  </tr>
</table>

| 素材 | GIF 規格 | 對應 MP4 |
|---|---|---|
| 骨架控制 | 320×569，175 個動畫影格 | [720×1280，175 幀，24 FPS](docs/assets/demo/v2/pose_control_video_175.mp4) |
| 歷史角色展示 | 320×674，64 個動畫影格 | [608×1280，192 幀，24 FPS](docs/assets/demo/character-2-orbit.mp4) |
| 歷史 Gaussian 展示 | 320×674，60 個動畫影格 | [608×1280，361 幀，30 FPS](docs/assets/demo/character-2-3d.mp4) |

第二、三欄使用 `docs/assets/demo/` 原有的角色 2 展示，不是目前角色 4 的生成或重建結果。骨架欄繼續使用上述角色 4 的控制影片。GIF 是展示用縮圖，其幀數與解析度不代表現行訓練資料規格。

## 系統架構說明（Architecture Overview）

五個 Stage 各自從同目錄的 `config.py` 讀取路徑與參數。Stage 2、3 是 ComfyUI HTTP client；Stage 1、4 使用 FFmpeg；Stage 5 啟動本機 Brush 執行檔。訓練以外另有 CPU 渲染工具，可從 PLY 與原始 COLMAP 相機重現展示影片。

```mermaid
flowchart TB
    Image["全身角色圖片"] --> S1["1_360_skeleton"]
    Models["DWPose ONNX 與輪廓備援"] --> S1
    S1 -->|"對齊圖片與骨架影片"| S2["2_suround_video"]
    Comfy["ComfyUI HTTP API"] --> S2
    S2 -->|"RGB 影片"| S3["3_remove_background"]
    Comfy --> S3
    S2 -->|"RGB 影片"| S4["4_virtual_camera"]
    S3 -->|"mask 影片"| S4
    S1 -->|"原始 COLMAP 相機"| S4
    S4 -->|"人工複製 brush_dataset"| S5["5_360brush"]
    Brush["brush_app.exe"] --> S5
    S5 --> PLY["Gaussian PLY"]
    PLY --> Render["render_gaussian_reference.py"]
    S5 -->|"input 中的原始相機"| Render
    Render --> Video["PNG 影格與 MP4"]
```

圖中的跨 Stage 箭頭是檔案交接，並非程式自動搬移。所有 Stage 都有自己的 `input/`、`output/`；尤其 Stage 5 預設讀取 `src/5_360brush/input/`，沒有直接掃描 Stage 4 的輸出。

| 邊界 | 上游負責 | 下游要求 |
|---|---|---|
| 1 → 2 | 對齊圖與骨架影片 | 每個第一層角色目錄同時包含兩個固定檔名 |
| 2 → 3 | 生成的 RGB 影片 | 將選定版本放進 Stage 3 input；保留角色子目錄 |
| 1、2、3 → 4 | 原始相機、RGB、mask | 同一角色目錄內提供相機模型與可唯一辨識的影片配對 |
| 4 → 5 | RGBA dataset | 手動複製完整 `brush_dataset/` 到 Stage 5 input |
| 5 → 渲染 | PLY 與訓練時相機 | 明確指定 PLY、COLMAP 路徑及輸出目錄 |

## 系統流程說明（System Flow）

一次處理的資料單位是角色資料夾。完整運作順序如下：

1. Stage 1 先輸出 `default` 固定骨架，再遞迴處理輸入圖片。前景偵測與置中完成後，採用 DWPose 關節；缺失關節使用輪廓補齊，模型初始化或偵測失敗則改用輪廓備援。
2. Stage 2 掃描第一層子目錄，以 `source_aligned.png` 與 `pose_control_video_175.mp4` 建立工作。檢查 ComfyUI 節點後上傳、提交 prompt、輪詢 history、下載 RGB 影片；目前 API workflow 只有一個 `SaveVideo`。
3. Stage 3 遞迴尋找影片。workflow 以第一幀的 BiRefNet mask 初始化 SAM3 追蹤，再把追蹤結果存成 mask 影片。輸出沿用輸入相對目錄及檔名主體。
4. Stage 4 抽樣影片像素辨識唯一的黑白 mask，再以寬、高、影格數配對唯一 RGB 影片。兩段影片及 COLMAP pose 都必須各有 175 筆；逐幀把 mask 的灰階值寫進 RGB 的 alpha，按原始註冊名稱保存 RGBA PNG。
5. Stage 5 檢查資料夾及 Brush flags，逐組執行訓練、記錄 `brush.log`，確認最後 PLY 存在且非空。渲染工具另行讀取這個 PLY 與原始相機，輸出逐幀影像及影片。

```mermaid
flowchart TD
    Start["角色圖片"] --> Align["偵測前景並對齊"]
    Align --> Pose{"DWPose 初始化與偵測成功？"}
    Pose -->|"是"| ModelPose["模型關節與輪廓補漏"]
    Pose -->|"否"| Fallback["輪廓姿態估計"]
    ModelPose --> Known["175 幀骨架與相機"]
    Fallback --> Known
    Known --> Generate["ComfyUI 生成 RGB"]
    Generate --> Mask["BiRefNet 初始化 SAM3 並輸出 mask"]
    Mask --> Pair{"影片配對唯一且 RGB、mask、pose 都是 175 筆？"}
    Pair -->|"否"| Stop["停止並輸出錯誤"]
    Pair -->|"是"| RGBA["按 pose 順序建立 RGBA"]
    RGBA --> Validate{"RGBA、註冊名稱與 sparse 檔案檢查通過？"}
    Validate -->|"否"| Stop
    Validate -->|"是"| Dataset["輸出 dataset 與 ZIP"]
    Dataset --> Copy["人工複製到 Stage 5 input"]
    Copy --> Train["Brush 訓練與 PLY 檢查"]
    Train --> Preview["沿原始相機渲染"]
```

| 情況 | 現有行為 |
|---|---|
| Stage 1 單張圖片處理失敗 | 繼續其他圖片；在 `batch_summary.json` 記錄失敗，最後拋出錯誤 |
| Stage 2 只有一個必要輸入檔 | 回報不完整目錄並停止；完全不含兩個必要檔名的目錄則略過 |
| Stage 2 已完成工作 | 以 workflow、圖片、影片的 SHA-256 簽章及 `.completed.json` 判斷是否可略過 |
| Stage 3 已有同名影片 | 預設略過；`--force` 會重新生成並覆寫，未比對輸入內容簽章 |
| Stage 4 任一組不合法 | 停止整批；先在暫存目錄組裝並驗證，通過後才取代該組 dataset |
| Stage 5 單組失敗 | 預設記錄失敗並繼續下一組；最後以非零結束碼回報 |

## 資料夾結構說明（Folder Structure）

以下結構同時標示程式目錄與目前工作區中的本機資料；不代表這些資料都已納入 Git。

```text
VideoSplat/
|-- src/
|   |-- 1_360_skeleton/
|   |   |-- main.py, config.py, dwpose_backend.py, README.md
|   |   |-- models/yzd-v/DWPose/
|   |   |-- third_party/comfyui_controlnet_aux/
|   |   |-- input/
|   |   `-- output/
|   |-- 2_suround_video/
|   |   |-- main.py, config.py
|   |   |-- workflows/VideoSplat_2.json
|   |   |-- workflows/VideoSplat_2_api.json
|   |   |-- input/
|   |   `-- output/
|   |-- 3_remove_background/
|   |   |-- main.py, config.py
|   |   |-- workflows/Sam3_remove_background.json
|   |   |-- workflows/Sam3_remove_background_api.json
|   |   |-- input/
|   |   `-- output/
|   |-- 4_virtual_camera/
|   |   |-- main.py, config.py
|   |   |-- input/
|   |   `-- output/
|   `-- 5_360brush/
|       |-- main.py, config.py
|       |-- brush_app.exe, brush-cli.exe
|       |-- input/
|       |-- output/
|       `-- target/autotune/
|-- tools/
|   |-- render_gaussian_reference.py
|   |-- render_gaussian_reference.md
|   |-- render_gaussian_orbit.py
|   `-- requirements-renderer.txt
|-- docs/
|   |-- assets/demo/
|   |   `-- v2/
|   `-- versions/
|-- test_file/
|   |-- Avater/black/
|   |-- Avater/white/
|   |-- suround/1/ ... 5/
|   `-- 1080/
|-- requirements.txt
|-- README.md, README_EN.md, CHANGELOG.md
`-- .gitignore, .gitattributes
```

| 位置 | 逐層責任 |
|---|---|
| `src/1_360_skeleton/models/` | 模型說明；`yzd-v/DWPose/` 保存下載的 detector 與 pose ONNX |
| `third_party/comfyui_controlnet_aux/src/custom_controlnet_aux/dwpose/` | vendored 姿態推論；`dw_onnx/` 是 ONNX 前後處理，`dw_torchscript/` 是另一路 TorchScript 實作 |
| 各 Stage 的 `input/`、`output/` | 執行時資料；角色 group 的意義由對應程式的 discovery 函式決定 |
| `src/5_360brush/target/autotune/` | 本機現有的 Brush／CubeCL autotune 快取 |
| `tools/` | PLY 離線渲染程式、重現說明及獨立依賴清單 |
| `docs/assets/demo/`、`v2/` | 舊展示與目前三個內嵌預覽的 GIF、MP4、除錯 PNG |
| `docs/versions/` | V1 文件快照、V1/V2 說明、架構、遷移、release notes 與版本政策；部分內容仍描述先前的四階段設計 |
| `test_file/` | 人物圖片與影片樣本，沒有測試 runner；`Avater`、`suround` 為實際拼字 |
| `test_file/suround/1/ ... 5/` | RGB／mask／角色圖片組；其中第 4 組圖片實際名為 `souce.png` |
| `test_file/1080/` | 三個 `video_latent_refine_*.mp4` 影片樣本 |

Stage 4 交給 Brush 的目錄只有 RGBA 影像與 COLMAP 模型，不含獨立 `masks/`：

```text
src/4_virtual_camera/output/<group>/
|-- brush_dataset/
|   |-- images/frame000.png ... frame174.png
|   `-- sparse/0/
|       |-- cameras.txt
|       |-- images.txt
|       `-- points3D.txt
`-- brush_dataset.zip
```

影像名稱實際沿用 `images.txt`；上例的 `frameNNN.png` 是 Stage 1 預設命名。複製到 Stage 5 後，入口為 `src/5_360brush/input/<group>/brush_dataset/`。

## 核心模組與重要檔案（Key Modules & Files）

| 檔案或函式 | 主要責任與關聯 |
|---|---|
| [Stage 1 main.py](src/1_360_skeleton/main.py) | `camera_for_frame` 建立外參，`write_colmap_model` 匯出相機，`generate_character` 產生影片與診斷資料 |
| [dwpose_backend.py](src/1_360_skeleton/dwpose_backend.py) | 驗證模型大小與 SHA-256、續傳下載、載入 `Wholebody`；避免執行第三方 package 不需要的初始化 |
| [Stage 2 main.py](src/2_suround_video/main.py) | `discover_jobs` 自動配對；`job_signature` 與完成記錄管理重跑；使用標準函式庫呼叫 ComfyUI |
| [生成 API workflow](src/2_suround_video/workflows/VideoSplat_2_api.json) | 32 個節點；MiniMax H3 reference-to-video、骨架 ControlNet、LoRA、記憶體最佳化與單一影片輸出 |
| [Stage 3 main.py](src/3_remove_background/main.py) | 遞迴影片 discovery、串流 multipart 上傳、保留相對目錄的下載與 `--force` 行為 |
| [去背 API workflow](src/3_remove_background/workflows/Sam3_remove_background_api.json) | 12 個節點；`VHS_LoadVideo`、BiRefNet 首幀 mask、SAM3 追蹤與 mask 影片保存 |
| [Stage 4 main.py](src/4_virtual_camera/main.py) | `detect_video_pair` 以內容／規格辨識配對；`write_rgba_images` 合成 alpha；`build_dataset` 驗證與匯出 |
| [Stage 5 main.py](src/5_360brush/main.py) | `detect_steps_flag`、`build_command`、`run_brush` 負責 CLI 相容性、訓練命令與 log |
| [目前預覽渲染器](tools/render_gaussian_reference.py) | 按數字順序讀 SH 欄位、重建協方差、投影橢圓、深度排序與 alpha 混合；要求 `--colmap` |
| [另一個離線腳本](tools/render_gaussian_orbit.py) | 內建自動環繞取景；其 vertical-axis parser 預期數字向量，不能直接處理本次 PLY 的 `Vertical axis: y` |
| [渲染說明](tools/render_gaussian_reference.md) | 描述輸出、相機模式及重現預覽的命令 |

```mermaid
flowchart TB
    C1["Stage 1 config.py"] --> M1["Stage 1 main.py"]
    M1 --> DW["dwpose_backend.py"]
    DW --> WB["vendored Wholebody"]
    C2["Stage 2 config.py"] --> M2["Stage 2 main.py"]
    W2["VideoSplat_2_api.json"] --> M2
    C3["Stage 3 config.py"] --> M3["Stage 3 main.py"]
    W3["Sam3_remove_background_api.json"] --> M3
    M2 --> API["ComfyUI HTTP API"]
    M3 --> API
    C4["Stage 4 config.py"] --> M4["Stage 4 main.py"]
    M4 --> Media["FFmpeg 與 FFprobe"]
    C5["Stage 5 config.py"] --> M5["Stage 5 main.py"]
    M5 --> EXE["BRUSH_EXE"]
    P["PLY 與原始 COLMAP 檔案"] --> R["render_gaussian_reference.py"]
    R --> CPU["NumPy / SciPy / Numba / Pillow"]
    R --> Encode["FFmpeg MP4 編碼"]
```

## 安裝與環境需求（Installation & Requirements）

### 執行環境

| 項目 | 可由目前檔案確認的要求 |
|---|---|
| Python | 至少 3.10，Stage 4 使用 `zip(..., strict=True)`；完整相依套件相容版號尚未定義 |
| 作業系統 | 現有 Brush 設定指向 Windows `.exe`；其他系統的完整部署流程尚未定義 |
| FFmpeg / FFprobe | 需在 PATH；Stage 1 編碼、Stage 4 偵測與解碼、渲染器輸出影片使用 |
| ComfyUI | Stage 2、3 各設定預設 `http://127.0.0.1:8188`；必要節點須出現在 `/object_info` |
| GPU / VRAM | 影片模型與 Brush 訓練的最低需求尚未定義；CPU 預覽渲染器不要求 CUDA |
| Brush | 工作區現有 `brush_app.exe` 與 `brush-cli.exe`；Stage 5 實際使用前者，啟動前驗證 flags |

從根目錄安裝 Python 依賴：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r tools/requirements-renderer.txt
ffmpeg -version
ffprobe -version
```

根 `requirements.txt` 透過 `-r` 引入 vendored requirements，包含 Torch、TorchVision、OpenCV、NumPy、Pillow、SciPy、Hugging Face Hub、MediaPipe、ONNX Runtime 等。渲染清單另外列出 NumPy、SciPy、Pillow、plyfile、Numba。這兩份清單均無完整版本鎖定。

### 模型與服務準備

DWPose 的 `yolox_l.onnx`、`dw-ll_ucoco_384.onnx` 由 backend 在需要時下載到 `models/yzd-v/DWPose/`；合計大小為 351,145,849 bytes。下載以 `.part` 續傳，驗證大小與 SHA-256，最多嘗試三次。也可事先放入符合驗證的檔案。

ComfyUI 的模型實際名稱來自兩個 API workflow：

| 階段 | 模型欄位內容 |
|---|---|
| 生成 | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| 生成 | `minimax_h3_video_vae_int8_convrot.safetensors` |
| 生成 | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| 生成 | `pika_h3_flex_ref2va_comfy_bf16.safetensors` |
| 生成 | `minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors` |
| mask | `BiRefNet-general`、`sam3.1_multiplex_fp16.safetensors` |

custom node 安裝來源與精確版本尚未由專案統一規範；以 workflow 的 `class_type` 和各 Stage 的 `--check` 核對節點完整性。`--check` 不會驗證所有模型檔內容，也不代表已完成生成。

沒有專案層級的必要 API key 或 `.env` schema。已使用的環境變數包括 `AUX_ORT_PROVIDERS`；backend 預設設為 `CPUExecutionProvider`。Stage 4 的 `VIDEOSPLAT_BRUSH_EXE` 只影響它印出的建議命令，不會改變 Stage 5 的 `BRUSH_EXE`。

## 使用方式（How to Use）

所有命令從 VideoSplat 根目錄執行。`<group>` 表示同一個角色的資料夾名稱；目前已存在的角色名稱為 `4`。

1. 將角色圖放入 `src/1_360_skeleton/input/`。目前的 `4.png` 會對應 `output/4/`。執行：

   ```powershell
   python -B src/1_360_skeleton/main.py
   ```

   檢查 `source_aligned.png`、`pose_alignment_preview.png` 與 `pose_detection.json`。保留該組 `colmap/sparse/0/`，後續必須沿用。

2. 將該組 `source_aligned.png`、`pose_control_video_175.mp4` 複製到 `src/2_suround_video/input/<group>/`，啟動已備妥節點與模型的 ComfyUI，執行：

   ```powershell
   python -B src/2_suround_video/main.py --check
   python -B src/2_suround_video/main.py
   ```

   RGB 影片下載到 `output/<group>/`。程式自動掃描角色目錄，沒有 `INPUT_JOBS` 名單需要編輯。

3. 將選定的 RGB 影片放進 `src/3_remove_background/input/<group>/`，執行：

   ```powershell
   python -B src/3_remove_background/main.py --check
   python -B src/3_remove_background/main.py
   ```

   相對位置相同的 `output/<group>/` 會產生 mask 影片。需要重做時加 `--force`；這會覆寫對應輸出。

4. 在 `src/4_virtual_camera/input/<group>/` 放入選定 RGB 與 mask，使用不同檔名保存；再放入 Stage 1 的 `colmap/sparse/0/`。執行：

   ```powershell
   python -B src/4_virtual_camera/main.py
   ```

   只放需要配對的影片版本，避免多個 RGB 候選導致歧義。輸出是 `output/<group>/brush_dataset/` 與 ZIP；預設會取代既有 dataset。

5. 把完整 `brush_dataset/` 複製到 `src/5_360brush/input/<group>/brush_dataset/`，確認 `config.py` 的 Brush 路徑後執行：

   ```powershell
   python -B src/5_360brush/main.py
   ```

   預設最終檔為 `src/5_360brush/output/<group>/<group>.ply`，同目錄保留 `brush.log`。`SKIP_COMPLETED=False`，重跑會再次訓練。

### 用訓練相機渲染現有模型

```powershell
python -B tools/render_gaussian_reference.py src/5_360brush/output/4/4.ply src/5_360brush/output/4/source_camera_render --colmap src/5_360brush/input/4/brush_dataset/sparse/0 --width 1080 --height 1920 --fps 24 --supersample 2
```

`--preview` 僅輸出四個診斷視角。正式執行依 `images.txt` 的順序輸出所有影格，並產生 MP4、alpha PNG 與 `render_manifest.json`；manifest 記錄來源雜湊、相機與每幀統計。輸出寬高須保持原相機長寬比。`--frames` 會被實際 pose 數量覆蓋。腳本不會自動替換 README 的 GIF。

## 設定說明（Configuration）

| 位置 | 目前值或項目 | 修改後的影響 |
|---|---|---|
| Stage 1 `config.py` | 720×1280、24 FPS、175 幀；相機半徑 4.8、高度 1.45、焦距 1850 px | 改變控制影片、相機內外參與投影 |
| Stage 1 `config.py` | `USE_DWPOSE`、`POSE_MIN_CONFIDENCE`、前景門檻、`JOINTS`、`DEBUG_FRAMES` | 控制偵測、備援姿態與診斷圖；改幀數時要同步除錯影格索引 |
| Stage 2 `config.py` | 兩個輸入固定檔名、ComfyUI URL、輪詢 2 秒、生成 timeout 6 小時 | 控制 discovery 與 API 等待；HTTP request timeout 為 30 秒 |
| Stage 2 API JSON | prompt、模型、noise seed、0.7 倍且取 32 倍數的影像尺寸、影片長度運算 | 控制生成結果；目前 7.1 秒參數經 expression 對齊為 175 幀，並非直接讀 Stage 1 config |
| Stage 3 `config.py` | 副檔名、上傳 chunk 1 MiB、上傳 timeout 30 分鐘、生成 timeout 6 小時 | 控制遞迴搜尋、串流上傳與服務等待 |
| Stage 3 API JSON | BiRefNet、SAM3 detection threshold 0.5、max objects 4、24 FPS | 控制 mask 初始化、追蹤與保存 |
| Stage 4 `config.py` | 5 個抽樣幀、64×64 分析、灰階比例 0.995、黑白比例 0.98 | 控制 mask 自動辨識，不會把匯出的 alpha 強制二值化 |
| Stage 4 `config.py` | `EXPECTED_FRAMES=175`、`EXPORT_ZIP=True`、`OVERWRITE_OUTPUT=True` | 控制幀數契約、ZIP 與既有資料集取代行為 |
| Stage 5 `config.py` | `BRUSH_EXE=BASE_DIR / "brush_app.exe"` | 指定真正啟動的訓練程式；不是由環境變數自動覆寫 |
| Stage 5 `config.py` | 50,000 steps、10,000,000 splat cap、2048 最大解析度、SH 3、seed 42、alpha weight 0.1 | 組成 Brush 訓練命令；未指定的 optional 參數為 `None`，不傳給 CLI |
| Stage 5 `config.py` | `DRY_RUN`、`SKIP_COMPLETED`、`CONTINUE_ON_ERROR`、`EXTRA_ARGS`、`EXTRA_ENV` | 控制僅列命令、略過、批次失敗處理、額外 flags 與 subprocess 環境 |
| 渲染器 CLI | `--colmap`、尺寸、FPS、超採樣 | 指定原始相機與輸出規格；背景目前寫在程式中，為淺灰色 |

Stage 4 強制 `REUSE_ORIGINAL_CAMERA_POSES=True`、`COMPOSE_RGBA_IMAGES=True`；`ESTIMATE_CAMERAS`、`OPTIMIZE_CAMERAS`、`MATCH_FEATURES`、`EXPORT_SEPARATE_MASKS` 必須是 `False`。違反時 `require_invariants()` 會停止執行。

`BRUSH_ALPHA_MODE="transparent"` 只用於 Stage 4 印出的建議命令。Stage 5 目前 `ALPHA_MODE=None`，實際命令沒有 `--alpha-mode`。Stage 5 的 `NO_TIMEOUT` 設定沒有被 main 讀取，訓練函式本身就未設定 timeout。

## 開發者指南（Developer Guide）

1. 先讀五個 `config.py`，再讀各自 `main()` 與 discovery 函式，確認輸入目錄深度、命名與略過條件。
2. 讀 Stage 1 的 `camera_for_frame`、`write_colmap_model` 和 Stage 4 的 COLMAP parser，理解逐幀對應規則。
3. 讀 Stage 2、3 的 API workflow 及 `prepare_workflow`；UI JSON 用於 ComfyUI 編輯，Python 實際提交 `*_api.json`。兩者不會自動同步。
4. 讀 Stage 4 的 `detect_video_pair`、`write_rgba_images`、`validate_dataset`，再讀 Stage 5 `build_command`，追蹤 alpha 到訓練的交接。
5. 修改渲染時讀 `read_colmap_cameras`、`load_gaussians`、`render`、`rasterize`，以相同原始鏡頭比較修改前後結果。

世界座標使用 `+Y` 向上、`+Z` 為角色正面；COLMAP 外參是 world-to-camera，四元數順序為 `qw,qx,qy,qz`，相機中心為 `C = -Rᵀt`。Stage 1 的第一幀與最後一幀分別對應 0°、360°，因此同為正面。

擴充時沿現有邊界新增設定、驗證與對應輸出。改變幀數必須同步 Stage 1、生成 workflow 長度與 Stage 4 契約；改變檔名必須維持 COLMAP 註冊名稱；改 Brush flags 時要同步 `REQUIRED_FLAGS` 與 steps 偵測。新增跨階段自動搬移功能時，應明確處理重跑與多版本配對。

目前沒有測試目錄或 CI。以下命令可在不啟動模型及服務的情況下檢查自有 Python 與 workflow 語法：

```powershell
python -B -c "import ast,pathlib; files=list(pathlib.Path('src').glob('*/*.py'))+list(pathlib.Path('tools').glob('*.py')); [ast.parse(p.read_text(encoding='utf-8-sig')) for p in files]; print('Python syntax OK')"
python -B -c "import json,pathlib; files=list(pathlib.Path('src').glob('*/workflows/*.json')); [json.loads(p.read_text(encoding='utf-8-sig')) for p in files]; print('Workflow JSON OK')"
```

以上只確認語法。完整驗證仍需在模型、ComfyUI、FFmpeg 與 Brush 都備妥的環境完成。

## 已知限制與待辦事項（Limitations & TODO）

### 已確認的限制

| 範圍 | 目前狀態 |
|---|---|
| 幾何與相機 | 只沿用既有姿態，不做相機估計、最佳化或稀疏特徵重建；Stage 1 的 `points3D.txt` 沒有 3D points |
| 生成一致性 | prompt 與骨架是控制條件，程式沒有驗證角色外觀、姿態與實際視角是否精確遵守 |
| 尺寸契約 | 目前訓練 RGBA 為 512×896，COLMAP 宣告為 720×1280；Stage 4 不檢查兩者尺寸一致，也不重算內參。影響與修正需另行驗證 |
| mask | 以抽樣灰階／黑白比例判別，歧義即停止；輸出 alpha 保留灰階，僅驗證非定值，未檢查每張語意遮罩是否正確 |
| 增量重跑 | Stage 2 有輸入／workflow 簽章；Stage 3 只有同名輸出判斷；Stage 5 預設重新訓練 |
| 驗證強度 | Stage 5 只抽查 PNG header；沒有專案測試套件、CI、端到端自動回歸或統一依賴 lockfile |
| 預覽 | 參考渲染器只接受 `SIMPLE_PINHOLE`／`PINHOLE` 與 SH 0–3；不能補回 PLY 缺失細節，漂浮高斯與模糊邊緣仍會顯示 |
| 樣本資料 | `test_file/suround/4/` 的影片各為 192 幀，不能直接通過目前 175 幀契約；資料夾名稱不是規格保證 |
| 部署與授權 | 最低硬體、完整套件相容矩陣、custom node 固定版本與專案根目錄授權尚未定義 |

### 可追溯的待辦範圍

自有 Python 與 workflow JSON 中未發現 `TODO`、`FIXME`、`XXX` 或 `NotImplemented` 標記。從上述缺口可明確列出的後續工程工作為：

1. 驗證影像尺寸與相機內參的交接，建立可重現的投影對照。
2. 為 175 幀、mask 配對、RGBA、原始 pose 與輸出 PLY 建立離線回歸檢查。
3. 統一各階段的重跑判斷與檔案搬移策略。
4. 固定外部模型、節點、套件及 Brush build 的相容組合。
5. 更新仍描述四階段流程的版本文件，並明確訂定專案授權。

上述為依程式現況整理的待辦，沒有在本次文件改寫中實作。

## 補充說明（Notes）

`3_remove_background` 的主要產物是 mask 影片，透明 RGBA 是到 Stage 4 才建立。`4_virtual_camera` 的名稱保留自程式目錄；它複製原始相機，不產生新的攝影機軌跡。

Stage 1 會額外輸出 `rigs.txt`、`frames.txt`。Stage 4 只複製 `cameras.txt`、`images.txt`、`points3D.txt`，且要求輸出 `sparse/0/` 的檔案集合恰為這三個檔案；不要把另外兩個檔案放回生成的 dataset。

`.gitignore` 排除 runtime input/output、模型、執行檔、訓練產物與快取；所以工作區裡現有的 ONNX、Brush EXE、PLY 不代表乾淨 checkout 也具備。文件展示的 MP4 有獨立例外規則，GIF 則未被該媒體忽略規則排除。

`docs/versions/` 與 `CHANGELOG.md` 保留歷史敘述，不可用其中舊的 Stage 編號取代現行 `src/` 路徑。專案根目錄沒有 LICENSE；vendored 程式各自附帶授權文件，不等於整個專案已採相同授權。
