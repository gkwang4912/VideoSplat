# First-Frame Canonical Gaussian Camera Dataset

本專案固定 Gaussian、不移動或訓練它，以原始 COLMAP 影像序列的第一個 registered frame 作為唯一 Front，建立六個固定 Camera Anchors 與十二條 90° great-circle camera paths。每張 Render 都有完整且可重現的 Camera Metadata。

## 執行

需要 Python、PyTorch CUDA、`gsplat`、`plyfile`、Pillow、ffmpeg/imageio-ffmpeg：

```bash
python -m pip install -r requirements.txt
python run.py
```

`run.py` 支援兩種 input 版型：

- 原始單組：`input/` 最上層放一個 `.ply` 與對應 COLMAP/images，輸出仍在 `output/`。
- 多組批次：`input/1/`、`input/2/` ... 每個子資料夾各放一個 `.ply` 與對應 COLMAP/images，程式會依資料夾自然排序逐組處理，輸出到 `output/1/`、`output/2/` ...

加速相關設定：

- `video_jobs`：平行產生 path 預覽影片的 ffmpeg 工作數，預設 `4`。
- `skip_completed_datasets`：已成功完成、設定相同、輸出版本相同的資料集會直接跳過，預設 `true`。要強制重跑可刪除該組 `output/<編號>/`，或暫時改成 `false`。

`config.json` 的 `mask_enabled` 預設為 `true`。Background-key mask pass 的顏色、距離 threshold 與後處理參數可在執行前調整。

目前機器的 base Python 缺少已編譯 gsplat CUDA extension，因此 `run.py` 會自動使用已存在且相容的 `nerfstudio` conda environment；命令仍是 `python run.py`。

每次執行會重建 `output/`，不修改 `input/` 或 Gaussian。沒有 COLMAP、資料模糊、超過一個 top-level PLY 或不支援的 distorted camera model時會停止並清楚報錯，不會猜測。

## 實際 Input 偵測

- Gaussian：`input/export_30000.ply`，binary little-endian，142,277 Gaussians，SH degree 3。
- Camera：COLMAP text `cameras.txt`、`images.txt`、`points3D.txt`；發現兩份 byte-identical sparse model，選擇具有直接匹配 `images/` 的模型。
- 原始 registered frames：189 張 RGBA，704×1344，自然排序 `0001.png` 至 `0189.png`。
- 第一幀：`0001.png`；Camera model 是 `SIMPLE_PINHOLE`，`fx=fy=5671.9785351142145`、`cx=352`、`cy=672`。
- `input/colmap/images/` 另有 189 張 RGB frames，`input/colmap/masks/` 有 189 張 masks；canonical camera 以 COLMAP registered RGBA frame 和 pose 為準。

## Canonical 座標

Robust subject center 使用 Gaussian position 各軸 1%–99% bounds 的中點。Front radial axis 為：

```text
F = normalize(original_first_camera_center - subject_center)
```

Up 使用第一幀 camera image-up 投影到 F 的切平面後正規化；Right 從第一幀 camera image-right 與右手 handedness 建立，再精確正交化。`canonical_coordinate_system.json` 保存 F/Back/Right/Left/Up/Down、中心、半徑、dot products 和 determinant。

Front Anchor 完整保留原始 COLMAP C2W/W2C，沒有重新 look-at。第一幀 forward 與 robust center 的實際誤差是 0.101106643°，因此 Front 仍以原始 pose 為最高優先；其他 anchors 朝向 center。

唯一球面半徑為第一幀 camera center 到 subject center 的距離：`3.7453197098534687`。不依 bounds、FOV、方向或 frame 自動 zoom。

## 相機與矩陣 Convention

- 世界為原始 COLMAP/Gaussian 右手座標；Gaussian 不旋轉。
- Camera 採 COLMAP/OpenCV：local `+X` 向右、`+Y` 向下、`+Z` 向前；影像原點左上。
- JSON 矩陣為 row-major 二維陣列，作用於 column vector：`p_camera = world_to_camera @ p_world`。
- COLMAP quaternion 為 `wxyz`；Gaussian `rot_0..3` 也為 `wxyz`。

## Anchors 與 Paths

`output/anchors/` 包含 Front、Left、Right、Back、Top、Bottom PNG，以及逐張 JSON 和 `cameras.json`。Front position 是原始第一幀 center；其他位置分別為 `center ± radius × canonical_axis`。

`output/paths/` 包含十二條 edge：Front→Left/Right、Left/Right→Back、Front→Top/Bottom、Top/Bottom→Back、Top→Left/Right、Left/Right→Bottom。

每條 path：

- 90° great-circle arc，不使用 XYZ lerp。
- `000.png` 至 `090.png`，每 1° 一張，共 91 張並保留兩端 Anchor。
- Position 嚴格位於固定半徑球面。
- Camera image-up 使用人物 Canonical Up 在當前視平面的投影，因此人物上下軸保持畫面垂直，不再沿路徑傾斜。
- Top/Bottom 精確極點的 Canonical Up 投影為零，因此使用該路徑的單側連續極限 rotation；極點 position、radius、intrinsics 仍共享，但 rotation 可依路徑不同。
- `preview.mp4` 逐 RGB/RGBA PNG 編碼；`preview_mask.mp4` 顯示 binary empty mask；`preview_side_by_side.mp4` 左側為 render、右側為同尺寸 mask。三者都沒有 crop、補幀、zoom 或 stabilization。
- `output/rings/` 先產生三支完整閉合環影片：Front→Left→Back→Right→Front、Front→Top→Back→Bottom→Front、Top→Left→Bottom→Right→Top。
- Ring camera 使用連續 roll transport，經過 Top/Bottom 極點時不重新選擇視角方向，避免垂直環與斜向環在極點產生方向跳變。
- 每個 ring 底下的 `segments/` 再由 ring frames 切出四段 90° segment 影片；不再先產生 12 支 path mp4 再合併。
- `output/combine.mp4` 只串接三支完整 ring `preview.mp4`，因此每個環會完整跑完四段才切到下一個環。

## PLY Splat 設定

`config.json` 的 `ply_splat` 會在 PLY 載入後、任何 render 前套用一次。調整後的同一份 Gaussian 會供 anchors、所有 path RGB、mask render pass 與後續影片共同使用：

- `splat_scale`：所有 Gaussian 的全域尺寸倍率；`1.0` 保持原始尺寸，`0` 自動切換為 point mode，只顯示 Gaussian 中心點。
- `splat_scale_xyz`：依 PLY local scale 三軸分別調整，再乘上 `splat_scale`。
- `opacity_multiplier`：opacity 倍率；結果限制在 `0..1`。
- `opacity_power`：先對 opacity 做次方，可調整半透明 splat 的分布；`1.0` 不變。
- `sh_degree`：`null` 使用 PLY 原始 SH degree，也可填入不大於原始 degree 的整數。
- `eps2d`：gsplat 投影的最小 2D covariance，預設 `0.3`。
- `radius_clip`：忽略投影半徑小於此像素值的 splat，`0.0` 表示不裁除。
- `point_size_pixels`：僅在 `splat_scale = 0` 時使用，控制 point mode 的近似螢幕點半徑。Point mode 會自動改用 classic rasterization，避免零尺寸 Gaussian 被 antialiased compensation 消除。

Point mode 的影像本來就是稀疏點，因此不套用只適合完整 splat silhouette 的 mask over-expansion 判定；mask 檔案格式、尺寸、binary values 與三種影片仍照常驗證。

實際套用值會寫入 `output/splat_configuration.json` 與 `camera_manifest.json`。

## Background-key Mask Render Pass

每個 Camera 會執行兩次相同 geometry/pose/intrinsics 的 gsplat rasterization。第一次是完全不變的正常 RGB；第二次只為 mask，將所有 Gaussian 設為白色 `[255,255,255]`、背景設為洋紅 `[255,0,255]`。Mask 不讀正常 RGB，也不做背景色精確相等：

```text
distance = length(mask_pass_rgb_0_255 - [255,0,255])
raw_mask = distance < 25
final_mask = close_3x3(raw_mask) + remove_components_smaller_than_24px

255 = repair / empty / unknown
0   = keep / occupied / known
```

每條路徑保留既有 `images/` 和 `cameras/`，新增 `mask/000.png ... 090.png`。Mask 是與原圖同解析度的單通道 8-bit `L` PNG，且只含 0/255。Anchors 也輸出至 `output/anchors/mask/`。

## Metadata

每張 camera JSON 含 path、start/end anchor、local angle、sphere radius、subject center、position、forward/up/right、resolution、fx/fy/cx/cy、FOV、K、C2W、W2C、rotation、translation、near/far、convention，以及 mask-pass 前景/背景色、距離 threshold、後處理參數、逐幀與前一版 alpha mask 的差異統計和相對路徑。`camera_manifest.json` 集中保存相同的可追溯資訊。

`path_graph.json` 將六個 anchors 表示為 nodes，十二條 90° arcs 表示為 edges，供後續 Video/Diffusion/Gaussian optimization 程式直接讀取。

## 自動驗證

執行完成會檢查全部 1092 個 path cameras：

- Radius 最大誤差：`1.3322676295501878e-15`。
- Path angle 最大誤差：`8.537736462515939e-07°`。
- 共享 Anchor position 最大誤差：`4.44089209850063e-16`；intrinsics 誤差：`0`。
- Front 對原始第一幀 position/rotation 誤差：`0`；由保存 JSON 重 Render pixel 誤差：`0`。
- Canonical Up 畫面垂直對齊最大誤差：`0.100707229°`（來自必須原樣保留的第一幀 pose；其他 frame 約為 0）。
- 最大相鄰 Camera rotation：`1.1007438434°`；沒有 camera flip，也沒有額外 endpoint roll correction。
- 每張 render 都有獨立 background-key mask pass；逐張檢查解析度、`L` mode 與僅含 0/255。
- 每條 path 都有三支影片；side-by-side 為 1408×1344，正好是原始 704×1344 的兩倍寬度。
- `validation.json` 記錄每條 path 相對上一版 alpha repair-aware mask 新增/移除的 pixels、very-strong coverage 誤遮比例與過度擴張判定。
- `output/preview/mask_before_after_foot_comparison.png` 比較 Front alpha mask 與 background-key mask；右側紅色是新加入的 repair pixels，紅框標示腿腳區。

可再次檢查現有輸出：

```bash
python scripts/validate_project.py
```

完整數字位於 `output/validation.json`；快速檢查圖位於 `output/preview/anchors_contact_sheet.jpg` 和 `path_graph_contact_sheet.jpg`。

後續 refined image 必須繼續使用同名 JSON 的固定 intrinsics 與完整 W2C/C2W，便可綁回唯一 Camera Pose 重新優化 Gaussian。
