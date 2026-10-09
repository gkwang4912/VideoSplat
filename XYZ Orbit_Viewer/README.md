# XYZ Orbit Viewer

這個本機 Viewer 會在啟動時掃描 `input`，讀取 12 條相機路徑、每張 PNG、相機 JSON 與 Gaussian Splat PLY。

## 啟動

在此資料夾開啟 PowerShell：

```powershell
python app.py
```

程式會自動開啟：

```text
http://127.0.0.1:8000/
```

若 8000 已被占用：

```powershell
python app.py --port 8765
```

按 `Ctrl+C` 可停止服務。Python 後端只使用標準函式庫，不需要安裝 pip 套件。

## 兩種模式

### 純圖片模式

- 在主影像上拖曳，可在所有 1,092 個相機位置中尋找最近影格。
- 方向鍵每次改變 1 度。
- 點擊右側任一路徑後，可用底部滑桿精確切換 0-90 度。
- 從路徑另一端進入時會自動反向使用 90 張影格；拖曳與方向鍵方向不會被顛倒。
- 右側會顯示目前路徑、影格與目標方向的角度偏差。

### PLY 對照模式

- 真正渲染 `export_30000.ply` 的 3D Gaussian Splats，不會退化成一般點雲。
- 顯示 XYZ 三軸、12 條相機軌道與全部相機位置。
- 拖曳旋轉，滾輪縮放。
- 金色圓環標示最近的資料相機，右側同步顯示該相機影像。
- 沿相反方向旋轉時會自動交換路徑起終點與影格順序，不需要另外準備反向動畫。
- `全景視角` 顯示完整相機球，`模型特寫` 用於查看 PLY 細節。

## 影格數說明

每條路徑的檔名是 `000.png` 到 `090.png`。每度一張且同時保留 0 度與 90 度端點，所以實際是 91 張，不是遺漏或多讀一張。

## input 格式

```text
input/
  path_graph.json
  export_30000.ply
  anchors/
  paths/
    01_front_to_left/
      cameras.json
      images/000.png ... 090.png
    ...
    12_right_to_bottom/
```

每次重啟 `app.py` 都會重新建立資料清單，因此替換同格式的 `input` 後不需要修改前端程式。

## config.json

所有顯示參數都在專案根目錄的 `config.json`。儲存後重新整理網頁即可套用，不必重啟 Python。

- `path_playback.smart_reverse`：啟用智慧反向播放。
- `camera_point_size_px`：相機點的螢幕像素大小。
- `camera_orbit_scale`：相機軌道與 PLY 中心的距離倍率。
- `model_scale`：PLY 人物的初始大小倍率，會維持主體中心不變；也可在 PLY 對照模式用「人物大小」滑桿即時調整。
- `model_scale_min`、`model_scale_max`、`model_scale_step`：人物大小滑桿的最小值、最大值與調整間距。
- `axis_length`、`axis_offset: [x, y, z]`：三軸長度與三軸相對 PLY 的位移。
- `selected_camera_marker_size_px`：目前相機的金色圓環大小。
- `overview_*`、`focus_*`、`min_zoom_*`、`max_zoom_*`：全景、特寫與縮放距離。
- `camera_point_color`、`path_color`、`selected_camera_color`：相機點、軌跡與選取點顏色。

數值倍率以 `1.0` 代表原始尺寸；`axis_offset` 使用 PLY 世界座標單位。

## 重新建置前端

已附上建置完成的 `frontend/dist`，一般使用不需要 Node.js。修改 `frontend/src` 後才需要：

```powershell
npm install
npm run build
```

## 驗證

```powershell
python -m unittest discover -s tests -v
```

測試會確認 12 條路徑、1,092 個相機位置、所有圖片、PLY 標頭與大型檔案 Range 下載。
