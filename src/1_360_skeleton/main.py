#!/usr/bin/env python3
"""生成可重現的 360° 骨架動畫與逐幀 COLMAP 相機資料。"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.dont_write_bytecode = True
import config as cfg


def ensure_pose_runtime() -> bool:
    """若目前不是模型環境，改用專案內 Python 重啟；回傳是否已代為執行。"""
    if not cfg.USE_DWPOSE or not cfg.POSE_PYTHON.is_file():
        return False
    if Path(sys.executable).resolve() == cfg.POSE_PYTHON.resolve():
        return False
    subprocess.run([str(cfg.POSE_PYTHON), "-B", str(Path(__file__).resolve())], check=True)
    return True


def validate_config() -> None:
    if cfg.FRAME_COUNT < 2:
        raise ValueError("FRAME_COUNT 必須至少為 2，才能包含 0° 與 360°。")
    if cfg.WIDTH <= 0 or cfg.HEIGHT <= 0 or cfg.FPS <= 0:
        raise ValueError("WIDTH、HEIGHT、FPS 必須大於 0。")
    if cfg.CAMERA_RADIUS <= 0 or cfg.FOCAL_LENGTH_PX <= 0:
        raise ValueError("CAMERA_RADIUS 與 FOCAL_LENGTH_PX 必須大於 0。")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("找不到 ffmpeg；請先安裝並加入 PATH。")


def prepare_directories(output_dir: Path) -> tuple[Path, Path]:
    cfg.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "colmap" / "images"
    sparse_dir = output_dir / "colmap" / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    # 只清理由本程式命名的舊幀，不碰 input 或其他使用者檔案。
    for old_frame in images_dir.glob("frame*.png"):
        old_frame.unlink()
    return images_dir, sparse_dir


def add_hands(points: dict[str, tuple[float, float, float]]) -> None:
    for side, wrist_name, sign in (("r", "r_wrist", -1), ("l", "l_wrist", 1)):
        wrist = np.asarray(points[wrist_name], dtype=np.float64)
        for finger in range(5):
            if f"{side}_f{finger}_1" in points and f"{side}_f{finger}_2" in points:
                continue
            spread = (finger - 2) * 0.022
            p1 = wrist + np.array([sign * 0.055, -0.025 + spread, 0.020 + abs(spread) * 0.3])
            p2 = wrist + np.array([
                sign * (0.105 + 0.008 * (2 - abs(finger - 2))),
                -0.045 + spread * 1.6,
                0.025,
            ])
            points[f"{side}_f{finger}_1"] = tuple(p1)
            points[f"{side}_f{finger}_2"] = tuple(p2)


def camera_for_frame(frame: int) -> dict:
    """回傳攝影機中心、影像基底與 COLMAP world-to-camera 外參。"""
    angle = 2.0 * math.pi * frame / (cfg.FRAME_COUNT - 1)
    center = np.array([
        cfg.CAMERA_RADIUS * math.sin(angle),
        cfg.CAMERA_HEIGHT,
        cfg.CAMERA_RADIUS * math.cos(angle),
    ], dtype=np.float64)
    target = np.asarray(cfg.CAMERA_TARGET, dtype=np.float64)
    forward = target - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # COLMAP camera axes are +X right, +Y down, +Z forward.
    rotation = np.stack([right, -up, forward], axis=0)
    translation = -rotation @ center
    qvec = rotation_matrix_to_qvec(rotation)
    return {
        "frame": frame,
        "angle_radians": angle,
        "angle_degrees": math.degrees(angle),
        "center": center,
        "right": right,
        "up": up,
        "forward": forward,
        "rotation": rotation,
        "translation": translation,
        "qvec": qvec,
    }


def rotation_matrix_to_qvec(rotation: np.ndarray) -> np.ndarray:
    """將 3×3 world-to-camera rotation 轉成 COLMAP 的 [qw,qx,qy,qz]。"""
    m = rotation
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                          (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif i == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                          0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                          (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def project(point, camera: dict) -> tuple[int, int, float]:
    rel = np.asarray(point, dtype=np.float64) - camera["center"]
    z = float(np.dot(rel, camera["forward"]))
    x = cfg.PRINCIPAL_POINT[0] + cfg.FOCAL_LENGTH_PX * float(np.dot(rel, camera["right"])) / z
    y = cfg.PRINCIPAL_POINT[1] - cfg.FOCAL_LENGTH_PX * float(np.dot(rel, camera["up"])) / z
    return int(round(x)), int(round(y)), z


def render_frame(
    frame: int,
    camera: dict,
    source_joints: dict[str, tuple[float, float, float]] | None = None,
) -> np.ndarray:
    canvas = np.zeros((cfg.HEIGHT, cfg.WIDTH, 3), dtype=np.uint8)
    joints = dict(source_joints or cfg.JOINTS)
    add_hands(joints)
    p2 = {name: project(point, camera) for name, point in joints.items()}

    # DWPose 類型的頭部輪廓點；背面仍保留頭部，但不捏造眼睛與耳朵。
    hx, hy, hz = p2["head_center"]
    head_rx = max(18, int(round(0.105 * cfg.FOCAL_LENGTH_PX / hz)))
    head_ry = max(24, int(round(0.145 * cfg.FOCAL_LENGTH_PX / hz)))
    for k in range(20):
        angle = 2.0 * math.pi * k / 20.0
        point = (int(round(hx + head_rx * math.cos(angle))),
                 int(round(hy + head_ry * math.sin(angle))))
        rgb = (255, 80, 255) if k % 2 == 0 else (255, 255, 255)
        cv2.circle(canvas, point, 3, rgb[::-1], -1, cv2.LINE_AA)

    face_names = {"nose", "r_eye", "l_eye", "r_ear", "l_ear"}
    front_face_visible = math.cos(camera["angle_radians"]) > 0.22
    segments = []
    for a, b, rgb in cfg.LIMBS:
        if not front_face_visible and (a in face_names or b in face_names):
            continue
        segments.append(((p2[a][2] + p2[b][2]) / 2.0, a, b, rgb, 9))
    hand_colors = {"r": (255, 220, 0), "l": (80, 255, 120)}
    for side in ("r", "l"):
        for finger in range(5):
            p1, p2_name = f"{side}_f{finger}_1", f"{side}_f{finger}_2"
            segments.append(((p2[f"{side}_wrist"][2] + p2[p1][2]) / 2.0,
                             f"{side}_wrist", p1, hand_colors[side], 4))
            segments.append(((p2[p1][2] + p2[p2_name][2]) / 2.0,
                             p1, p2_name, hand_colors[side], 4))
    for _, a, b, rgb, thickness in sorted(segments, reverse=True):
        cv2.line(canvas, p2[a][:2], p2[b][:2], rgb[::-1], thickness, cv2.LINE_AA)
    for name, (x, y, _) in sorted(p2.items(), key=lambda item: item[1][2], reverse=True):
        if not front_face_visible and name in face_names:
            continue
        cv2.circle(canvas, (x, y), 3 if "_f" in name else 6, (255, 255, 255), -1, cv2.LINE_AA)
    return canvas


def write_colmap_model(sparse_dir: Path, cameras: list[dict]) -> None:
    cx, cy = cfg.PRINCIPAL_POINT
    (sparse_dir / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: 1\n"
        f"1 SIMPLE_PINHOLE {cfg.WIDTH} {cfg.HEIGHT} {cfg.FOCAL_LENGTH_PX:.17g} {cx:.17g} {cy:.17g}\n",
        encoding="utf-8",
    )

    image_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(cameras)}, mean observations per image: 0",
    ]
    frame_lines = [
        "# Frame list with one line of data per frame:",
        "#   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], NUM_DATA_IDS, DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)",
        f"# Number of frames: {len(cameras)}",
    ]
    for image_id, camera in enumerate(cameras, start=1):
        values = [*camera["qvec"], *camera["translation"]]
        pose = " ".join(f"{float(value):.17g}" for value in values)
        name = cfg.FRAME_NAME_PATTERN.format(frame=camera["frame"])
        image_lines.extend([f"{image_id} {pose} 1 {name}", ""])
        frame_lines.append(f"{image_id} 1 {pose} 1 CAMERA 1 {image_id}")

    (sparse_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (sparse_dir / "frames.txt").write_text("\n".join(frame_lines) + "\n", encoding="utf-8")
    (sparse_dir / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: 0, mean track length: 0\n",
        encoding="utf-8",
    )
    (sparse_dir / "rigs.txt").write_text(
        "# Rig calib list with one line of data per calib:\n"
        "#   RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, SENSORS[] as (SENSOR_TYPE, SENSOR_ID, HAS_POSE, [QW, QX, QY, QZ, TX, TY, TZ])\n"
        "# Number of rigs: 1\n"
        "1 1 CAMERA 1\n",
        encoding="utf-8",
    )


def write_video(images_dir: Path, output_dir: Path) -> Path:
    output_path = output_dir / cfg.VIDEO_FILENAME
    pattern = images_dir / cfg.FRAME_NAME_PATTERN.replace("{frame:03d}", "%03d")
    command = [
        "ffmpeg", "-y", "-framerate", str(cfg.FPS), "-i", str(pattern),
        "-frames:v", str(cfg.FRAME_COUNT), "-c:v", "libx264", "-preset", "slow",
        "-crf", str(cfg.VIDEO_CRF), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an", str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def write_debug(selected: dict[int, np.ndarray], cameras: list[dict], output_dir: Path) -> None:
    first = selected[cfg.DEBUG_FRAMES[0]]
    cv2.imwrite(str(output_dir / "pose_control_frame000.png"), first)
    thumb_w, thumb_h = 288, 512
    debug = np.zeros((thumb_h + 58, thumb_w * len(cfg.DEBUG_FRAMES), 3), dtype=np.uint8)
    for column, frame in enumerate(cfg.DEBUG_FRAMES):
        thumb = cv2.resize(selected[frame], (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        debug[:thumb_h, column * thumb_w:(column + 1) * thumb_w] = thumb
        label = f"Frame {frame} / {cameras[frame]['angle_degrees']:.1f} deg"
        cv2.putText(debug, label, (column * thumb_w + 18, thumb_h + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_dir / "pose_control_debug.png"), debug)


def write_camera_json(cameras: list[dict], output_dir: Path) -> None:
    serializable = []
    for camera in cameras:
        serializable.append({
            "frame": camera["frame"],
            "time_seconds": round(camera["frame"] / cfg.FPS, 9),
            "orbit_degrees": round(camera["angle_degrees"], 9),
            "camera_center_world": camera["center"].tolist(),
            "colmap_qvec_world_to_camera": camera["qvec"].tolist(),
            "colmap_tvec_world_to_camera": camera["translation"].tolist(),
            "image_name": cfg.FRAME_NAME_PATTERN.format(frame=camera["frame"]),
        })
    payload = {
        "coordinate_system": "world +Y up, +Z character front; COLMAP camera +X right, +Y down, +Z forward",
        "intrinsics": {"model": "SIMPLE_PINHOLE", "width": cfg.WIDTH, "height": cfg.HEIGHT,
                       "f": cfg.FOCAL_LENGTH_PX, "cx": cfg.PRINCIPAL_POINT[0], "cy": cfg.PRINCIPAL_POINT[1]},
        "frame_count": cfg.FRAME_COUNT,
        "fps": cfg.FPS,
        "poses": serializable,
    }
    (output_dir / "camera_poses.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_image(path: Path) -> np.ndarray:
    """以支援 Windows Unicode 路徑的方式讀圖，並統一成 BGR。"""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"無法讀取圖片：{path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        image = (image[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return image


def detect_foreground(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """偵測白色／透明背景上的主要人物，回傳遮罩與 x/y/w/h。"""
    h, w = image.shape[:2]
    border = np.concatenate([
        image[:max(1, h // 40)].reshape(-1, 3),
        image[-max(1, h // 40):].reshape(-1, 3),
        image[:, :max(1, w // 40)].reshape(-1, 3),
        image[:, -max(1, w // 40):].reshape(-1, 3),
    ])
    background = np.median(border, axis=0)
    distance = np.linalg.norm(image.astype(np.float32) - background.astype(np.float32), axis=2)
    mask = (distance >= cfg.BACKGROUND_COLOR_DISTANCE).astype(np.uint8) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise ValueError("找不到人物前景；請使用背景較單純、人物完整的圖片。")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[largest, cv2.CC_STAT_AREA])
    if area < h * w * cfg.FOREGROUND_MIN_AREA_RATIO:
        raise ValueError("偵測到的前景太小；請確認圖片中有完整人物。")
    main = (labels == largest).astype(np.uint8) * 255
    main = cv2.dilate(main, np.ones((3, 3), np.uint8), iterations=1)
    x, y, bw, bh = cv2.boundingRect(main)
    return main, (x, y, bw, bh)


def align_character(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], np.ndarray]:
    """將人物等比例置中到影片畫布，並回傳相同轉換後的遮罩。"""
    x, y, w, h = bbox
    scale = min(cfg.WIDTH * 0.90 / w, cfg.HEIGHT * 0.94 / h)
    tx = cfg.WIDTH * 0.5 - scale * (x + w * 0.5)
    ty = cfg.HEIGHT * 0.5 - scale * (y + h * 0.5)
    matrix = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)
    aligned = cv2.warpAffine(
        image, matrix, (cfg.WIDTH, cfg.HEIGHT), flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    aligned_mask = cv2.warpAffine(mask, matrix, (cfg.WIDTH, cfg.HEIGHT), flags=cv2.INTER_NEAREST)
    ax, ay, aw, ah = cv2.boundingRect(aligned_mask)
    return aligned, aligned_mask, (ax, ay, aw, ah), matrix


def row_extent(mask: np.ndarray, y: float, band: int, fallback: tuple[float, float]) -> tuple[float, float]:
    yi = int(round(y))
    y0, y1 = max(0, yi - band), min(mask.shape[0], yi + band + 1)
    xs = np.where(mask[y0:y1] > 0)[1]
    if xs.size < 2:
        return fallback
    return float(np.percentile(xs, 2)), float(np.percentile(xs, 98))


def leg_centers(mask: np.ndarray, y: float, center: float, width: float) -> tuple[float, float]:
    """在指定高度找左右腿中心；找不到分離輪廓時使用人體比例。"""
    yi = int(round(y))
    band = mask[max(0, yi - 5):min(mask.shape[0], yi + 6)] > 0
    columns = np.where(band.sum(axis=0) >= 2)[0]
    if columns.size:
        groups = np.split(columns, np.where(np.diff(columns) > 1)[0] + 1)
        groups = [group for group in groups if len(group) >= 3]
        left = [group for group in groups if float(group.mean()) < center]
        right = [group for group in groups if float(group.mean()) >= center]
        if left and right:
            return float(max(left, key=len).mean()), float(max(right, key=len).mean())
    return center - width * 0.105, center + width * 0.105


def estimate_joint_pixels(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, tuple[float, float]]:
    """從人物輪廓估計正面全身關節；適合單人、完整、直立角色圖。"""
    x, y, w, h = [float(value) for value in bbox]
    cx = x + w * 0.5
    at = lambda ratio: y + h * ratio

    shoulder_y, elbow_y, wrist_y = at(0.245), at(0.385), at(0.515)
    shoulder_l, shoulder_r = row_extent(mask, shoulder_y, 5, (cx - w * 0.20, cx + w * 0.20))
    shoulder_l = max(shoulder_l, cx - w * 0.25)
    shoulder_r = min(shoulder_r, cx + w * 0.25)
    elbow_l, elbow_r = row_extent(mask, elbow_y, 8, (cx - w * 0.34, cx + w * 0.34))

    arm_zone = mask[int(at(0.31)):int(at(0.58))] > 0
    ys, xs = np.where(arm_zone)
    if xs.size:
        wrist_l = float(np.percentile(xs, 0.5))
        wrist_r = float(np.percentile(xs, 99.5))
        left_near = ys[xs <= np.percentile(xs, 2)]
        right_near = ys[xs >= np.percentile(xs, 98)]
        if left_near.size:
            wrist_y_l = float(np.median(left_near) + at(0.31))
        else:
            wrist_y_l = wrist_y
        if right_near.size:
            wrist_y_r = float(np.median(right_near) + at(0.31))
        else:
            wrist_y_r = wrist_y
    else:
        wrist_l, wrist_r = cx - w * 0.47, cx + w * 0.47
        wrist_y_l = wrist_y_r = wrist_y

    knee_y, ankle_y = at(0.735), at(0.925)
    knee_l, knee_r = leg_centers(mask, knee_y, cx, w)
    ankle_l, ankle_r = leg_centers(mask, ankle_y, cx, w)
    hip_l, hip_r = cx - w * 0.105, cx + w * 0.105

    return {
        "pelvis": (cx, at(0.535)), "spine": (cx, at(0.365)),
        "neck": (cx, at(0.205)), "head_center": (cx, at(0.115)),
        "crown": (cx, at(0.025)), "nose": (cx, at(0.125)),
        "r_eye": (cx - w * 0.035, at(0.105)), "l_eye": (cx + w * 0.035, at(0.105)),
        "r_ear": (cx - w * 0.085, at(0.115)), "l_ear": (cx + w * 0.085, at(0.115)),
        "r_shoulder": (shoulder_l, shoulder_y), "r_elbow": (elbow_l, elbow_y),
        "r_wrist": (wrist_l, wrist_y_l), "l_shoulder": (shoulder_r, shoulder_y),
        "l_elbow": (elbow_r, elbow_y), "l_wrist": (wrist_r, wrist_y_r),
        "r_hip": (hip_l, at(0.55)), "r_knee": (knee_l, knee_y),
        "r_ankle": (ankle_l, ankle_y), "r_toe": (ankle_l, at(0.985)),
        "r_heel": (ankle_l, at(0.965)), "l_hip": (hip_r, at(0.55)),
        "l_knee": (knee_r, knee_y), "l_ankle": (ankle_r, ankle_y),
        "l_toe": (ankle_r, at(0.985)), "l_heel": (ankle_r, at(0.965)),
    }


def merge_dwpose_joints(
    detected: dict[str, tuple[float, float]],
    scores: dict[str, float],
    fallback: dict[str, tuple[float, float]],
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """以模型關節為主，缺失點才使用輪廓估計，並補出 pelvis/spine/crown/足尖。"""
    points = dict(fallback)
    confidence = {name: 0.0 for name in fallback}
    for name, point in detected.items():
        if name in points and scores.get(name, 0.0) >= cfg.POSE_MIN_CONFIDENCE:
            points[name] = point
            confidence[name] = scores[name]

    def midpoint(a: str, b: str) -> tuple[float, float]:
        return ((points[a][0] + points[b][0]) * 0.5, (points[a][1] + points[b][1]) * 0.5)

    if confidence["r_hip"] > 0 and confidence["l_hip"] > 0:
        points["pelvis"] = midpoint("r_hip", "l_hip")
        confidence["pelvis"] = min(confidence["r_hip"], confidence["l_hip"])
    if confidence["neck"] > 0 and confidence["pelvis"] > 0:
        neck, pelvis = points["neck"], points["pelvis"]
        points["spine"] = (neck[0] * 0.45 + pelvis[0] * 0.55,
                           neck[1] * 0.45 + pelvis[1] * 0.55)
        confidence["spine"] = min(confidence["neck"], confidence["pelvis"])

    face_names = ("nose", "r_eye", "l_eye", "r_ear", "l_ear")
    face = [points[name] for name in face_names if confidence[name] > 0]
    if face:
        points["head_center"] = (
            sum(point[0] for point in face) / len(face),
            sum(point[1] for point in face) / len(face),
        )
        confidence["head_center"] = min(scores.get(name, 1.0) for name in face_names if confidence[name] > 0)
        points["crown"] = (points["head_center"][0], fallback["crown"][1])
        confidence["crown"] = confidence["head_center"]

    # DWPose body 18 點沒有足尖；腳踝 x 比輪廓比例可靠，y 使用人物底部。
    for side in ("r", "l"):
        ankle = f"{side}_ankle"
        if confidence[ankle] > 0:
            for foot in (f"{side}_toe", f"{side}_heel"):
                points[foot] = (points[ankle][0], fallback[foot][1])
                confidence[foot] = confidence[ankle]

    # DWPose 的 21 點手掌：每根手指取中間關節與指尖，取代固定扇形假手。
    for side in ("r", "l"):
        for finger, base in enumerate((1, 5, 9, 13, 17)):
            middle, tip = f"{side}_hand_{base + 1}", f"{side}_hand_{base + 3}"
            if middle in detected and tip in detected:
                p1, p2 = f"{side}_f{finger}_1", f"{side}_f{finger}_2"
                points[p1], points[p2] = detected[middle], detected[tip]
                confidence[p1] = scores[middle]
                confidence[p2] = scores[tip]
    return points, confidence


def pixels_to_world(pixel_joints: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float, float]]:
    """將正面畫面關節反投影成 3D，沿用預設骨架的深度。"""
    camera = camera_for_frame(0)
    world = {}
    cx, cy = cfg.PRINCIPAL_POINT
    for name, (u, v) in pixel_joints.items():
        if name in cfg.JOINTS:
            z = float(cfg.JOINTS[name][2])
        elif name.startswith("r_f"):
            z = float(cfg.JOINTS["r_wrist"][2])
        elif name.startswith("l_f"):
            z = float(cfg.JOINTS["l_wrist"][2])
        else:
            continue
        direction = (camera["forward"] + camera["right"] * ((u - cx) / cfg.FOCAL_LENGTH_PX)
                     + camera["up"] * ((cy - v) / cfg.FOCAL_LENGTH_PX))
        distance = (z - camera["center"][2]) / direction[2]
        point = camera["center"] + direction * distance
        point[2] = z
        world[name] = tuple(float(value) for value in point)
    return world


def write_alignment_preview(
    aligned: np.ndarray,
    skeleton: np.ndarray,
    output_dir: Path,
) -> None:
    overlay = cv2.addWeighted(aligned, 0.68, skeleton, 0.85, 0.0)
    cv2.imwrite(str(output_dir / "source_aligned.png"), aligned)
    cv2.imwrite(str(output_dir / "pose_alignment_preview.png"), overlay)


def generate_character(
    output_dir: Path,
    joints: dict[str, tuple[float, float, float]],
    source: Path | None = None,
    aligned: np.ndarray | None = None,
    pose_method: str = "default",
) -> dict:
    images_dir, sparse_dir = prepare_directories(output_dir)
    cameras = []
    selected = {}
    for frame in range(cfg.FRAME_COUNT):
        camera = camera_for_frame(frame)
        image = render_frame(frame, camera, joints)
        cameras.append(camera)
        cv2.imwrite(str(images_dir / cfg.FRAME_NAME_PATTERN.format(frame=frame)), image)
        if frame in cfg.DEBUG_FRAMES:
            selected[frame] = image.copy()

    video_path = write_video(images_dir, output_dir)
    write_debug(selected, cameras, output_dir)
    write_colmap_model(sparse_dir, cameras)
    write_camera_json(cameras, output_dir)

    if aligned is not None and cfg.SAVE_ALIGNMENT_PREVIEW:
        write_alignment_preview(aligned, selected[0], output_dir)

    if not cfg.SAVE_FRAME_IMAGES:
        for frame_path in images_dir.glob("frame*.png"):
            frame_path.unlink()

    summary = {
        "name": output_dir.name,
        "source_image": str(source.relative_to(cfg.INPUT_DIR)) if source else None,
        "video": str(video_path.relative_to(cfg.ROOT)),
        "frames": cfg.FRAME_COUNT,
        "fps": cfg.FPS,
        "resolution": [cfg.WIDTH, cfg.HEIGHT],
        "colmap_sparse": str(sparse_dir.relative_to(cfg.ROOT)),
        "camera_path": "constant-radius 360-degree orbit",
        "pose": pose_method,
    }
    (output_dir / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def input_images() -> list[Path]:
    extensions = {extension.lower() for extension in cfg.INPUT_IMAGE_EXTENSIONS}
    return sorted(
        (path for path in cfg.INPUT_DIR.rglob("*") if path.is_file() and path.suffix.lower() in extensions),
        key=lambda path: str(path.relative_to(cfg.INPUT_DIR)).casefold(),
    )


def safe_output_name(path: Path, used: set[str]) -> str:
    base = re.sub(r"[^\w.-]+", "_", path.stem, flags=re.UNICODE).strip("._") or "character"
    name, number = base, 2
    while name.casefold() in used or name.casefold() == cfg.DEFAULT_OUTPUT_NAME.casefold():
        name = f"{base}_{number}"
        number += 1
    used.add(name.casefold())
    return name


def main() -> None:
    if ensure_pose_runtime():
        return
    validate_config()
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [generate_character(
        cfg.OUTPUT_DIR / cfg.DEFAULT_OUTPUT_NAME, dict(cfg.JOINTS), pose_method="default")]
    failures = []
    used = {cfg.DEFAULT_OUTPUT_NAME.casefold()}

    estimator = None
    estimator_error = None
    if cfg.USE_DWPOSE:
        try:
            from dwpose_backend import DWPoseEstimator
            estimator = DWPoseEstimator()
        except Exception as error:
            estimator_error = str(error)
            print(f"[DWPose 無法載入，改用輪廓備援] {error}", file=sys.stderr)

    for source in input_images():
        name = safe_output_name(source, used)
        try:
            image = read_image(source)
            mask, bbox = detect_foreground(image)
            aligned, aligned_mask, aligned_bbox, _ = align_character(image, mask, bbox)
            fallback = estimate_joint_pixels(aligned_mask, aligned_bbox)
            confidence = {joint: 0.0 for joint in fallback}
            pose_method = "silhouette fallback"
            if estimator is not None:
                try:
                    detected, scores = estimator.detect(aligned)
                    pixels, confidence = merge_dwpose_joints(detected, scores, fallback)
                    pose_method = "DWPose with silhouette fallback for missing joints"
                except Exception as error:
                    estimator_error = str(error)
                    pixels = fallback
                    print(f"[DWPose 偵測失敗，改用輪廓備援] {source.name}: {error}", file=sys.stderr)
            else:
                pixels = fallback
            joints = pixels_to_world(pixels)
            output_dir = cfg.OUTPUT_DIR / name
            summaries.append(generate_character(
                output_dir, joints, source, aligned, pose_method=pose_method))
            (output_dir / "pose_detection.json").write_text(
                json.dumps({
                    "method": pose_method,
                    "model_error": estimator_error,
                    "pixel_joints": {key: [round(value[0], 4), round(value[1], 4)]
                                     for key, value in pixels.items()},
                    "confidence": {key: round(value, 6) for key, value in confidence.items()},
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as error:
            failures.append({"source_image": str(source.relative_to(cfg.INPUT_DIR)), "error": str(error)})
            print(f"[略過] {source.name}: {error}", file=sys.stderr)

    batch = {
        "generated": len(summaries),
        "failed": len(failures),
        "outputs": summaries,
        "failures": failures,
    }
    (cfg.OUTPUT_DIR / "batch_summary.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(batch, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError(f"有 {len(failures)} 張輸入圖無法處理；詳見 output/batch_summary.json")


if __name__ == "__main__":
    main()

