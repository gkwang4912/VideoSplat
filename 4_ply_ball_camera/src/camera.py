from __future__ import annotations

import math
import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-15)


def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    a = normalize(axis); x, y, z = a; c, s, C = math.cos(angle), math.sin(angle), 1-math.cos(angle)
    return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                     [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                     [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])


def pose_from_position_up(position: np.ndarray, center: np.ndarray, image_up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = normalize(center - position)
    up = normalize(image_up - forward * np.dot(image_up, forward))
    down = -up; right = normalize(np.cross(down, forward)); down = normalize(np.cross(forward, right))
    c2w = np.eye(4); c2w[:3, :3] = np.stack([right, down, forward], axis=1); c2w[:3, 3] = position
    return c2w, np.linalg.inv(c2w)


def build_canonical(center: np.ndarray, source) -> tuple[dict, dict]:
    front = normalize(source.position - center)
    up = normalize(source.up - front * np.dot(source.up, front))
    right = normalize(np.cross(up, front))
    if np.dot(right, source.right) < 0: right = -right
    up = normalize(np.cross(front, right))
    radius = float(np.linalg.norm(source.position - center))
    axes = {"front": front, "back": -front, "right": right, "left": -right, "top": up, "bottom": -up}
    positions = {k: center + radius*v for k, v in axes.items()}
    anchors = {}
    for name in ("front", "left", "right", "back", "top", "bottom"):
        if name == "front": c2w, w2c = source.camera_to_world.copy(), source.world_to_camera.copy()
        else:
            image_up = up if name in {"left", "right", "back"} else front
            c2w, w2c = pose_from_position_up(positions[name], center, image_up)
        anchors[name] = {"position": positions[name], "c2w": c2w, "w2c": w2c}
    basis = {"front_axis": front, "back_axis": -front, "right_axis": right, "left_axis": -right,
             "up_axis": up, "down_axis": -up}
    return anchors, {"center": center, "radius": radius, **basis}


def camera_metadata(image: str, camera: dict, K: np.ndarray, width: int, height: int,
                    fov_x: float, fov_y: float, near: float, far: float, **extra) -> dict:
    c2w, w2c = camera["c2w"], camera["w2c"]
    return {
        "image": image, **extra,
        "camera_position": c2w[:3, 3].tolist(), "camera_right": c2w[:3, 0].tolist(),
        "camera_up": (-c2w[:3, 1]).tolist(), "camera_forward": c2w[:3, 2].tolist(),
        "width": width, "height": height, "fx": float(K[0,0]), "fy": float(K[1,1]),
        "cx": float(K[0,2]), "cy": float(K[1,2]),
        "fov_x_degrees": math.degrees(fov_x), "fov_y_degrees": math.degrees(fov_y),
        "intrinsic_matrix": K.tolist(), "camera_to_world": c2w.tolist(), "world_to_camera": w2c.tolist(),
        "rotation_matrix": w2c[:3,:3].tolist(), "translation": w2c[:3,3].tolist(),
        "near": near, "far": far,
        "coordinate_system": "COLMAP/OpenCV camera: +X right, +Y down, +Z forward; image origin top-left",
        "matrix_convention": "row-major JSON, column vectors, p_camera = world_to_camera @ p_world",
    }


def great_circle_path(start_name: str, end_name: str, anchors: dict, basis: dict, degrees_per_frame: float,
                      K: np.ndarray, width: int, height: int, fov_x: float, fov_y: float,
                      near: float, far: float) -> list[dict]:
    center, radius = basis["center"], basis["radius"]
    a = normalize(anchors[start_name]["position"] - center); b = normalize(anchors[end_name]["position"] - center)
    omega = math.acos(np.clip(float(np.dot(a, b)), -1, 1)); steps = int(round(math.degrees(omega)/degrees_per_frame))
    if abs(math.degrees(omega)-90.0) > 1e-7: raise ValueError(f"{start_name}->{end_name} is not 90 degrees")
    axis = normalize(np.cross(a, b))
    canonical_up = basis["up_axis"]
    directions = [rodrigues(axis, omega*(i/steps)) @ a for i in range(steps+1)]
    ups: list[np.ndarray | None] = []
    for direction in directions:
        projected = canonical_up - direction*np.dot(canonical_up, direction)
        ups.append(normalize(projected) if np.linalg.norm(projected) > 1e-10 else None)
    # At exact Top/Bottom, canonical Up has zero image projection. Use the one-sided path limit,
    # which intentionally makes pole rotation path-dependent while keeping the subject upright.
    for i, up in enumerate(ups):
        if up is None:
            neighbor = ups[1] if i == 0 else ups[-2]
            assert neighbor is not None
            tangent = neighbor - directions[i]*np.dot(neighbor, directions[i])
            ups[i] = normalize(tangent)
    frames = []
    for i in range(steps+1):
        direction = directions[i]; position = center + radius*direction; up = ups[i]
        assert up is not None
        c2w, w2c = pose_from_position_up(position, center, up); cam = {"position": position, "c2w": c2w, "w2c": w2c}
        if i == 0 and start_name not in {"top", "bottom"}: cam = anchors[start_name]
        frames.append(camera_metadata(f"{i:03d}.png", cam, K, width, height, fov_x, fov_y, near, far,
                      path=f"{start_name}_to_{end_name}", start_anchor=start_name, end_anchor=end_name,
                      local_path_angle_degrees=float(i*degrees_per_frame), sphere_radius=radius, subject_center=center.tolist(),
                      orientation_mode="projected_canonical_up_with_path_specific_poles", applied_roll_correction_degrees=0.0))
    return frames


def continuous_ring_path(ring: dict, anchors: dict, basis: dict, degrees_per_frame: float,
                         K: np.ndarray, width: int, height: int, fov_x: float, fov_y: float,
                         near: float, far: float) -> list[dict]:
    center, radius = basis["center"], basis["radius"]
    canonical_up = basis["up_axis"]
    directions: list[np.ndarray] = []
    segment_ranges: list[tuple[int, int, dict]] = []
    for segment_index, segment in enumerate(ring["segments"]):
        a = normalize(anchors[segment["from"]]["position"] - center)
        b = normalize(anchors[segment["to"]]["position"] - center)
        omega = math.acos(np.clip(float(np.dot(a, b)), -1, 1))
        steps = int(round(math.degrees(omega) / degrees_per_frame))
        if abs(math.degrees(omega) - 90.0) > 1e-7:
            raise ValueError(f"{segment['from']}->{segment['to']} is not 90 degrees")
        axis = normalize(np.cross(a, b))
        start_frame = len(directions) - (1 if segment_index else 0)
        frame_range = range(steps + 1) if segment_index == 0 else range(1, steps + 1)
        for i in frame_range:
            directions.append(rodrigues(axis, omega * (i / steps)) @ a)
        segment_ranges.append((start_frame, start_frame + steps, segment))

    def projected_canonical_up(direction: np.ndarray) -> np.ndarray | None:
        projected = canonical_up - direction * np.dot(canonical_up, direction)
        return normalize(projected) if np.linalg.norm(projected) > 1e-10 else None

    up = projected_canonical_up(directions[0])
    if up is None:
        up = projected_canonical_up(directions[1])
    if up is None:
        raise RuntimeError(f"Could not initialize continuous ring up for {ring['id']}")
    ups = [up]
    for direction in directions[1:]:
        transported = ups[-1] - direction * np.dot(ups[-1], direction)
        if np.linalg.norm(transported) <= 1e-10:
            transported = projected_canonical_up(direction)
        if transported is None:
            raise RuntimeError(f"Could not continue ring up through {ring['id']}")
        ups.append(normalize(transported))

    frame_segments = {}
    for start, end, segment in segment_ranges:
        for i in range(start, end + 1):
            frame_segments[i] = segment

    frames = []
    for i, (direction, image_up) in enumerate(zip(directions, ups)):
        position = center + radius * direction
        c2w, w2c = pose_from_position_up(position, center, image_up)
        segment = frame_segments[i]
        frames.append(camera_metadata(f"{i:04d}.png", {"position": position, "c2w": c2w, "w2c": w2c},
                      K, width, height, fov_x, fov_y, near, far,
                      ring=ring["id"], segment_path=segment["path"], path=f"{segment['from']}_to_{segment['to']}",
                      start_anchor=segment["from"], end_anchor=segment["to"],
                      ring_frame_index=i, sphere_radius=radius, subject_center=center.tolist(),
                      orientation_mode="continuous_parallel_transport_ring", applied_roll_correction_degrees=0.0))
    return frames
