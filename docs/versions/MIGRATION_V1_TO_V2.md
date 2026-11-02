# Migrating from VideoSplat V1 to V2

V2 is a breaking architecture change. Do not copy V1 runtime folders into V2 and expect them to run. Use the `v1.0.0` tag for the complete V1 source and [README_V1.md](./README_V1.md) for its preserved documentation.

## Folder structure

| V1 | V2 | Change |
|---|---|---|
| `src/1_suround_video` | `src/1_360_skeleton` | V2 starts by generating skeleton control and known cameras, not a character orbit video. |
| `src/2_colmap` | No equivalent | V2 does not estimate camera poses with COLMAP. |
| `src/3_360brush` | `src/4_360brush` | Brush training remains; V2 Stage 3 now produces the RGBA input consumed by Stage 4. |
| `src/4_ply_ball_camera` | No equivalent | V2 does not render V1's three rings and twelve paths. |
| `src/5_XYZ Orbit_VideoModel` | `src/2_suround_video` plus `src/3_virtual_camera` | V2 performs one controlled generation pass, then builds a known-pose dataset. |
| `XYZ Orbit_Viewer` | No equivalent | No Viewer is present in V2. |
| `tests/` | `tests/test_rgba_handoff.py` | V1 tests are removed; V2 adds a focused RGBA handoff regression test. |

## Operating sequence

V1 generated an orbit video, estimated its cameras, trained an initial 3DGS, rendered controlled paths, refined those paths, and prepared a second reconstruction. V2 defines a fixed camera orbit first, uses a skeleton video to control generation, preserves the known cameras, and performs one Brush reconstruction.

## Installation changes

- V2 requires the DWPose runtime and two local ONNX model files.
- V2 requires the ComfyUI nodes and model files referenced by `VideoSplat_2.json`.
- `VIDEOSPLAT_BRUSH_EXE` replaces machine-specific Brush paths.
- V1 Viewer Node.js dependencies are no longer required.
- V1 Stage 4 gsplat/CUDA rendering dependencies are no longer part of the V2 source.

## Configuration changes

- Character/camera settings are in `src/1_360_skeleton/config.py`.
- Explicit ComfyUI jobs are listed in `src/2_suround_video/config.py`.
- Frame, mask, RGBA, ZIP, and known-pose invariants are in `src/3_virtual_camera/config.py`.
- Brush training options are in `src/4_360brush/config.py`.

## Input and output changes

- Stage 1 accepts full-body images and produces aligned images, skeleton-control videos, camera JSON, and COLMAP text models.
- Stage 2 accepts an explicitly paired aligned image and skeleton video and returns two generated videos.
- Stage 3 requires the generated color and mask videos plus the Stage 1 COLMAP text model. It outputs RGBA `images/`, separate `masks/`, and `sparse/0/` directories.
- Stage 4 consumes `src/3_virtual_camera/output/<group>/brush_dataset/` directly and exports PLY files.

## Removed V1 behavior

- Feature-based COLMAP reconstruction and registration-quality thresholds.
- Initial Gaussian path rendering with six anchors, three rings, and twelve paths.
- Second-stage path refinement and reconstruction datasets.
- XYZ Orbit Viewer.
- V1-specific tests and experimental programs on the current branch.

## New V2 behavior

- DWPose/silhouette character pose alignment.
- Deterministic 175-frame skeleton orbit and known camera export.
- Skeleton-controlled ComfyUI generation with workflow-produced masks.
- Strict RGBA/mask/pose validation and ZIP export.
- Stage 3 binary-mask-to-alpha conversion and direct Stage 3 → 4 dataset handoff.

## Upgrade checklist

1. Keep V1 projects with the `v1.0.0` checkout; do not overwrite their runtime folders.
2. Install V2 dependencies and local DWPose models.
3. Configure the V2 ComfyUI workflow and explicit `INPUT_JOBS`.
4. Re-run the V2 pipeline from Stage 1; V1 intermediate outputs are not a supported V2 input contract.
5. Verify all 175-frame handoffs before starting Brush training.

Stage 3 now owns RGBA conversion, and Stage 4 reads its output directory by default. Verify the 175 RGBA alpha channels and complete one target-environment Brush run before publishing V2.
