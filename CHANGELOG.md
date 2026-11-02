# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - Unreleased

### Added

- Initial public release of the five-stage VideoSplat research pipeline.
- Repository layout with the five core pipeline stages under `src/`, the standalone Viewer at the root, and experimental test programs under `tests/`.
- Single-image orbit-video and mask generation through a ComfyUI workflow.
- Frame extraction, RGBA cleanup, COLMAP camera-pose estimation, and reconstruction quality validation.
- Initial 3D Gaussian Splatting training through the Brush CLI.
- Canonical-coordinate construction, six anchor views, three complete camera rings, and twelve fixed camera paths.
- Video-model refinement and export of second-stage COLMAP, Brush, and Nerfstudio-compatible datasets using known camera poses.
- Local XYZ Orbit Viewer for inspecting Gaussian PLY models, anchors, paths, and rendered views.
- Validation scripts and targeted tests for camera conversion, path orientation, dataset indexing, and the Viewer server.

### Known Limitations

- The project has no LICENSE file; reuse terms have not yet been defined.
- End-to-end automated tests and exact tested dependency versions are not defined.
- Stages 1 and 5 require a separately configured ComfyUI installation, compatible custom nodes, and local model weights.
- Stage 3 requires a compatible Brush CLI build with the flags used by this project.
- Stage 4 is currently Windows-specific, expects Visual Studio 2022 C++ Build Tools and CUDA, and defaults `TORCH_CUDA_ARCH_LIST` to `12.0`.
- Stage 5 exports an empty COLMAP `points3D.txt` because it preserves known camera poses without inventing 3D feature points.
- `npm audit` reports one high-severity advisory group for the locked Vite 8.2.2 development dependency; upgrading to the available 8.3.5 fix requires a separate compatibility review.

[1.0.0]: https://github.com/gkwang4912/VideoSplat/releases/tag/v1.0.0
