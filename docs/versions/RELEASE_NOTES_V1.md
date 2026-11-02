# VideoSplat V1.0.0 Release Notes

**Status:** Prepared for release  
**Release date:** Unreleased

VideoSplat V1.0.0 establishes the first formal release of the existing research toolchain. It preserves the five-stage pipeline, bundled ComfyUI workflows, configuration files, tests, demo media, and the XYZ Orbit Viewer without changing core algorithms or data contracts.

The five numbered pipeline stages are organized under `src/`, the independent Viewer remains at the repository root, and experimental test programs are grouped under `tests/`.

## Highlights

- Generate a character orbit video and mask from a single image through ComfyUI.
- Convert generated video into cleaned RGBA frames and a validated COLMAP camera reconstruction.
- Train an initial 3D Gaussian Splatting representation with Brush.
- Render six anchor views, three full rings, and twelve fixed camera paths around the Gaussian model.
- Refine rendered path videos and export known-pose COLMAP, Brush, and Nerfstudio datasets.
- Inspect Gaussian PLY models and camera trajectories in the local XYZ Orbit Viewer.

## Documentation and release infrastructure

- Added a Keep a Changelog-style changelog and a Semantic Versioning policy.
- Added a stable V1 architecture and feature reference.
- Added matching V1.0.0 status and release navigation to the Chinese and English READMEs.
- Expanded ignore rules for local secrets, caches, logs, databases, model artifacts, and generated training data while retaining `docs/assets/demo/` media.

## Known limitations

- The repository does not contain a LICENSE. Reuse terms require an explicit maintainer decision.
- External tool, ComfyUI custom-node, and model versions are not fully pinned.
- Stage 4 currently targets Windows with Visual Studio 2022 C++ Build Tools and CUDA.
- Full end-to-end tests are not available; existing tests cover selected camera, dataset, server, and frontend behavior.
- Known-pose COLMAP exports from Stage 5 intentionally contain no reconstructed `points3D` entries.
- `npm audit` reports one high-severity advisory group affecting the locked Vite 8.2.2 development dependency. Vite 8.3.5 is listed as a fix, but upgrading remains a separate compatibility-reviewed change.

## Upgrade notes

This is the initial formal version, so there is no migration from an earlier tagged release. Existing users should retain their local model weights, ComfyUI setup, runtime inputs, and generated outputs, all of which remain outside Git.

## Tagging

Create the annotated tag only after the release-preparation changes are committed and reviewed:

```bash
git tag -a v1.0.0 -m "VideoSplat V1.0.0"
```

The tag must point to the final release commit, not the earlier documentation-only baseline at `71d3929`.
