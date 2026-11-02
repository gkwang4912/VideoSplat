# VideoSplat V2.0.0 Release Notes

**Status:** Prepared for release  
**Release date:** Unreleased

V2.0.0 replaces the V1 generated-video-first reconstruction architecture with a known-camera-first, four-stage pipeline.

## Major updates

- Generate a character-aligned 360-degree skeleton-control video and known COLMAP camera poses before video synthesis.
- Use a checked-in ComfyUI workflow to combine the aligned source image with skeleton motion control.
- Export separate generated color and mask videos using the workflow's foreground-removal and tracking path.
- Build a validated known-pose RGBA/mask dataset while reusing camera poses unchanged.
- Convert binary Stage 3 masks into matching image alpha channels and pass the resulting dataset directly to Stage 4.
- Train the final 3D Gaussian Splatting model through a configurable Brush CLI batch runner.

## Architecture changes

- V2 retains the V1 convention of placing main programs under `src/`, with four V2-specific stage directories instead of the V1 five-stage pipeline.
- Feature-based camera estimation is no longer part of the current pipeline.
- The V1 controlled-path renderer, second reconstruction pass, Viewer, and V1 tests are absent from V2.
- V1 source and its original root READMEs remain available through `v1.0.0`; readable snapshots are also retained under `docs/versions/`.

## Known limitations

- Local DWPose and ComfyUI model weights are required but not distributed.
- ComfyUI/custom-node/model versions are not fully locked.
- Stage 1 → 2 and Stage 2 → 3 handoffs are manual and require exact 175-frame correspondence; Stage 3 → 4 is directly connected.
- An empty `points3D.txt` causes Brush to use random initialization.
- V2 has no Viewer and only a focused RGBA handoff regression test, not a complete automated suite.
- Stage 3 outputs RGBA plus separate masks and validates that every alpha channel matches its mask; target-environment ComfyUI/Brush validation remains required.
- No project-level open-source license has been selected.

## Upgrade notes

V1 intermediate data is not a supported direct input to V2. Start from the source character image and follow [MIGRATION_V1_TO_V2.md](./MIGRATION_V1_TO_V2.md). The detailed design is documented in [V2_ARCHITECTURE.md](./V2_ARCHITECTURE.md), and usage is in the current [README](../../README.md).

## Tagging

Do not create the tag until the release candidate is manually validated with the required models, ComfyUI environment, and Brush CLI:

```powershell
git tag -a v2.0.0 -m "VideoSplat V2.0.0"
```
