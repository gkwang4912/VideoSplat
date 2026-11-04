# Gaussian reference preview renderer

`render_gaussian_reference.py` renders the Brush model using anisotropic Gaussian
covariance, SH degree 0 through 3, perspective projection, depth sorting and
front-to-back alpha compositing. CPU rendering includes all loaded Gaussians;
screen clipping and the standard per-pixel alpha threshold still apply.

## Reproduce the published preview

Run from the VideoSplat repository root:

```powershell
python -m pip install -r tools/requirements-renderer.txt
python tools/render_gaussian_reference.py src/5_360brush/output/4/4.ply src/5_360brush/output/4/source_camera_render --colmap src/5_360brush/input/4/brush_dataset/sparse/0 --width 1080 --height 1920 --fps 24 --supersample 2
```

FFmpeg must be on PATH. `--preview` renders four diagnostic views only and does
not encode a video. With `--colmap`, the frame count comes from `images.txt` and
overrides `--frames`. The 175 original world-to-camera quaternion/translation
records remain in source file order. Camera centers are computed as `-R.T @ t`.
Output resolution scales `fx`, `fy`, `cx` and `cy` equally, with no recentering,
camera smoothing or new orbit. Only `SIMPLE_PINHOLE` and `PINHOLE` cameras are
accepted; unsupported distortion models cause an error.

Outputs include RGB frames, diagnostic alpha frames, an H.264 MP4 and
`render_manifest.json`. The manifest records the PLY SHA-256, both COLMAP source
file hashes, every camera pose, rendering parameters and per-frame statistics.
The script writes to its output directory; replacing the published preview is a
separate step after checking the completed render.

The target PLY contains 60,507 Gaussians with SH degree 3 and a `Vertical axis: y`
comment. SH fields are loaded by numeric suffix, even though the file header
orders `f_rest_10` before `f_rest_2`. Log scales use `exp`; opacity logits use
sigmoid; quaternions are interpreted as `w,x,y,z` and normalized. Two-times
supersampling uses Lanczos reduction, a light-gray background and no color
grading or image sharpening.

The mathematical conventions can be checked against the original
[3DGS forward rasterization implementation](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu)
and [Gaussian parameter activation and PLY layout](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py).

The renderer cannot recover detail absent from the trained PLY. Floating
Gaussians and blurred or translucent edges in the model remain visible.
