"""Deterministic CPU 3D Gaussian renderer for Brush PLY previews.

Uses every Gaussian, quaternion covariance, SH0..3, perspective EWA footprints,
front-to-back alpha compositing and supersampling. Dependencies: numpy, scipy,
Pillow, plyfile, numba; ffmpeg is needed for video encoding.

Example (from repository root):
    python tools/render_gaussian_reference.py src/5_360brush/output/4/4.ply output --colmap src/5_360brush/input/4/brush_dataset/sparse/0

Mathematical conventions are documented by the original 3DGS implementation:
https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu
No external rasterizer code is required at runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import subprocess

import numpy as np
from numba import njit, prange
from PIL import Image
from plyfile import PlyData
from scipy.special import expit
from scipy.spatial.transform import Rotation


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_gaussians(path):
    ply = PlyData.read(path)
    vertices = ply['vertex'].data
    def columns(names):
        return np.column_stack([vertices[name] for name in names]).astype(np.float64)
    xyz = columns(('x', 'y', 'z'))
    scales = np.exp(columns([f'scale_{i}' for i in range(3)]))
    quaternions = columns([f'rot_{i}' for i in range(4)])
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError('Zero-length Gaussian quaternion')
    quaternions /= norms
    rotation = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()
    transform = rotation * scales[:, None, :]
    covariance = transform @ transform.transpose(0, 2, 1)
    dc = columns([f'f_dc_{i}' for i in range(3)])[:, None, :]
    # Read by NUMERIC suffix, not the lexicographic order in Brush's PLY header.
    rest_names = sorted((n for n in vertices.dtype.names if n.startswith('f_rest_')),
                        key=lambda n: int(n.rsplit('_', 1)[1]))
    rest = columns(rest_names) if rest_names else np.zeros((len(xyz), 0))
    coefficients = 1 + len(rest_names) // 3
    degree = math.isqrt(coefficients) - 1
    if len(rest_names) % 3 or (degree + 1)**2 != coefficients or degree > 3:
        raise ValueError('Expected complete SH degree 0..3 coefficients')
    sh = np.concatenate((dc, rest.reshape(len(xyz), 3, coefficients-1).transpose(0, 2, 1)), axis=1)
    opacity = expit(np.asarray(vertices['opacity'], dtype=np.float64))
    up = np.array([0., 1., 0.])
    for comment in ply.comments:
        if comment.lower().startswith('vertical axis:'):
            value = comment.split(':', 1)[1].strip().lower()
            axes = {'x': [1.,0.,0.], 'y': [0.,1.,0.], 'z': [0.,0.,1.],
                    '-x': [-1.,0.,0.], '-y': [0.,-1.,0.], '-z': [0.,0.,-1.]}
            up = np.asarray(axes[value] if value in axes else list(map(float, value.split())))
    if up.shape != (3,) or np.linalg.norm(up) < 1e-12:
        raise ValueError('Invalid vertical axis')
    up /= np.linalg.norm(up)
    for a in (xyz, covariance, sh, opacity):
        if not np.isfinite(a).all():
            raise ValueError('Non-finite Gaussian data')
    return xyz, covariance, sh, opacity, up, degree


def evaluate_sh(sh, direction):
    x, y, z = direction.T
    xx, yy, zz = x*x, y*y, z*z
    basis = np.stack((np.full_like(x, .28209479177387814),
        -.4886025119029199*y, .4886025119029199*z, -.4886025119029199*x,
        1.0925484305920792*x*y, -1.0925484305920792*y*z,
        .31539156525252005*(2*zz-xx-yy), -1.0925484305920792*x*z,
        .5462742152960396*(xx-yy),
        -.5900435899266435*y*(3*xx-yy), 2.890611442640554*x*y*z,
        -.4570457994644658*y*(4*zz-xx-yy), .3731763325901154*z*(2*zz-3*xx-3*yy),
        -.4570457994644658*x*(4*zz-xx-yy), 1.445305721320277*z*(xx-yy),
        -.5900435899266435*x*(xx-3*yy)), axis=1)
    return np.maximum(.5 + np.einsum('nk,nkc->nc', basis[:, :sh.shape[1]], sh), 0.)


@njit(parallel=True, cache=True)
def rasterize(width, height, uv, conic, bounds, opacity, colors, background):
    """Independent screen tiles preserve identical per-pixel depth ordering."""
    image = np.zeros((height, width, 3), np.float32)
    transmission = np.ones((height, width), np.float32)
    tile = 32
    nx = (width + tile - 1)//tile
    ny = (height + tile - 1)//tile
    for tile_id in prange(nx*ny):
        tx, ty = tile_id % nx, tile_id // nx
        left, top = tx*tile, ty*tile
        right, bottom = min(left+tile, width), min(top+tile, height)
        for i in range(len(opacity)):
            x0 = max(left, bounds[i, 0])
            x1 = min(right, bounds[i, 1])
            y0 = max(top, bounds[i, 2])
            y1 = min(bottom, bounds[i, 3])
            if x1 <= x0 or y1 <= y0:
                continue
            a, b, c = conic[i]
            for y in range(y0, y1):
                dy = y-uv[i, 1]
                for x in range(x0, x1):
                    t = transmission[y, x]
                    if t < 1e-6:
                        continue
                    dx = x-uv[i, 0]
                    power = -.5*(a*dx*dx+2*b*dx*dy+c*dy*dy)
                    if power > 0:
                        continue
                    alpha = min(.99, opacity[i]*math.exp(power))
                    if alpha < 1./255:
                        continue
                    weight = t*alpha
                    for channel in range(3):
                        image[y, x, channel] += weight*colors[i, channel]
                    transmission[y, x] = t*(1-alpha)
        for y in range(top, bottom):
            for x in range(left, right):
                for channel in range(3):
                    image[y, x, channel] += transmission[y, x]*background[channel]
    return image, 1-transmission


def read_colmap_cameras(folder, width, height):
    """Read world-to-camera poses in file order; never re-estimate or recenter."""
    cameras = {}
    for line in (folder/'cameras.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.split()
        camera_id, model, w, h = int(fields[0]), fields[1], int(fields[2]), int(fields[3])
        params = list(map(float, fields[4:]))
        if model == 'SIMPLE_PINHOLE' and len(params) == 3:
            fx, fy, cx, cy = params[0], params[0], params[1], params[2]
        elif model == 'PINHOLE' and len(params) == 4:
            fx, fy, cx, cy = params
        else:
            raise ValueError(f'Unsupported camera model: {model}')
        sx, sy = width/w, height/h
        if not math.isclose(sx, sy):
            raise ValueError('Output must preserve the original camera aspect ratio')
        cameras[camera_id] = dict(fx=fx*sx, fy=fy*sy, cx=cx*sx, cy=cy*sy,
                                  source_width=w, source_height=h, source_params=params)
    lines = (folder/'images.txt').read_text(encoding='utf-8').splitlines()
    poses = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith('#'):
            continue
        fields = line.split(maxsplit=9)
        if len(fields) != 10:
            raise ValueError('Invalid COLMAP image pose row')
        q = np.asarray(list(map(float, fields[1:5])))
        t = np.asarray(list(map(float, fields[5:8])))
        rotation = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        camera_id = int(fields[8])
        poses.append(dict(image_id=int(fields[0]), image_name=fields[9], camera_id=camera_id,
                          qvec=q.tolist(), tvec=t.tolist(), rotation=rotation,
                          eye=-rotation.T @ t, **cameras[camera_id]))
        # COLMAP's next row is POINTS2D, including when that row is empty.
        if index < len(lines):
            index += 1
    if not poses:
        raise ValueError('COLMAP has no registered cameras')
    return poses


def render(cloud, eye, target, width, height, focal, supersample, background, camera=None):
    xyz, covariance, sh, opacity, up, _ = cloud
    width, height, focal = width*supersample, height*supersample, focal*supersample
    if camera is None:
        forward = target-eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack((right, down, forward))
        fx, fy, cx, cy = focal, focal, (width-1)/2, (height-1)/2
    else:
        rotation = camera['rotation']
        fx, fy, cx, cy = (camera[k]*supersample for k in ('fx','fy','cx','cy'))
    camera_xyz = (xyz-eye) @ rotation.T
    ids = np.flatnonzero((camera_xyz[:, 2] > .01) & (opacity >= 1/255))
    p = camera_xyz[ids]
    z = p[:, 2]
    uv = p[:, :2]/z[:, None]*[fx,fy] + [cx,cy]
    cc = rotation[None] @ covariance[ids] @ rotation.T[None]
    jacobian = np.zeros((len(ids), 2, 3))
    jacobian[:, 0, 0] = fx/z
    jacobian[:, 1, 1] = fy/z
    jacobian[:, 0, 2] = -fx*np.clip(p[:, 0]/z, -1.3*width/(2*fx), 1.3*width/(2*fx))/z
    jacobian[:, 1, 2] = -fy*np.clip(p[:, 1]/z, -1.3*height/(2*fy), 1.3*height/(2*fy))/z
    projected = jacobian @ cc @ jacobian.transpose(0, 2, 1)
    # Match original EWA low-pass in output-pixel units, also when supersampling.
    projected[:, 0, 0] += .3*supersample**2
    projected[:, 1, 1] += .3*supersample**2
    support = np.sqrt(2*np.log(np.maximum(opacity[ids]*255, 1)))
    extent = support[:, None]*np.sqrt(np.stack((projected[:, 0, 0], projected[:, 1, 1]), axis=1))
    lo = np.maximum(np.floor(uv-extent), 0)
    hi = np.minimum(np.ceil(uv+extent)+1, [width, height])
    determinant = projected[:, 0, 0]*projected[:, 1, 1]-projected[:, 0, 1]**2
    keep = np.all(hi > lo, axis=1) & (determinant > 0)
    ids, p, uv, projected, lo, hi, determinant = [v[keep] for v in
        (ids, p, uv, projected, lo, hi, determinant)]
    order = np.argsort(p[:, 2], kind='stable')
    conic = np.stack((projected[:, 1, 1]/determinant,
                      -projected[:, 0, 1]/determinant,
                      projected[:, 0, 0]/determinant), axis=1)
    directions = xyz[ids]-eye
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    colors = evaluate_sh(sh[ids], directions)
    bounds = np.stack((lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1]), axis=1).astype(np.int32)
    image, alpha = rasterize(width, height, uv[order], conic[order], bounds[order],
                            opacity[ids][order], colors[order], background)
    return np.uint8(np.clip(image, 0, 1)*255+.5), alpha, len(ids)


def orbit_camera(cloud, width, height):
    xyz, _, _, opacity, up, _ = cloud
    # Opacity-weighted bounds keep a small group of distant opaque floaters
    # from shrinking the character. This affects framing, never cloud contents.
    def bounds(values):
        result = np.empty((2, values.shape[1]))
        for j in range(values.shape[1]):
            order = np.argsort(values[:, j])
            cumulative = np.cumsum(opacity[order])
            cumulative /= cumulative[-1]
            result[:, j] = np.interp([.01, .99], cumulative, values[order, j])
        return result
    low, high = bounds(xyz)
    target = (low+high)/2
    axis = np.array([0., 0., 1.])
    if abs(axis @ up) > .9:
        axis = np.array([0., -1., 0.])
    axis -= up*np.dot(axis, up)
    axis /= np.linalg.norm(axis)
    side = np.cross(up, axis)
    local = (xyz-target) @ np.stack((side, axis, up)).T
    ranges = bounds(local)
    radial = max(np.abs(ranges[:, :2]).max(), .01)
    vertical = max(np.abs(ranges[:, 2]).max(), .01)
    focal = height*1.45
    distance = max(vertical*focal/(height*.43), radial*focal/(width*.43))+radial
    return target, axis, side, focal, distance


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('ply', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--width', type=int, default=720)
    ap.add_argument('--height', type=int, default=1280)
    ap.add_argument('--frames', type=int, default=144)
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--supersample', type=int, default=2)
    ap.add_argument('--colmap', type=Path, required=True, help='Original sparse/0 with cameras.txt and images.txt; overrides --frames')
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()
    if min(args.width, args.height, args.frames, args.fps, args.supersample) < 1:
        ap.error('Dimensions, frame count, FPS and supersampling must be positive')
    if args.width % 2 or args.height % 2:
        ap.error('H.264 yuv420p requires even dimensions')
    args.output.mkdir(parents=True, exist_ok=True)
    cloud = load_gaussians(args.ply)
    poses = read_colmap_cameras(args.colmap, args.width, args.height) if args.colmap else None
    if poses:
        args.frames = len(poses)
        target, axis, side, focal, distance = None, None, None, 1., None
    else:
        target, axis, side, focal, distance = orbit_camera(cloud, args.width, args.height)
    source_hash = sha256_file(args.ply)
    manifest = dict(source=str(args.ply.resolve()), source_sha256=source_hash,
        gaussian_count=len(cloud[0]), sh_degree=cloud[5], up=cloud[4].tolist(),
        resolution=[args.width,args.height], supersample=args.supersample,
        frames=args.frames, fps=args.fps, target=target.tolist() if target is not None else None, distance=distance,
        focal=float(focal), background=[.96,.96,.96],
        renderer='CPU full-covariance EWA, SH0..3, depth-sorted alpha, supersampling',
        gaussian_subsampling=False, completed=False, frame_stats=[])
    if poses:
        manifest['camera_mode'] = 'original COLMAP world-to-camera, source file order; no recentering'
        manifest['colmap_source'] = str(args.colmap.resolve())
        manifest['camera_files_sha256'] = {name: hashlib.sha256((args.colmap/name).read_bytes()).hexdigest()
                                           for name in ('cameras.txt','images.txt')}
        manifest['camera_poses'] = [{k:(v.tolist() if isinstance(v,np.ndarray) else v)
                                    for k,v in pose.items()} for pose in poses]
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('frame_stats','camera_poses')}), flush=True)
    indices = [0, args.frames//4, args.frames//2, 3*args.frames//4] if args.preview else range(args.frames)
    start = time.perf_counter()
    for i in indices:
        t = time.perf_counter()
        angle = 2*math.pi*i/args.frames
        camera = poses[i] if poses else None
        eye = camera['eye'] if camera else target + distance*(axis*math.cos(angle)+side*math.sin(angle))
        image, alpha, count = render(cloud, eye, target, args.width,args.height,
                                    focal,args.supersample,np.array(manifest['background']), camera)
        size = (args.width, args.height)
        result = Image.fromarray(image).resize(size, Image.Resampling.LANCZOS)
        result.save(args.output/f'frame_{i:04d}.png')
        alpha_image = Image.fromarray(np.uint8(alpha*255+.5)).resize(size, Image.Resampling.LANCZOS)
        alpha_image.save(args.output/f'alpha_{i:04d}.png')
        stats = dict(frame=i, image_name=camera['image_name'] if camera else None, projected=count,
                     alpha_pixels=int((np.asarray(alpha_image)>25).sum()),
                     seconds=round(time.perf_counter()-t,3))
        manifest['frame_stats'].append(stats)
        print(json.dumps(stats), flush=True)
    manifest['elapsed_seconds'] = round(time.perf_counter()-start,3)
    (args.output/'render_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if not args.preview:
        subprocess.run(['ffmpeg','-v','error','-y','-framerate',str(args.fps),'-i',
            str(args.output/'frame_%04d.png'),'-frames:v',str(args.frames),
            '-c:v','libx264','-preset','slow','-crf','15','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(args.output/'gaussian_model_orbit.mp4')],check=True)
        manifest['completed'] = True
        manifest['video_sha256'] = sha256_file(args.output/'gaussian_model_orbit.mp4')
        (args.output/'render_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
