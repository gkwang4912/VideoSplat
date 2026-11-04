"""Offline Brush/3DGS renderer: full covariance, SH3 and ordered alpha blending.

Requires numpy, scipy, plyfile, Pillow, numba and ffmpeg on PATH.
Writes frames first; never overwrites the published preview by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from numba import njit
from PIL import Image
from plyfile import PlyData
from scipy.special import expit
from scipy.spatial.transform import Rotation


def load(path):
    ply = PlyData.read(path)
    v = ply['vertex']
    col = lambda names: np.stack([v[n] for n in names], axis=1).astype(np.float64)
    xyz = col('xyz')
    scale = np.exp(col([f'scale_{i}' for i in range(3)]))
    quat = col([f'rot_{i}' for i in range(4)])
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    rot = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()
    rs = rot * scale[:, None, :]
    cov = rs @ rs.transpose(0, 2, 1)
    sh = np.concatenate((col([f'f_dc_{i}' for i in range(3)])[:, None, :],
                         col([f'f_rest_{i}' for i in range(45)]).reshape(-1, 3, 15).transpose(0, 2, 1)), axis=1)
    opacity = expit(np.asarray(v['opacity'], dtype=np.float64))
    up = np.array([0., 0., 1.])
    for comment in ply.comments:
        if comment.startswith('Vertical axis:'):
            up = np.array(list(map(float, comment.split(':', 1)[1].split())))
    up /= np.linalg.norm(up)
    if not all(np.isfinite(a).all() for a in (xyz, cov, sh, opacity)):
        raise ValueError('Non-finite Gaussian parameters')
    return xyz, cov, sh, opacity, up


def sh_colors(sh, directions):
    x, y, z = directions.T
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
    return np.maximum(.5 + np.einsum('nk,nkc->nc', basis, sh), 0)


@njit(cache=True)
def composite(width, height, uv, inverse, bounds, alpha, colors):
    image = np.zeros((height, width, 3), np.float32)
    trans = np.ones((height, width), np.float32)
    for i in range(len(alpha)):
        x0, x1, y0, y1 = bounds[i]
        a, b, c = inverse[i]
        for y in range(y0, y1):
            dy = y - uv[i, 1]
            for x in range(x0, x1):
                if trans[y, x] < 1e-5:
                    continue
                dx = x - uv[i, 0]
                power = -.5*(a*dx*dx + 2*b*dx*dy + c*dy*dy)
                if power > 0:
                    continue
                value = min(.99, alpha[i]*math.exp(power))
                if value < 1/255:
                    continue
                weight = trans[y, x]*value
                for ch in range(3):
                    image[y, x, ch] += weight*colors[i, ch]
                trans[y, x] *= 1-value
    for y in range(height):
        for x in range(width):
            for ch in range(3):
                image[y, x, ch] += trans[y, x]*.96
    return image, 1-trans


def render(cloud, eye, target, up, width, height, focal):
    xyz, cov, sh, opacity, _ = cloud
    forward = target-eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    r = np.stack((right, down, forward))
    p = (xyz-eye) @ r.T
    valid = (p[:, 2] > .01) & (opacity >= 1/255)
    ids = np.flatnonzero(valid)
    p = p[ids]
    z = p[:, 2]
    uv = p[:, :2]/z[:, None]*focal + [width/2, height/2]
    cc = r[None] @ cov[ids] @ r.T[None]
    j = np.zeros((len(ids), 2, 3))
    j[:, 0, 0] = focal/z
    j[:, 1, 1] = focal/z
    # Standard 3DGS perspective-Jacobian FOV clamp stabilizes offscreen splats.
    j[:, 0, 2] = -focal*np.clip(p[:, 0]/z, -1.3*width/(2*focal), 1.3*width/(2*focal))/z
    j[:, 1, 2] = -focal*np.clip(p[:, 1]/z, -1.3*height/(2*focal), 1.3*height/(2*focal))/z
    c = j @ cc @ j.transpose(0, 2, 1)
    c[:, 0, 0] += .3
    c[:, 1, 1] += .3
    # Opacity-dependent support includes every contribution >= 1/255.
    support = np.sqrt(2*np.log(np.maximum(opacity[ids]*255, 1)))
    extent = support[:, None]*np.sqrt(np.stack((c[:, 0, 0], c[:, 1, 1]), axis=1))
    lo = np.maximum(np.floor(uv-extent), 0)
    hi = np.minimum(np.ceil(uv+extent)+1, [width, height])
    keep = np.all(hi > lo, axis=1)
    ids, p, uv, c, lo, hi = [a[keep] for a in (ids, p, uv, c, lo, hi)]
    order = np.argsort(p[:, 2], kind='stable')
    det = c[:, 0, 0]*c[:, 1, 1]-c[:, 0, 1]**2
    inv = np.stack((c[:, 1, 1]/det, -c[:, 0, 1]/det, c[:, 0, 0]/det), axis=1)
    dirs = xyz[ids]-eye
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    colors = sh_colors(sh[ids], dirs)
    bounds = np.stack((lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1]), axis=1).astype(np.int32)
    im, alpha = composite(width, height, uv[order], inv[order], bounds[order], opacity[ids][order], colors[order])
    return np.uint8(np.clip(im, 0, 1)*255+.5), alpha, len(ids)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('ply', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--width', type=int, default=720)
    ap.add_argument('--height', type=int, default=1280)
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cloud = load(args.ply)
    xyz, cov, sh, opacity, up = cloud
    # Use opaque core for framing only: all Gaussians remain in rendering.
    core = xyz[opacity > .5]
    low, high = np.quantile(core, [.01, .99], axis=0)
    target = (low+high)/2
    axis = np.array([0., -1., 0.])
    axis -= up*np.dot(axis, up)
    axis /= np.linalg.norm(axis)
    side = np.cross(up, axis)
    projected = (core-target) @ np.stack((side, axis, up)).T
    ranges = np.quantile(projected, [.01, .99], axis=0)
    radius = max(abs(ranges[0, :2]).max(), abs(ranges[1, :2]).max())
    vertical = max(abs(ranges[:, 2]).max(), 1e-3)
    focal = args.height*1.35
    distance = max(vertical*focal/(args.height*.40), radius*focal/(args.width*.40))+radius
    meta = dict(source=str(args.ply.resolve()), count=len(xyz), sh_degree=3, up=up.tolist(),
                target=target.tolist(), distance=float(distance), resolution=[args.width,args.height],
                frames=args.frames, fps=args.fps, renderer='CPU EWA covariance SH3 ordered alpha',
                source_sha256=hashlib.file_digest(args.ply.open('rb'), 'sha256').hexdigest())
    print(json.dumps(meta), flush=True)
    indices = [0, args.frames//4, args.frames//2, 3*args.frames//4] if args.preview else range(args.frames)
    for i in indices:
        angle = 2*math.pi*i/args.frames
        eye = target+distance*(axis*math.cos(angle)+side*math.sin(angle))
        image, alpha, visible = render(cloud, eye, target, up, args.width,args.height,focal)
        Image.fromarray(image).save(args.output/f'frame_{i:04d}.png')
        Image.fromarray(np.uint8(np.clip(alpha, 0, 1)*255)).save(args.output/f'alpha_{i:04d}.png')
        print(f'{i+1}/{args.frames}: {visible} projected Gaussians, alpha pixels {(alpha>.1).sum()}', flush=True)
    (args.output/'render_manifest.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    if not args.preview:
        subprocess.run(['ffmpeg','-y','-framerate',str(args.fps),'-i',str(args.output/'frame_%04d.png'),
                        '-c:v','libx264','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',
                        str(args.output/'gaussian_model_orbit.mp4')],check=True)


if __name__ == '__main__':
    main()
