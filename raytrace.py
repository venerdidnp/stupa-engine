"""Module 3 -- a one-frame ray tracer: pinhole rays + Moller-Trumbore, nearest
hit, the SAME texture and Lambert light as raster3d.py.

This is the "backward" (image-order) rendering direction from Module 2, in 3D:
for each pixel, shoot a ray, find the nearest triangle. It should produce the
same primary-visibility image as the rasteriser (compare.py checks).

Run:  python raytrace.py --model data/pusaka_model.obj --w 256 --h 256
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from transforms import look_at, rot_y, scale, translate
from objio import load_obj
from ppm_io import read_ppm, write_ppm
from srgb import srgb_to_linear, linear_to_srgb
from raster3d import sample_bilinear


def moller_trumbore(orig, dirs, v0, v1, v2):
    """Vectorised over `dirs` (P,3). Returns (t, u, v) with t=inf on miss."""
    eps = 1e-8
    e1 = v1 - v0
    e2 = v2 - v0
    pvec = np.cross(dirs, e2)
    det = pvec @ e1
    inv = np.where(np.abs(det) < eps, 0.0, 1.0 / np.where(det == 0, 1, det))
    tvec = orig - v0
    u = (pvec @ tvec) * inv
    qvec = np.cross(tvec, e1)
    v = (dirs @ qvec) * inv
    t = (qvec @ e2) * inv
    hit = (np.abs(det) >= eps) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > eps)
    t = np.where(hit, t, np.inf)
    return t, u, v


def render(mesh, *, w, h, tex_u8=None, model=None,
           eye=(2.6, 2.0, 3.4), center=(0, 0.7, 0), fovy=45.0,
           light_dir=(0.4, 0.9, 0.6), ambient=0.15):
    if model is None:
        model = np.eye(4)
    pos_h = np.c_[mesh.positions, np.ones(len(mesh.positions))]
    world = (model @ pos_h.T).T[:, :3]
    tris = world[mesh.faces]                        # (F,3,3)
    tuv = mesh.triangle_uvs()                       # (F,3,2)

    eye = np.asarray(eye, float)
    fwd = np.asarray(center, float) - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, (0, 1, 0)); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    aspect = w / h
    th = np.tan(np.radians(fovy) / 2)

    ii, jj = np.mgrid[0:h, 0:w]
    ndc_x = (jj + 0.5) / w * 2 - 1
    ndc_y = 1 - (ii + 0.5) / h * 2
    dirs = (ndc_x[..., None] * right * th * aspect +
            ndc_y[..., None] * up * th +
            fwd)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    D = dirs.reshape(-1, 3)

    best_t = np.full(len(D), np.inf)
    best_f = np.full(len(D), -1, np.int64)
    best_uv = np.zeros((len(D), 2))
    for fi in range(len(tris)):
        v0, v1, v2 = tris[fi]
        t, bu, bv = moller_trumbore(eye, D, v0, v1, v2)
        closer = t < best_t
        if closer.any():
            best_t[closer] = t[closer]
            best_f[closer] = fi
            bw = 1 - bu - bv
            uvv = (bw[:, None] * tuv[fi, 0] + bu[:, None] * tuv[fi, 1] + bv[:, None] * tuv[fi, 2])
            best_uv[closer] = uvv[closer]

    L = np.asarray(light_dir, float); L /= np.linalg.norm(L)
    col = np.zeros((len(D), 3))
    tex_lin = None if tex_u8 is None else srgb_to_linear(tex_u8.astype(float) / 255)
    for fi in np.unique(best_f):
        if fi < 0:
            continue
        m = best_f == fi
        v0, v1, v2 = tris[fi]
        n = np.cross(v1 - v0, v2 - v0); n /= np.linalg.norm(n) + 1e-12
        # orient toward the camera (same rule as raster3d), then one-sided Lambert
        if n @ (eye - (v0 + v1 + v2) / 3.0) < 0:
            n = -n
        shade = ambient + (1 - ambient) * max(0.0, float(n @ L))
        if tex_lin is not None:
            rgb = sample_bilinear(tex_lin, best_uv[m, 0], best_uv[m, 1])
        else:
            rgb = np.full((m.sum(), 3), 0.7)
        col[m] = np.clip(rgb * shade, 0, 1)
    img = linear_to_srgb(col).reshape(h, w, 3)
    return (np.clip(img * 255, 0, 255).astype(np.uint8), best_t.reshape(h, w))


def ray_trace(mesh, texture=None, settings=None, camera=None, m_matrix=None):
    w = settings.width if settings else 256
    h = settings.height if settings else 256
    eye = camera.eye if camera else (2.6, 2.0, 3.4)
    center = camera.target if camera else (0, 0.7, 0)
    fovy = camera.fovy_deg if camera else 45.0
    tex = texture.levels[0] if hasattr(texture, "levels") else texture
    if isinstance(tex, np.ndarray) and tex.dtype != np.uint8 and tex.max() <= 1.0:
        tex = np.round(np.clip(tex, 0, 1) * 255).astype(np.uint8)
    img, _ = render(mesh, w=w, h=h, tex_u8=tex, model=m_matrix, eye=eye, center=center, fovy=fovy)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--out", default="out/temple_rt.ppm")
    args = ap.parse_args()
    mesh = load_obj(args.model)
    tex = read_ppm("data/relief.ppm") if os.path.exists("data/relief.ppm") else None
    model = translate((0, 0, 0)) @ rot_y(35) @ scale((1.4, 1.4, 1.4))
    img, t = render(mesh, w=args.w, h=args.h, tex_u8=tex, model=model)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_ppm(args.out, img)
    hit = np.isfinite(t)
    print(f"model: {mesh.n_tris} triangles   {args.w}x{args.h}   "
          f"pixels hit: {int(hit.sum())}/{args.w*args.h}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
