"""Module 3 -- compare two rendered frames (PPM). Reports size, PSNR, SSIM,
mean |diff|; optionally writes an amplified difference image.

  python compare.py A.ppm B.ppm [--diff out/diff.ppm --scale 6]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from ppm_io import read_ppm, write_ppm
from metrics import psnr, ssim


def compare_images(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.uint8)
    b = b.astype(np.uint8)
    p = psnr(a, b)
    s = ssim(a, b)
    d = np.abs(a.astype(int) - b.astype(int))
    diff_pixels = int((d > 0).any(axis=2).sum()) if d.ndim == 3 else int((d > 0).sum())
    return {
        "psnr": p,
        "psnr_db": p,
        "ssim": s,
        "mean_diff": float(d.mean()),
        "max_diff": int(d.max()),
        "different_pixels": diff_pixels,
        "differing_pixels": diff_pixels,
    }


def compare(path_a: str, path_b: str, diff_path: str | None = None, scale: float = 6.0):
    A = read_ppm(path_a)
    B = read_ppm(path_b)
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch {A.shape} vs {B.shape}")
    H, W = A.shape[:2]
    p = psnr(A, B)
    s = ssim(A, B)
    d = np.abs(A.astype(int) - B.astype(int))
    md = float(d.mean())
    mxd = int(d.max())
    print(f"same size {W}x{H}   PSNR: {'inf' if np.isinf(p) else f'{p:.1f}'} dB   SSIM: {s:.3f}   "
          f"mean |diff|: {md:.1f}   max |diff|: {mxd}")
    if diff_path:
        diff_u8 = np.clip(d * scale, 0, 255).astype(np.uint8)
        write_ppm(diff_path, diff_u8)
        print(f"wrote amplified diff ({scale}x) to {diff_path}")
    return p, s, md, mxd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--diff")
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--min-psnr", type=float, default=None)
    args = ap.parse_args()

    scale = 6.0
    if args.scale is not None:
        scale = args.scale
    elif args.gain is not None:
        scale = args.gain

    A = read_ppm(args.a).astype(np.int32)
    B = read_ppm(args.b).astype(np.int32)
    if A.shape != B.shape:
        print(f"size mismatch: {A.shape} vs {B.shape} -- render both at the "
              f"same w/h and setPixelRatio(1)")
        sys.exit(2)
    d = np.abs(A - B)
    p = psnr(A.astype(np.uint8), B.astype(np.uint8))
    print(f"same size {A.shape[1]}x{A.shape[0]}   "
          f"PSNR: {'inf' if np.isinf(p) else f'{p:.1f}'} dB   "
          f"SSIM: {ssim(A.astype(np.uint8), B.astype(np.uint8)):.3f}   "
          f"mean |diff|: {d.mean():.1f}   max |diff|: {int(d.max())}")
    if args.diff:
        vis = np.clip(d * scale, 0, 255).astype(np.uint8)
        write_ppm(args.diff, vis)
        print(f"wrote {args.diff} (|A-B| x{scale:g})")

    if args.min_psnr is not None:
        if p < args.min_psnr:
            sys.exit(1)


if __name__ == "__main__":
    main()
