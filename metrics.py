"""Module 1 -- image similarity metrics from scratch (NumPy only).

  psnr(a, b)  : peak signal-to-noise ratio in dB (8-bit, peak 255).
  ssim(a, b)  : a SIMPLE GLOBAL structural-similarity index (single window =
                the whole image). The real SSIM (Wang et al. 2004) slides an
                11x11 Gaussian window; this global version is enough to rank
                "how different" two downscales are and is what the module
                text quotes. Compare against skimage.metrics.ssim in an
                Optional Extension Lab and explain the gap.

Run:  python metrics.py --selftest
"""
from __future__ import annotations

import sys
import numpy as np


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"psnr: shape mismatch {a.shape} vs {b.shape}")
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(peak * peak / mse))


def ssim(a: np.ndarray, b: np.ndarray, L: float = 255.0) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"ssim: shape mismatch {a.shape} vs {b.shape}")
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def diff_image(a: np.ndarray, b: np.ndarray, gain: float = 4.0) -> np.ndarray:
    d = np.abs(a.astype(float) - b.astype(float))
    return np.clip(d * gain, 0, 255).astype(np.uint8)


def _selftest() -> None:
    x = (np.arange(64, dtype=np.uint8).reshape(8, 8) * 3)
    assert psnr(x, x) == float("inf")
    assert abs(ssim(x, x) - 1.0) < 1e-9

    # a constant offset of 52 -> known PSNR
    off = np.full((8, 8), 100, np.uint8)
    off2 = np.full((8, 8), 152, np.uint8)
    expect = 10 * np.log10(255 ** 2 / 52 ** 2)
    assert abs(psnr(off, off2) - expect) < 1e-6, f"{psnr(off, off2)} vs {expect}"
    print(f"metrics self-test OK  (psnr(equal)=inf; ssim(equal)=1; "
          f"psnr for a 52-level shift = {expect:.2f} dB)")


if __name__ == "__main__":
    if "--selftest" in sys.argv or len(sys.argv) == 1:
        _selftest()
