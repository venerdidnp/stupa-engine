"""
metrics.py -- image comparison metrics for the renderer output.

All functions take two RGB images of the same shape: uint8 arrays on the
usual 0..255 scale, or float arrays expressing the same scale (the `peak`
parameter says what "white" is; it defaults to 255).

  psnr(a, b)           peak signal-to-noise ratio in dB (inf: identical)
  ssim(a, b)           simple GLOBAL SSIM on luma, one window per frame
  mean_abs_diff(a, b)  average |a - b| over all pixels and channels
  max_abs_diff(a, b)   largest |a - b| anywhere ("how wrong at worst")
  diff_image(a, b)     amplified |a - b| as a uint8 RGB image

Why global SSIM: classic SSIM slides a small window and averages local
scores. The plan asks for "a simple global SSIM", so this module uses ONE
window covering the whole frame: blind to localized artefacts, but exactly
what phases 4 and 7 need (a good render scores ~1, a structurally broken
one scores far below), with no free parameters beyond the standard
constants.

Definitions (images x, y on a 0..peak scale, N = number of samples):
  MSE  = mean((x - y)^2)
  PSNR = 10 * log10(peak^2 / MSE)                       (inf when MSE = 0)
  SSIM = (2*mu_x*mu_y + C1) * (2*cov_xy + C2)
       / ((mu_x^2 + mu_y^2 + C1) * (var_x + var_y + C2))
  C1 = (0.01*peak)^2,  C2 = (0.03*peak)^2               (standard values)
Means, variances and the covariance are population moments (divide by N,
ddof = 0). Colour is folded into luma with the Rec. 709 weights
(0.2126, 0.7152, 0.0722) -- the perceptual weights sRGB itself is built
around.
"""

import numpy as np


def _pair(a, b):
    """Validate and convert two images to matching (H, W, 3) float arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"metrics: expected (H, W, 3) RGB, got {a.shape}")
    if a.shape != b.shape:
        raise ValueError(f"metrics: shape mismatch {a.shape} vs {b.shape}")
    return a, b


def _luma(img):
    """Rec. 709 luma of an RGB float image -> (H, W)."""
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def psnr(a, b, peak=255.0):
    """Peak signal-to-noise ratio in dB (inf when the images are equal)."""
    a, b = _pair(a, b)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(peak * peak / mse)


def ssim(a, b, peak=255.0):
    """Simple global SSIM on luma, as defined in the module docstring."""
    a, b = _pair(a, b)
    x = _luma(a).ravel()
    y = _luma(b).ravel()
    c1 = (0.01 * peak) ** 2
    c2 = (0.03 * peak) ** 2
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()                  # population moments (ddof = 0)
    cov = float(np.mean((x - mx) * (y - my)))
    num = (2.0 * mx * my + c1) * (2.0 * cov + c2)
    den = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(num / den)


def mean_abs_diff(a, b):
    """Average |a - b| over every pixel and channel."""
    a, b = _pair(a, b)
    return float(np.mean(np.abs(a - b)))


def max_abs_diff(a, b):
    """Largest |a - b| over every pixel and channel."""
    a, b = _pair(a, b)
    return float(np.max(np.abs(a - b)))


def diff_image(a, b, gain=4.0):
    """Amplified absolute difference as a uint8 RGB image.

    Inputs: a, b -- same-shape RGB images; gain -- amplification factor.
    Output: (H, W, 3) uint8: black = identical, brighter = bigger error,
            clipped at white (any difference of 255/gain or more saturates).
    """
    a, b = _pair(a, b)
    return np.round(np.clip(np.abs(a - b) * gain, 0.0, 255.0)).astype(np.uint8)
