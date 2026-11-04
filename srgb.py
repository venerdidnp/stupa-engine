"""Module 1 -- sRGB <-> linear-light transfer functions.

Job: the single most-reused primitive in the module. Everything that averages,
blends, filters, or lights colour must first decode sRGB codes to linear
light, operate, then re-encode.

Reproduces:
  Example E4 (50/50 black/white blend): linear_to_srgb_g2(0.5)*255 -> ~180;
             linear_to_srgb(0.5)*255 -> ~187.6
  Example E5 (luma of an orange pixel): the g2 decode 0.784**2 -> 0.615, etc.

Run:  python srgb.py --demo
"""
from __future__ import annotations

import sys
import numpy as np


# ---- the real sRGB curve (ship this) ---------------------------------------
def srgb_to_linear(c):
    """c in [0,1] (float array). Real sRGB EOTF (electro-optical transfer)."""
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(l):
    """l in [0,1] (float array). Real sRGB OETF (opto-electronic transfer)."""
    l = np.clip(np.asarray(l, dtype=np.float64), 0.0, 1.0)
    return np.where(l <= 0.0031308, 12.92 * l, 1.055 * (l ** (1 / 2.4)) - 0.055)


# ---- the "gamma 2" teaching approximation (used in the hand examples) ------
def srgb_to_linear_g2(c):
    return np.asarray(c, dtype=np.float64) ** 2


def linear_to_srgb_g2(l):
    return np.sqrt(np.clip(np.asarray(l, dtype=np.float64), 0.0, 1.0))


# ---- byte <-> float helpers ----------------------------------------------
def to_float(u8):
    return np.asarray(u8, dtype=np.float64) / 255.0


def to_u8(f):
    return np.round(np.clip(f, 0, 1) * 255).astype(np.uint8)


def _demo() -> None:
    # round-trip identity
    c = np.linspace(0, 1, 11)
    back = linear_to_srgb(srgb_to_linear(c))
    assert np.allclose(c, back, atol=1e-9), "sRGB round-trip broken"
    print("round-trip max error :", float(np.max(np.abs(c - back))))

    # Example E4 -- the 50/50 blend
    mid_g2 = float(linear_to_srgb_g2(0.5) * 255)
    mid_real = float(linear_to_srgb(0.5) * 255)
    print(f"E4  correct mid-grey code : gamma-2 = {mid_g2:.1f}   "
          f"real sRGB = {mid_real:.1f}   naive (255+0)/2 = 127.5")
    # expected: gamma-2 ~= 180.3 ; real sRGB ~= 187.6

    # code 0.5 -> how much light?
    print("code 0.5 -> linear light  :",
          f"gamma-2 = {float(srgb_to_linear_g2(0.5)):.3f}   "
          f"real sRGB = {float(srgb_to_linear(0.5)):.3f}")
    # expected: gamma-2 = 0.250 ; real sRGB ~= 0.214


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
    else:
        print(__doc__)
