"""Module 1 -- from-scratch PGM/PPM (Netpbm) reader and writer.

Job: read and write images with NO imaging library, so the deterministic
data set can live in Git as text-headed binary that any tool can open, and so
the "no imaging library for the from-scratch work" rule (Practicum Module
Plan) is literally true.

Supports the binary Netpbm formats:
  P5 = grayscale (1 channel),  P6 = RGB (3 channels), maxval 255.

Reproduces: nothing numeric on its own; it is infrastructure for every other
unit. make_data.py writes with it; histogram.py / resample.py / ... read with
it.

Run:  python ppm_io.py --selftest      # round-trips a tiny array
"""
from __future__ import annotations

import sys
import numpy as np


def _read_token(f) -> bytes:
    """Read the next whitespace-delimited token from a binary Netpbm header,
    skipping '#' comment lines."""
    tok = b""
    while True:
        ch = f.read(1)
        if ch == b"":
            break
        if ch == b"#":                       # comment to end of line
            while ch not in (b"\n", b""):
                ch = f.read(1)
            continue
        if ch.isspace():
            if tok:
                break
            else:
                continue
        tok += ch
    return tok


def read_ppm(path: str) -> np.ndarray:
    """Return a (H, W) uint8 array for P5 or (H, W, 3) uint8 for P6."""
    with open(path, "rb") as f:
        magic = _read_token(f)
        if magic not in (b"P5", b"P6"):
            raise ValueError(f"{path}: not a binary PGM/PPM (magic={magic!r})")
        w = int(_read_token(f))
        h = int(_read_token(f))
        maxval = int(_read_token(f))
        if maxval != 255:
            raise ValueError(f"{path}: only maxval 255 supported (got {maxval})")
        # exactly one whitespace byte was consumed after maxval by _read_token
        chan = 1 if magic == b"P5" else 3
        data = np.frombuffer(f.read(w * h * chan), dtype=np.uint8)
    if data.size != w * h * chan:
        raise ValueError(f"{path}: truncated pixel data")
    return data.reshape((h, w) if chan == 1 else (h, w, 3)).copy()


def write_ppm(path: str, img: np.ndarray) -> int:
    """Write a uint8 (H,W) as P5 or (H,W,3) as P6. Returns bytes written."""
    img = np.asarray(img)
    if img.dtype != np.uint8:
        raise ValueError("write_ppm expects a uint8 array (clip/round first)")
    if img.ndim == 2:
        magic, h, w = b"P5", img.shape[0], img.shape[1]
    elif img.ndim == 3 and img.shape[2] == 3:
        magic, h, w = b"P6", img.shape[0], img.shape[1]
    else:
        raise ValueError(f"write_ppm: bad shape {img.shape}")
    header = magic + b"\n" + f"{w} {h}\n255\n".encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(img.tobytes())
    return len(header) + img.size


def read_header(path: str) -> tuple[int, int, int]:
    """(width, height, channels) without loading the pixels -- for budget.py."""
    with open(path, "rb") as f:
        magic = _read_token(f)
        w = int(_read_token(f))
        h = int(_read_token(f))
        _ = _read_token(f)  # maxval
    return w, h, (1 if magic == b"P5" else 3)


def _selftest() -> None:
    a = (np.arange(2 * 3 * 3).reshape(2, 3, 3) * 7).astype(np.uint8)
    write_ppm("_tmp_selftest.ppm", a)
    b = read_ppm("_tmp_selftest.ppm")
    assert np.array_equal(a, b), "PPM round-trip failed"
    g = (np.arange(4 * 4, dtype=np.uint8).reshape(4, 4) * 10)
    write_ppm("_tmp_selftest.pgm", g)
    assert np.array_equal(g, read_ppm("_tmp_selftest.pgm")), "PGM round-trip failed"
    print("ppm_io self-test OK  (P5 and P6 round-trip)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
