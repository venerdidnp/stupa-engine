"""
ppm_io.py -- read and write binary P6 PPM images.

The P6 format is the simplest portable RGB format there is:
    ASCII header:  P6 <width> <height> <maxval>
    binary raster: height x width pixels, 3 bytes (R, G, B) each,
                   written row by row, top row first.

Project conventions:
- Images are uint8 arrays with shape (height, width, 3).
- Row 0 is the TOP row of the image (the usual file/display convention).
  This is exactly why the renderer flips v once when loading UVs: UVs use
  a bottom-left origin, image storage uses a top-left origin.
- We only write and accept maxval = 255 (8 bits per channel).

Only the standard library plus NumPy is used: the header is parsed by
hand and the raster is copied as raw bytes.
"""

from pathlib import Path

import numpy as np

# Bytes that terminate a header token. A comment ('#') runs to end of line.
_WHITESPACE = b" \t\r\n"


def _next_token(data, pos):
    """Return the next header token and the position just after it.

    Inputs:  data -- the whole file as bytes; pos -- cursor into it.
    Output:  (token: bytes, new_pos: int).
    Why hand-rolled: PPM allows '#' comments between any two header
    fields, so a plain split() is not enough; this skips whitespace AND
    comments before every token.
    """
    n = len(data)
    while pos < n:
        ch = data[pos]
        if ch in _WHITESPACE:
            pos += 1
        elif ch == 0x23:  # '#': skip to end of line
            while pos < n and data[pos] not in b"\r\n":
                pos += 1
        else:
            break
    start = pos
    while pos < n and data[pos] not in _WHITESPACE:
        pos += 1
    if start == pos:
        raise ValueError("read_ppm: unexpected end of header")
    return data[start:pos], pos


def read_ppm(path):
    """Load a binary P6 PPM file.

    Input:  path -- file path (str or Path).
    Output: (width, height, pixels) where pixels is a (height, width, 3)
            uint8 array with row 0 = top row.

    Why maxval must be 255: the renderer works with 8-bit channels
    everywhere; accepting 16-bit PPMs would silently corrupt values.
    """
    data = Path(path).read_bytes()

    magic, pos = _next_token(data, 0)
    if magic != b"P6":
        raise ValueError(f"read_ppm: expected P6, got {magic!r}")
    width = int(_next_token(data, pos)[0])
    pos = _next_token(data, pos)[1]
    height = int(_next_token(data, pos)[0])
    pos = _next_token(data, pos)[1]
    maxval = int(_next_token(data, pos)[0])
    pos = _next_token(data, pos)[1]

    if maxval != 255:
        raise ValueError(f"read_ppm: only maxval=255 supported, got {maxval}")
    if width <= 0 or height <= 0:
        raise ValueError(f"read_ppm: bad size {width}x{height}")

    # Exactly ONE whitespace character separates the header from the
    # raster, per the PPM spec; handle the CRLF pair as a single break.
    if pos >= len(data) or data[pos] not in _WHITESPACE:
        raise ValueError("read_ppm: missing whitespace after maxval")
    if data[pos] == 0x0D and pos + 1 < len(data) and data[pos + 1] == 0x0A:
        pos += 2
    else:
        pos += 1

    raster = data[pos:]
    expected = width * height * 3
    if len(raster) != expected:
        raise ValueError(
            f"read_ppm: expected {expected} raster bytes, found {len(raster)}"
        )
    # .copy() because frombuffer returns a read-only view over `data`.
    img = np.frombuffer(raster, dtype=np.uint8).reshape(height, width, 3).copy()
    return width, height, img


def write_ppm(path, img):
    """Write a uint8 RGB image as a binary P6 PPM file.

    Inputs: path -- file path (str or Path).
            img  -- (height, width, 3) uint8 array; row 0 is written first.
    Output: the file on disk (return value is None).

    Why ascontiguousarray: .tobytes() re-walks memory in C order, so the
    array must be contiguous or the bytes would come out scrambled.
    """
    img = np.ascontiguousarray(img)
    if img.dtype != np.uint8:
        raise ValueError(f"write_ppm: expected uint8, got {img.dtype}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"write_ppm: expected (H, W, 3), got {img.shape}")

    height, width = img.shape[0], img.shape[1]
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    Path(path).write_bytes(header + img.tobytes())
