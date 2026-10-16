"""
make_model.py -- generates the stepped temple assets, deterministically.

Running this file writes:
    data/temple.obj  -- 15 stacked axis-aligned boxes:
                        120 vertex records, 360 UV records and 90 quad
                        faces (fans triangulate to 180 triangles),
                        with per-face planar UVs.
    data/relief.ppm  -- 256x256 procedural sandstone texture.
    data/relief.png  -- the same texture as PNG, for the web viewer
                        (browsers cannot decode PPM).

The same three files are copied to web/data/ as well: the viewer is
served on its own (npx serve web, GitHub Pages), so it needs the assets
reachable from its own folder.

The asset counts are pinned by the plan: 180 triangles, 120 vertices.
15 boxes x 8 corners = 120 vertices, and 15 x 6 quad faces = 90 quads
fan-triangulated to 180 triangles. Exactly right, with no vertex
duplication: OBJ decouples per-face UV indices from positions, so the
8 shared corner positions can still carry per-face planar UVs.

The texture is wrapping value noise (fbm), so it tiles seamlessly when
the UVs run past 0..1. A fixed RNG seed makes both files byte-identical
on every machine, which keeps the renders comparable.

Run:  python make_model.py
"""

import shutil
from collections import namedtuple
from pathlib import Path

import numpy as np

import ppm_io

# Fixed RNG seed: the assets must be reproducible byte for byte.
SEED = 12345

# One full texture tile spans this many world units along a UV axis.
# 1/3 means the 256px texture covers a 3x3 world-unit patch, so the
# 6-unit-wide base shows two repeats of the pattern.
UV_SCALE = 1.0 / 3.0

# Where the generated files go (script-relative so the working
# directory does not matter).
DATA_DIR = Path(__file__).resolve().parent / "data"

# Where the viewer looks for its assets when it is served on its own
# (npx serve web, GitHub Pages); build() copies them here.
WEB_DATA_DIR = Path(__file__).resolve().parent / "web" / "data"

# A box: axis-aligned, described by its center and half-extents.
Box = namedtuple("Box", ["name", "center", "half"])

# The six faces of a box, each as (quad corner indices, u_axis, v_axis).
# Corner index bits: 4*ix + 2*iy + iz with ix/iy/iz = 0 for
# negative half-extent, 1 for positive (see _box_corners).
# Facts to know about every entry:
#   - the quad order is CCW when viewed from OUTSIDE the box;
#   - (u_axis, v_axis) is right-handed on the face: u x v = outward
#     normal, which keeps the texture unmirrored when looked at from
#     outside and matches the CCW winding.
_FACES = (
    # left   (-x): right when viewed from -x is +z
    ((1, 3, 2, 0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    # right  (+x): right when viewed from +x is -z
    ((5, 4, 6, 7), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    # front  (+z): right is +x
    ((1, 5, 7, 3), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    # back   (-z): right when viewed from -z is -x
    ((0, 2, 6, 4), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    # top    (+y): right is +x, texture "up" points to -z
    ((3, 7, 6, 2), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    # bottom (-y): right is +x, texture "up" points to +z
    ((0, 4, 5, 1), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
)


def temple_boxes():
    """Return the 15 boxes of the temple, bottom to top.

    Layout (heights in world units, y up from 0):
      - three receding terraces (base_1..3),
      - the body, topped by a slightly WIDER cornice molding,
      - a seven-step pyramidal roof (roof_1..7),
      - a small spire (spire_base, spire, tip) up to y = 10.6.

    Every box sits exactly on the one below (top of box i equals bottom
    of box i+1), so there are no gaps and no overlapping volumes. Shared
    boundary planes cannot z-fight either: the two faces there have
    opposite normals, so back-face culling always removes one of them.
    """
    return [
        # name        center                    half-extents
        Box("base_1", (0.0, 0.35, 0.0), (3.0, 0.35, 3.0)),
        Box("base_2", (0.0, 1.05, 0.0), (2.6, 0.35, 2.6)),
        Box("base_3", (0.0, 1.75, 0.0), (2.25, 0.35, 2.25)),
        Box("body", (0.0, 3.4, 0.0), (1.6, 1.3, 1.6)),
        Box("cornice", (0.0, 4.925, 0.0), (1.95, 0.225, 1.95)),
        Box("roof_1", (0.0, 5.475, 0.0), (1.65, 0.325, 1.65)),
        Box("roof_2", (0.0, 6.125, 0.0), (1.4, 0.325, 1.4)),
        Box("roof_3", (0.0, 6.775, 0.0), (1.15, 0.325, 1.15)),
        Box("roof_4", (0.0, 7.425, 0.0), (0.9, 0.325, 0.9)),
        Box("roof_5", (0.0, 8.025, 0.0), (0.675, 0.275, 0.675)),
        Box("roof_6", (0.0, 8.55, 0.0), (0.475, 0.25, 0.475)),
        Box("roof_7", (0.0, 9.025, 0.0), (0.3, 0.225, 0.3)),
        Box("spire_base", (0.0, 9.475, 0.0), (0.2, 0.225, 0.2)),
        Box("spire", (0.0, 9.975, 0.0), (0.12, 0.275, 0.12)),
        Box("tip", (0.0, 10.425, 0.0), (0.06, 0.175, 0.06)),
    ]


def _box_corners(box):
    """Return the 8 corners of a box as a (8, 3) float64 array.

    Corner index encoding: bit 2 = x side, bit 1 = y side, bit 0 = z
    side, where a set bit means the POSITIVE half-extent:
        index 0 = (-x, -y, -z) ... index 7 = (+x, +y, +z).
    This fixed encoding is what the corner tables in _FACES rely on.
    """
    cx, cy, cz = box.center
    hx, hy, hz = box.half
    return np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz + hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz - hz],
            [cx + hx, cy + hy, cz + hz],
        ],
        dtype=np.float64,
    )


def _write_obj(path, boxes):
    """Write the boxes as an OBJ file with per-face planar UVs.

    Planar UVs: each face projects its corners onto its own 2D frame
    (u_axis, v_axis from _FACES) measured in WORLD units, times
    UV_SCALE. Sharing one projection per face (instead of an arbitrary
    global plane) keeps the grain unstretched on the walls and on the
    roofs alike, and since the boxes are axis-aligned the projection is
    exact - no distortion at all, just a uniform texel/unit scale.

    The vt records are emitted per face (24 per box): position and UV
    indices are separate in OBJ, so the 8 shared corner positions can
    legally carry 4 different UVs each, one per adjacent face.
    """
    lines = [
        "# stupa-engine stepped temple",
        f"# generated by make_model.py -- {len(boxes)} boxes, "
        f"{8 * len(boxes)} vertices, {12 * len(boxes)} triangles",
        "",
    ]
    vertex_base = 0  # 0-based index of this box's first vertex
    uv_base = 0      # 0-based index of this box's first uv
    for box in boxes:
        lines.append(f"o {box.name}")
        corners = _box_corners(box)

        # 8 position records for this box.
        for corner in corners:
            lines.append(f"v {corner[0]:.6f} {corner[1]:.6f} {corner[2]:.6f}")

        # 24 uv records: 4 per face, in the face's quad corner order.
        for quad, u_axis, v_axis in _FACES:
            for corner_index in quad:
                p = corners[corner_index]
                u = float(np.dot(p, u_axis)) * UV_SCALE
                v = float(np.dot(p, v_axis)) * UV_SCALE
                lines.append(f"vt {u:.6f} {v:.6f}")

        # 6 quad faces, each referencing (vertex, uv) pairs.
        for face_index, (quad, _, _) in enumerate(_FACES):
            corner_tokens = []
            for j, corner_index in enumerate(quad):
                v_index = vertex_base + corner_index + 1  # OBJ is 1-based
                vt_index = uv_base + 4 * face_index + j + 1
                corner_tokens.append(f"{v_index}/{vt_index}")
            lines.append("f " + " ".join(corner_tokens))

        vertex_base += 8
        uv_base += 24

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _value_noise(size, fx, fy, rng):
    """One octave of value noise on a size x size grid, wrapping at edges.

    Inputs: size -- texture edge in pixels (square);
            fx, fy -- lattice cells across the texture (must divide size);
            rng -- numpy Generator (random lattice values in [0, 1)).
    Output: (size, size) float64 array in [0, 1).

    Why wrapping lattices: the noise must tile, because UVs run past
    0..1 and the sampler repeats the texture. Sampling the lattice with
    index % fx / % fy makes the left edge blend into the right edge, so
    no seam appears at tile boundaries.
    Why smootherstep (3t^2 - 2t^3) instead of linear interpolation: it
    makes the interpolated gradient zero at every lattice node, so the
    cells join without visible creases.
    """
    if size % fx or size % fy:
        raise ValueError(f"lattice {fx}x{fy} must divide texture size {size}")

    lattice = rng.random((fy, fx))

    # Sample positions in lattice units along each axis.
    xs = np.arange(size) * (fx / size)
    ys = np.arange(size) * (fy / size)
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    tx = xs - x0
    ty = ys - y0

    # Wrapped lattice indices: cell fx wraps back to cell 0, etc.
    x0w, x1w = x0 % fx, (x0 + 1) % fx
    y0w, y1w = y0 % fy, (y0 + 1) % fy

    tx = tx * tx * (3.0 - 2.0 * tx)
    ty = ty * ty * (3.0 - 2.0 * ty)

    # Bilinear blend of the four lattice corners around each sample.
    n00 = lattice[np.ix_(y0w, x0w)]
    n10 = lattice[np.ix_(y0w, x1w)]
    n01 = lattice[np.ix_(y1w, x0w)]
    n11 = lattice[np.ix_(y1w, x1w)]
    nx0 = n00 + (n10 - n00) * tx[None, :]
    nx1 = n01 + (n11 - n01) * tx[None, :]
    return nx0 + (nx1 - nx0) * ty[:, None]


def _fbm(size, fx, fy, octaves, rng):
    """Fractal noise: octaves of _value_noise with halving amplitude.

    Each octave doubles the lattice frequency (fx, fy must still divide
    `size`, which holds whenever the starting values do) and adds half
    the previous amplitude, giving the usual 1/f look. Dividing by the
    summed amplitude keeps the result in [0, 1).
    """
    total = np.zeros((size, size), dtype=np.float64)
    amplitude = 1.0
    norm = 0.0
    for _ in range(octaves):
        total += amplitude * _value_noise(size, fx, fy, rng)
        norm += amplitude
        amplitude *= 0.5
        fx *= 2
        fy *= 2
    return total / norm


def sandstone_texture(size=256, seed=SEED):
    """Generate the procedural sandstone texture.

    Inputs: size -- texture edge in pixels (default 256);
            seed -- RNG seed for reproducibility.
    Output: (size, size, 3) uint8 RGB array, row 0 = top.

    The look is built from four cheap ingredients, all tileable:
      - "mottle": large-scale 5-octave fbm -- the blotchy color variation
        of weathered stone;
      - "bands": horizontal strata via sin() along the v axis, with the
        mottle wobbling the band phase so the layers are not ruler-straight;
      - "streaks": high-frequency-across / low-frequency-along fbm --
        water streaks running down the walls;
      - "grain": per-pixel white noise -- the fine sand speckle.
    The combined variation multiplies a warm tan base color; channel
    weights desaturate the darks (green/blue react a bit less) which
    reads as dry stone rather than grey dirt.
    """
    rng = np.random.default_rng(seed)

    mottle = _fbm(size, 4, 4, 5, rng) - 0.5          # in [-0.5, 0.5)
    streaks = _fbm(size, 64, 8, 2, rng) - 0.5        # vertical weathering
    grain = rng.random((size, size)) - 0.5           # per-pixel speckle

    # Strata: 6 bands over the tile height, phase-wobbled by the mottle
    # (0.30 cycle of wobble keeps the lines from looking computer-straight).
    v01 = (np.arange(size, dtype=np.float64) + 0.5) / size
    bands = np.sin(2.0 * np.pi * (v01[:, None] * 6.0 + 0.30 * mottle))

    variation = 0.55 * mottle + 0.12 * bands + 0.20 * streaks + 0.13 * grain

    base = np.array([0.804, 0.708, 0.545])    # warm sandstone tan
    channel = np.array([1.00, 0.92, 0.80])    # keep darks warm, not grey
    rgb = base[None, None, :] * (1.0 + variation[:, :, None] * channel[None, None, :])

    return np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def build(out_dir=None, seed=SEED):
    """Generate the assets into out_dir (default: data/ next to this file).

    Output: (obj_path, ppm_path) -- Paths to the two written files.
    Deterministic: same seed -> byte-identical files, so renders and
    diffs across machines stay comparable.

    For the default location only, the same three files are copied to
    web/data/ as well (see the module docstring for why).
    """
    canonical = out_dir is None
    out_dir = Path(out_dir) if out_dir is not None else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_path = out_dir / "temple.obj"
    ppm_path = out_dir / "relief.ppm"
    png_path = out_dir / "relief.png"

    boxes = temple_boxes()
    _write_obj(obj_path, boxes)
    texture = sandstone_texture(256, seed)
    ppm_io.write_ppm(ppm_path, texture)

    # Browsers cannot decode PPM, so the web viewer needs the same texture
    # as a PNG. Pillow is imported HERE, on purpose: in this project it is
    # allowed to convert images for display, never to render or filter
    # them. The PPM stays the artifact the pipeline samples.
    from PIL import Image  # conversion only, see comment above

    Image.fromarray(texture).save(png_path)

    if canonical:
        WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
        for src in (obj_path, ppm_path, png_path):
            shutil.copyfile(src, WEB_DATA_DIR / src.name)

    return obj_path, ppm_path


if __name__ == "__main__":
    obj_path, ppm_path = build()
    print(f"wrote {obj_path}")
    print(f"  {15 * 8} vertices, {15 * 24} uvs, {15 * 12} triangles")
    print(f"wrote {ppm_path}")
    print(f"wrote {ppm_path.with_suffix('.png')} (viewer copy)")
    print(f"copied the three files to {WEB_DATA_DIR} (viewer folder)")
    print("  256x256 RGB sandstone texture")
