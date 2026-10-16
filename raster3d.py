"""
raster3d.py -- the software rasterizer (build plan phase 3, in nine steps).

Pipeline, per triangle:
    object -> world -> eye (view) -> clip (projection) -> near clip in clip
    space -> w-divide to NDC -> back-face cull -> viewport (y flipped to
    screen pixels) -> barycentric fill with z-buffer -> perspective-correct
    UV -> shade -> framebuffer (linear light), encoded to sRGB once at the end.

The nine stages (see stage_settings):
    1 wireframe       project vertices, draw edges; no depth, no cull
    2 flat fill       barycentric fill inside a pixel bounding box
    3 z-buffer        depth test; NDC z interpolated linearly (see below)
    4 near clip       Sutherland-Hodgman against z + w >= 0 in clip space
    5 back-face cull  CCW in NDC (y up) is front; skip the rest
    6 Lambert         ambient + diffuse, one directional light
    7 texture         perspective-correct UV (u/w, v/w, 1/w, divide per pixel)
    8 bilinear        four-texel blend with a wrapping sampler
    9 mipmaps         4-level gamma-correct pyramid, level from texel/pixel ratio

Why NDC z interpolates linearly in screen space (no /w divide needed for
depth): for a planar triangle 1/z_eye is affine in screen coordinates, and
NDC z = (A*z_eye + B)/(-z_eye) = -A - B*(1/z_eye) is an affine function of
1/z_eye, so it is affine in screen space too. Barycentric interpolation is
therefore exact. UVs are NOT affine under perspective, which is why stage 7
interpolates u/w, v/w and 1/w instead and divides per pixel.

Why all shading happens in linear light: sRGB values are encoded for display,
not for arithmetic; multiplying or blending them directly darkens midtones.
The texture is decoded to linear once (Texture), every product (albedo *
light) happens in linear space, and the framebuffer is encoded back to sRGB
exactly once when the image is written.

Texture orientation contract (pinned with objio and ppm_io):
    - objio flips vt once: v_loaded = 1 - v_file.
    - ppm_io keeps row 0 = top of the image.
    - The sampler maps (u, v) -> continuous texel coordinates
          x = u * width  - 0.5        (column, wraps modulo width)
          y = (1 - v) * height - 0.5  (image row from the top, wraps modulo height)
      so v = 0 samples the bottom image row and v = 1 the top one. At the
      exact wrap seam, nearest picks the top row and bilinear blends the two
      seam rows 50/50. The Three.js viewer must use flipY = false to match
      this pairing exactly.

Outputs (written by main()):
    out/temple_raster.ppm   color, final stage settings
    out/temple_depth.ppm    depth, (z_ndc + 1) / 2, background = white
    out/step{1..9}_*.ppm    one frame per stage with --step / --steps
    matching .png previews for the color images (Pillow converts only)

Run:
    python raster3d.py            renders the final frame
    python raster3d.py --step 3   renders one stage (1..9)
    python raster3d.py --steps    renders every stage, then the final frame
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import objio
import ppm_io
from transforms import look_at, perspective, project, viewport

# Where the generated assets live and where renders go (script-relative,
# so the working directory does not matter).
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "out"

# Pinned scene colors (sRGB, 0..1). The background is the dawn sky; "sand"
# is the flat fallback albedo before texturing exists; "ink" draws the
# stage-1 wireframe.
SKY_SRGB = (0.87, 0.90, 0.95)
SAND_SRGB = (0.80, 0.68, 0.52)
INK_SRGB = (0.28, 0.20, 0.14)

# Human-readable stage names, used in the output file names.
STAGE_NAMES = {
    1: "wireframe",
    2: "flat",
    3: "zbuffer",
    4: "nearclip",
    5: "cull",
    6: "lambert",
    7: "texture",
    8: "bilinear",
    9: "mipmap",
}


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def srgb_to_linear(c):
    """Convert sRGB values to linear light (exact sRGB piecewise curve).

    Input:  c -- scalar or array in [0, 1], sRGB encoded.
    Output: same shape, linear light in [0, 1].

    The curve is linear near black (slope 1/12.92) and a power law above
    0.04045. Using the exact curve (not a plain 2.2 gamma) keeps the
    round trip through linear space bit-stable for our 8-bit images.
    """
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    """Inverse of srgb_to_linear (encode linear light for display).

    Input:  c -- scalar or array in [0, 1], linear light.
    Output: same shape, sRGB encoded in [0, 1].
    """
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


def barycentric(xs, ys, tri):
    """Weights of one or many 2D points relative to a triangle.

    Inputs:
      xs, ys -- (P,) query coordinates (one batch of points).
      tri    -- (3, 2) triangle vertices, any winding, non-degenerate.
    Output: (P, 3) weights (w0, w1, w2), summing to 1 at every point.

    How: each weight is an edge-function cross product divided by the
    triangle's signed doubled area. At a vertex the matching weight is 1;
    inside the triangle all three weights are in [0, 1]. Dividing by the
    signed area (not its absolute value) makes the sign cancel, so the
    "all weights >= 0" inside test works for either winding, and the same
    numbers double as interpolation weights for linear attributes.
    A degenerate triangle (area exactly 0) is the caller's responsibility
    to avoid: the rasterizer checks the area before dividing.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    x0, y0 = tri[0, 0], tri[0, 1]
    x1, y1 = tri[1, 0], tri[1, 1]
    x2, y2 = tri[2, 0], tri[2, 1]

    area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    w0 = ((x1 - xs) * (y2 - ys) - (x2 - xs) * (y1 - ys)) / area2
    w1 = ((x2 - xs) * (y0 - ys) - (x0 - xs) * (y2 - ys)) / area2
    w2 = 1.0 - w0 - w1  # sums to 1 by construction; cheaper than a third cross
    return np.stack([w0, w1, w2], axis=-1)


def interp_linear(weights, values):
    """Interpolate a per-vertex attribute with barycentric weights.

    Inputs:
      weights -- (..., 3) barycentric weights.
      values  -- (3,) attribute value per triangle vertex.
    Output: (...,) interpolated attribute.

    Valid for anything affine in screen space: NDC z (see the top of this
    file) and any attribute of an orthographic projection. UVs under
    perspective need interp_perspective instead.
    """
    return np.einsum("...i,i->...", weights, values)


def interp_perspective(weights, values, inv_ws):
    """Perspective-correct interpolation of a per-vertex attribute.

    Inputs:
      weights -- (..., 3) barycentric weights (screen space).
      values  -- (3,) attribute per triangle vertex (e.g. u or v).
      inv_ws  -- (3,) reciprocal clip-space w per triangle vertex.
    Output: (...,) perspective-correct attribute.

    How: attribute/w is affine in screen space for a planar triangle, so
    sum(w_i * a_i / w_i) and sum(w_i / w_i) can both be interpolated
    linearly; dividing the two at each point recovers the true attribute.
    Example pinned by the plan: a texture edge spanning a big depth change
    (w = 1 on the left, w = 9 on the right, u from 0 to 1) samples
    u = 0.1 at the screen midpoint, not the naive 0.5.
    """
    num = np.einsum("...i,i,i->...", weights, values, inv_ws)
    den = np.einsum("...i,i->...", weights, inv_ws)
    return num / den


def is_front_facing(ndc_xy):
    """True when a triangle in NDC (y up) is counter-clockwise.

    Input:  ndc_xy -- (3, 2) NDC positions (before the viewport y flip!).
    Output: bool.

    Why NDC and not screen space: the viewport flips y, which negates the
    signed area. Culling before the flip keeps the sign convention the
    whole pipeline was verified with: CCW seen from outside = front.
    Degenerate triangles (area exactly 0) count as back-facing; the
    rasterizer would drop them anyway.
    """
    p = np.asarray(ndc_xy, dtype=np.float64)
    area2 = (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (
        p[1, 1] - p[0, 1]
    )
    return bool(area2 > 0.0)


def clip_near_polygon(clip_points, uvs=None):
    """Clip a triangle against the near plane, in clip space.

    Inputs:
      clip_points -- (3, 4) homogeneous clip coordinates of one triangle.
      uvs         -- (3, 2) matching UVs, or None.
    Output: list of (points (3, 4), uvs (3, 2) or None) triangles.

    How: the near plane in clip space is exactly z + w >= 0 (with the
    projection matrix of transforms.perspective, z + w is proportional to
    -(z_eye + near), so the test reads "z_eye <= -near", i.e. in front of
    the near plane). Sutherland-Hodgman walks the polygon edges and emits
    inside vertices plus one intersection per crossing; a triangle yields
    a triangle or a quad, and the quad fans into two triangles. Crossing
    points interpolate all four clip coordinates and the UVs with the same
    parameter t, so texture continuity survives the cut.

    Why clip before the w-divide: points behind the camera have w <= 0 and
    would mirror onto the screen if divided first. After this clip every
    vertex has w >= near > 0.

    Degenerate slivers (duplicate points when a vertex lies exactly on the
    plane) are legal output; the rasterizer drops zero-area triangles.
    """
    points = np.asarray(clip_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 4 or points.shape[0] < 3:
        raise ValueError(f"clip_near_polygon: expected (N, 4), got {points.shape}")
    if uvs is not None:
        uvs = np.asarray(uvs, dtype=np.float64)
        if uvs.shape != (points.shape[0], 2):
            raise ValueError(
                f"clip_near_polygon: uvs shape {uvs.shape} != ({points.shape[0]}, 2)"
            )

    # Signed distance to the near plane, scaled: d >= 0 means "keep".
    d = points[:, 2] + points[:, 3]

    out_points = []
    out_uvs = []
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        di, dj = d[i], d[j]
        if di >= 0.0:
            out_points.append(points[i])
            if uvs is not None:
                out_uvs.append(uvs[i])
        if (di >= 0.0) != (dj >= 0.0):
            # Edge crosses the plane: emit the intersection at
            # t = di / (di - dj), which lies in 0..1 between the vertices.
            t = di / (di - dj)
            out_points.append(points[i] + t * (points[j] - points[i]))
            if uvs is not None:
                out_uvs.append(uvs[i] + t * (uvs[j] - uvs[i]))

    # Fan-triangulate the clipped polygon (3 or 4 vertices remain).
    triangles = []
    for k in range(1, len(out_points) - 1):
        tri_points = np.array([out_points[0], out_points[k], out_points[k + 1]])
        tri_uvs = None
        if uvs is not None:
            tri_uvs = np.array([out_uvs[0], out_uvs[k], out_uvs[k + 1]])
        triangles.append((tri_points, tri_uvs))
    return triangles


def draw_line(color_fb, x0, y0, x1, y1, color):
    """Draw one line into the framebuffer with dense sampling (stage 1).

    Inputs: color_fb -- (H, W, 3) float framebuffer; (x0, y0)-(x1, y1) --
            continuous screen coordinates; color -- (3,) linear color.
    Output: number of unique pixels the line touches (int).

    Why dense sampling instead of Bresenham: two samples per pixel along
    the major axis plus floor() to the containing pixel covers the same
    pixels without special-casing octants, and it is a few lines of NumPy.
    Out-of-bounds samples are simply dropped.
    """
    h, w = color_fb.shape[:2]
    length = float(max(abs(x1 - x0), abs(y1 - y0)))
    n = int(length * 2.0) + 2
    ts = np.linspace(0.0, 1.0, n)
    xs = x0 + ts * (x1 - x0)
    ys = y0 + ts * (y1 - y0)
    xi = np.floor(xs).astype(np.intp)
    yi = np.floor(ys).astype(np.intp)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    color_fb[yi[ok], xi[ok]] = color
    # Unique pixels: the dense sampling writes most pixels twice.
    return int(np.unique(yi[ok] * w + xi[ok]).size)


# ---------------------------------------------------------------------------
# Texture and mip pyramid
# ---------------------------------------------------------------------------

def build_mip_pyramid(img, levels=4):
    """Build a gamma-correct box-filter pyramid by successive halving.

    Inputs: img    -- (H, W, ...) float array already in LINEAR light.
            levels -- how many levels to produce (default 4).
    Output: list of arrays; level k has shape (H/2^k, W/2^k, ...).

    Each level averages 2x2 texel blocks of the level above. Averaging in
    linear light (not in sRGB) is the "gamma-correct" part: averaging sRGB
    values would make the mip levels darker than they should be. For the
    256x256 temple texture this yields 256, 128, 64 and 32.
    """
    img = np.asarray(img, dtype=np.float64)
    pyramid = [img]
    while len(pyramid) < levels:
        prev = pyramid[-1]
        if min(prev.shape[0], prev.shape[1]) < 2 or prev.shape[0] % 2 or prev.shape[1] % 2:
            break  # cannot halve further; stop early instead of cropping
        pyramid.append(
            0.25
            * (
                prev[0::2, 0::2]
                + prev[1::2, 0::2]
                + prev[0::2, 1::2]
                + prev[1::2, 1::2]
            )
        )
    return pyramid


def _sample_level(img, u, v, bilinear):
    """Sample one mip level at continuous UVs, wrapping at the edges.

    Inputs: img -- (H, W, 3) LINEAR float level; u, v -- (P,) UVs (any
            range, wrapped); bilinear -- nearest or 2x2 blend.
    Output: (P, 3) linear texels.

    Coordinate convention (see the top of this file): the continuous texel
    coordinate is x = u*W - 0.5 (so texel centers land on integers) and
    y = (1 - v)*H - 0.5 (v = 0 is the bottom image row; row 0 is stored at
    the top). Wrapping happens on the integer indices; UVs run past 0..1
    freely because the texture tiles seamlessly.
    """
    h, w = img.shape[0], img.shape[1]
    x = u * w - 0.5
    y = (1.0 - v) * h - 0.5

    if not bilinear:
        # Nearest = floor(continuous coordinate + 0.5) = the texel whose
        # center is closest; at the seam this picks the top row.
        col = np.floor(x + 0.5).astype(np.intp) % w
        row = np.floor(y + 0.5).astype(np.intp) % h
        return img[row, col]

    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    x0 %= w
    y0 %= h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h

    c00 = img[y0, x0]
    c10 = img[y0, x1]
    c01 = img[y1, x0]
    c11 = img[y1, x1]
    row_upper = c00 + (c10 - c00) * fx       # blend along u on the upper row
    row_lower = c01 + (c11 - c01) * fx       # blend along u on the lower row
    return row_upper + (row_lower - row_upper) * fy  # blend the rows along v


class Texture:
    """An sRGB image with a linear-light mip pyramid and a wrapping sampler.

    Building decodes sRGB once; sampling never touches sRGB again. fetch()
    groups pixels by mip level so each level is sampled with one vectorized
    operation per triangle, no matter how the levels mix within it.
    """

    def __init__(self, img, levels=4):
        """img: (H, W, 3) uint8 array, row 0 = top (as ppm_io returns)."""
        img = np.asarray(img)
        if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
            raise ValueError(
                f"Texture: expected (H, W, 3) uint8, got {img.shape} {img.dtype}"
            )
        linear = srgb_to_linear(img.astype(np.float64) / 255.0)
        self.levels = build_mip_pyramid(linear, levels=levels)
        self.width = img.shape[1]
        self.height = img.shape[0]

    def fetch(self, u, v, level, bilinear):
        """Sample the texture for a batch of pixels.

        Inputs: u, v -- (P,) UVs; level -- (P,) int mip levels;
                bilinear -- bool (nearest otherwise).
        Output: (P, 3) linear-light colors.
        """
        u = np.asarray(u, dtype=np.float64).ravel()
        v = np.asarray(v, dtype=np.float64).ravel()
        level = np.asarray(level, dtype=np.intp)
        if level.ndim == 0:
            # One level for the whole batch (mipmaps off): sample directly.
            return _sample_level(self.levels[int(level)], u, v, bilinear)
        level = level.ravel()
        out = np.empty((u.size, 3), dtype=np.float64)
        for lev in np.unique(level):
            mask = level == lev
            out[mask] = _sample_level(self.levels[int(lev)], u[mask], v[mask], bilinear)
        return out


def pixel_mip_level(tri, weights, u, v, tex_size, n_levels):
    """Per-pixel mip level from the analytic UV gradient (quotient rule).

    Inputs: tri       -- ScreenTriangle (needs uv, inv_w, pos);
            weights   -- (P, 3) barycentric weights of the pixels;
            u, v      -- (P,) perspective-correct UVs already computed;
            tex_size  -- larger texture dimension in texels;
            n_levels  -- number of pyramid levels.
    Output: (P,) int levels in [0, n_levels - 1].

    How: screen-space affine functions have constant gradients, so from
    the barycentric gradient matrix (db/dx and db/dy of each weight) we
    get constant gradients of U = sum(b_i * u_i / w_i), of V likewise,
    and of W1 = sum(b_i / w_i). The quotient rule then gives the real UV
    gradients per pixel: u = U / W1, du/dx = (dU/dx * W1 - U * dW1/dx) /
    W1^2. The texel footprint of a pixel is the larger of the u- and
    v-direction gradient magnitudes; its log2 is the mip level.
    """
    x0, y0 = tri.pos[0]
    x1, y1 = tri.pos[1]
    x2, y2 = tri.pos[2]
    # b = invM @ [x, y, 1], so d(b_i)/dx = invM[i, 0] and d(b_i)/dy = invM[i, 1].
    m = np.array([[x0, x1, x2], [y0, y1, y2], [1.0, 1.0, 1.0]])
    inv_m = np.linalg.inv(m)

    big_u = tri.uv[:, 0] * tri.inv_w  # u / w per vertex
    big_v = tri.uv[:, 1] * tri.inv_w
    big_w = tri.inv_w

    # Constant per-triangle gradients of the linear (pre-divide) quantities.
    d_u_dx = float(inv_m[:, 0] @ big_u)
    d_u_dy = float(inv_m[:, 1] @ big_u)
    d_v_dx = float(inv_m[:, 0] @ big_v)
    d_v_dy = float(inv_m[:, 1] @ big_v)
    d_w_dx = float(inv_m[:, 0] @ big_w)
    d_w_dy = float(inv_m[:, 1] @ big_w)

    den = weights @ big_w  # W1 per pixel, always > 0 inside the triangle
    num_u = u * den        # U = u * W1 (avoids a second dot product)
    num_v = v * den
    du_dx = (d_u_dx * den - num_u * d_w_dx) / (den * den)
    du_dy = (d_u_dy * den - num_u * d_w_dy) / (den * den)
    dv_dx = (d_v_dx * den - num_v * d_w_dx) / (den * den)
    dv_dy = (d_v_dy * den - num_v * d_w_dy) / (den * den)

    rho = np.maximum(np.hypot(du_dx, dv_dx), np.hypot(du_dy, dv_dy)) * tex_size
    lod = np.log2(np.maximum(rho, 1e-12))  # rho <= 1 texel/pixel -> level 0
    return np.clip(np.round(lod), 0, n_levels - 1).astype(np.intp)


# ---------------------------------------------------------------------------
# Scene description: camera, settings, per-triangle job, render result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Camera:
    """Pinhole camera for the default three-quarter view of the temple.

    The values frame the whole model in an 800x600 frame with margin: the
    spire tip lands near the top and all four base corners stay inside.
    All angles end in _deg (converted at the boundary), per the project
    conventions.
    """

    eye: tuple = (11.0, 8.0, 11.0)
    target: tuple = (0.0, 4.6, 0.0)
    up: tuple = (0.0, 1.0, 0.0)
    fovy_deg: float = 50.0
    near: float = 1.0
    far: float = 80.0


@dataclass
class RenderSettings:
    """Every switch the staged build needs; defaults are the final stage.

    Fields:
      width, height    -- framebuffer size in pixels.
      wireframe        -- stage 1: draw edges instead of filled triangles.
      use_depth        -- stage 3+: test and write the z-buffer.
      clip_near        -- stage 4+: near-plane clip in clip space.
      cull_backface    -- stage 5+: drop back-facing triangles.
      lambert          -- stage 6+: shade with ambient + diffuse.
      use_texture      -- stage 7+: sample the texture (else flat sand).
      bilinear         -- stage 8+: bilinear filtering (else nearest).
      mipmaps          -- stage 9+: choose a pyramid level per pixel.
      background_srgb  -- framebuffer clear color (sRGB).
      ambient, diffuse -- Lambert weights; light_dir points surface -> light.
    """

    width: int = 800
    height: int = 600
    wireframe: bool = False
    use_depth: bool = True
    clip_near: bool = True
    cull_backface: bool = True
    lambert: bool = True
    use_texture: bool = True
    bilinear: bool = True
    mipmaps: bool = True
    background_srgb: tuple = SKY_SRGB
    ambient: float = 0.3
    diffuse: float = 0.7
    light_dir: tuple = (0.35, 0.8, 0.65)


def stage_settings(step):
    """Return RenderSettings for one of the nine build stages (1..9).

    The stages are cumulative: each one adds a single technique to make
    the plan's "render after every step" regressions visible. Stage 9
    equals the defaults, i.e. the final frame.
    """
    if step not in STAGE_NAMES:
        raise ValueError(f"stage_settings: step must be 1..9, got {step}")
    return RenderSettings(
        wireframe=(step == 1),
        use_depth=(step >= 3),
        clip_near=(step >= 4),
        cull_backface=(step >= 5),
        lambert=(step >= 6),
        use_texture=(step >= 7),
        bilinear=(step >= 8),
        mipmaps=(step >= 9),
    )


@dataclass
class ScreenTriangle:
    """One triangle ready to rasterize, in screen space.

    Fields:
      pos     -- (3, 2) screen pixel coordinates (x right, y down).
      z_ndc   -- (3,) NDC depth per vertex (linear in screen space).
      inv_w   -- (3,) 1/w per vertex, for perspective-correct UVs.
      uv      -- (3, 2) UVs, or None for a flat color.
      color   -- (3,) linear-light albedo, used when uv is None.
      light   -- Lambert factor (1.0 when shading is off).
    """

    pos: np.ndarray
    z_ndc: np.ndarray
    inv_w: np.ndarray
    uv: np.ndarray | None
    color: np.ndarray
    light: float


@dataclass
class RenderResult:
    """What render_scene returns.

    color_linear -- (H, W, 3) float linear-light framebuffer (for tests).
    color_srgb   -- (H, W, 3) uint8, encoded once for writing/display.
    depth_ndc    -- (H, W) NDC z; +inf marks background.
    depth01      -- (H, W) (z_ndc + 1) / 2 in [0, 1]; background = 1.0.
    stats        -- dict: submitted, unprojectable, clipped, culled,
                    rasterized, pixels, seconds.
    """

    color_linear: np.ndarray
    color_srgb: np.ndarray
    depth_ndc: np.ndarray
    depth01: np.ndarray
    stats: dict


# ---------------------------------------------------------------------------
# The rasterizer core
# ---------------------------------------------------------------------------

def rasterize_triangle(color_fb, depth_fb, tri, texture, settings):
    """Fill one screen-space triangle into the framebuffers.

    Inputs:
      color_fb -- (H, W, 3) float linear framebuffer, modified in place.
      depth_fb -- (H, W) float NDC depth buffer (+inf = empty), modified
                  in place when settings.use_depth.
      tri      -- ScreenTriangle.
      texture  -- Texture for sampling when tri.uv is not None (else None).
      settings -- RenderSettings (depth test on/off, filtering, mipmaps).
    Output: number of pixels written (int).

    How: the triangle's pixel bounding box is scanned at pixel centers
    (integer + 0.5), barycentric weights give both the inside test (all
    weights >= 0) and the interpolation weights, the depth test keeps the
    nearest fragment (strict <, so ties keep the first writer), UVs go
    through the perspective-correct divide, and the shaded color is one
    write per surviving pixel.
    """
    h, w = color_fb.shape[:2]
    xs, ys = tri.pos[:, 0], tri.pos[:, 1]

    area2 = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
    if abs(area2) < 1e-12:
        return 0  # degenerate (also produced by clipping slivers)

    # Pixel bounding box: pixels whose CENTER (integer + 0.5) lies within
    # the box that contains the triangle, clamped to the framebuffer.
    x_lo = max(0, int(np.ceil(xs.min() - 0.5)))
    x_hi = min(w - 1, int(np.floor(xs.max() - 0.5)))
    y_lo = max(0, int(np.ceil(ys.min() - 0.5)))
    y_hi = min(h - 1, int(np.floor(ys.max() - 0.5)))
    if x_hi < x_lo or y_hi < y_lo:
        return 0

    cols = np.arange(x_lo, x_hi + 1)
    rows = np.arange(y_lo, y_hi + 1)
    grid_x, grid_y = np.meshgrid(cols + 0.5, rows + 0.5)
    px_idx, py_idx = np.meshgrid(cols, rows)
    weights = barycentric(grid_x.ravel(), grid_y.ravel(), tri.pos)
    xi = px_idx.ravel()
    yi = py_idx.ravel()

    inside = np.all(weights >= 0.0, axis=1)
    sel = np.flatnonzero(inside)
    if sel.size == 0:
        return 0

    wts = weights[sel]
    xi_sel = xi[sel]
    yi_sel = yi[sel]

    # Depth: NDC z is affine in screen space, so barycentric interpolation
    # is exact (see the top of this file). Smallest z wins; strict test.
    z = interp_linear(wts, tri.z_ndc)
    if settings.use_depth:
        ok = z < depth_fb[yi_sel, xi_sel]
        if not ok.any():
            return 0
        wts = wts[ok]
        xi_sel = xi_sel[ok]
        yi_sel = yi_sel[ok]
        z = z[ok]
        depth_fb[yi_sel, xi_sel] = z

    # Color: flat albedo or perspective-correct texture sample, shaded by
    # the per-triangle Lambert factor (all in linear light).
    if tri.uv is not None and texture is not None:
        u = interp_perspective(wts, tri.uv[:, 0], tri.inv_w)
        v = interp_perspective(wts, tri.uv[:, 1], tri.inv_w)
        if settings.mipmaps:
            tex_size = max(texture.width, texture.height)
            level = pixel_mip_level(tri, wts, u, v, tex_size, len(texture.levels))
        else:
            level = 0
        albedo = texture.fetch(u, v, level, settings.bilinear)
    else:
        albedo = np.broadcast_to(tri.color, (len(xi_sel), 3))

    color_fb[yi_sel, xi_sel] = albedo * tri.light
    return int(len(xi_sel))


def render_scene(mesh, texture, settings, camera=Camera()):
    """Render a whole mesh with the current stage settings.

    Inputs:
      mesh     -- objio.ObjMesh; positions are world coordinates (the
                  temple needs no model transform, so M = identity).
      texture  -- Texture (required when settings.use_texture).
      settings -- RenderSettings.
      camera   -- Camera (defaults to the pinned three-quarter view).
    Output: RenderResult.

    How: one MVP projection for all unique vertices, then per triangle:
    near clip (optional), w-divide, back-face cull (optional), viewport,
    fill. World-space face normals drive the Lambert factor; the shading
    is flat per triangle, which is exact for the axis-aligned boxes.
    """
    width, height = settings.width, settings.height
    if width < 1 or height < 1:
        raise ValueError(f"render_scene: bad size {width}x{height}")

    # Clear color: decode the sky to linear ONCE, so the final encode
    # round-trips it back to exactly the authored sRGB value.
    color_fb = np.empty((height, width, 3), dtype=np.float64)
    color_fb[:] = srgb_to_linear(np.array(settings.background_srgb))
    depth_fb = np.full((height, width), np.inf, dtype=np.float64)

    stats = {
        "submitted": 0,
        "unprojectable": 0,
        "clipped": 0,
        "culled": 0,
        "rasterized": 0,
        "pixels": 0,
        "seconds": 0.0,
    }
    t0 = time.perf_counter()

    # Camera chain: P @ V @ M, M = identity (mesh is authored in world space).
    p_matrix = perspective(camera.fovy_deg, width / height, camera.near, camera.far)
    v_matrix = look_at(camera.eye, camera.target, camera.up)
    clip, _ = project(p_matrix, v_matrix, np.eye(4), mesh.positions)  # (N, 4)

    light_dir = np.asarray(settings.light_dir, dtype=np.float64)
    light_dir = light_dir / np.linalg.norm(light_dir)
    sand_linear = srgb_to_linear(np.array(SAND_SRGB))
    ink_linear = srgb_to_linear(np.array(INK_SRGB))

    face_vertices = mesh.face_vertex_indices  # (T, 3)
    face_uvs = mesh.face_uv_indices           # (T, 3)
    stats["submitted"] = int(len(face_vertices))

    for t in range(len(face_vertices)):
        idx = face_vertices[t]
        pts4 = clip[idx]  # (3, 4) clip coordinates of the three corners
        w_corner = pts4[:, 3]

        # Without near clipping there is no safe divide for vertices at or
        # behind the eye; skip such triangles. (The default camera never
        # triggers this; stage 4 exists to make it exact instead of skip.)
        if not settings.clip_near and np.any(w_corner <= 0.0):
            stats["unprojectable"] += 1
            continue

        # UVs per corner (None when untextured or the file had no vt).
        tri_uv = None
        if settings.use_texture and texture is not None:
            uv_idx = face_uvs[t]
            if np.all(uv_idx >= 0):
                tri_uv = mesh.uvs[uv_idx]

        # Flat Lambert factor from the world-space face normal. Computed
        # from the source triangle, so clipped pieces stay consistent.
        p0, p1, p2 = mesh.positions[idx]
        if settings.lambert:
            normal = np.cross(p1 - p0, p2 - p0)
            norm_len = np.linalg.norm(normal)
            ndotl = float(normal @ light_dir) / norm_len if norm_len > 0.0 else 0.0
            light = settings.ambient + settings.diffuse * max(0.0, ndotl)
        else:
            light = 1.0

        # Near clip (stage 4+): z + w >= 0 in clip space, before the divide.
        pieces = [(pts4, tri_uv)]
        if settings.clip_near:
            if np.any(pts4[:, 2] + pts4[:, 3] < 0.0):
                stats["clipped"] += 1
                pieces = clip_near_polygon(pts4, tri_uv)

        for piece_pts, piece_uv in pieces:
            w = piece_pts[:, 3]
            ndc = piece_pts[:, :3] / w[:, None]

            # Back-face cull in NDC (y up), before the viewport flip.
            if settings.cull_backface and not is_front_facing(ndc[:, :2]):
                stats["culled"] += 1
                continue

            screen = viewport(ndc, width, height)  # (3, 3): x, y, depth01

            if settings.wireframe:
                # Stage 1: all three edges in ink, no depth involved.
                for e in range(3):
                    f = (e + 1) % 3
                    stats["pixels"] += draw_line(
                        color_fb,
                        screen[e, 0],
                        screen[e, 1],
                        screen[f, 0],
                        screen[f, 1],
                        ink_linear,
                    )
                stats["rasterized"] += 1
                continue

            tri = ScreenTriangle(
                pos=screen[:, :2],
                z_ndc=ndc[:, 2],
                inv_w=1.0 / w,
                uv=piece_uv,
                color=sand_linear,
                light=light,
            )
            drawn = rasterize_triangle(color_fb, depth_fb, tri, texture, settings)
            if drawn:
                stats["rasterized"] += 1
            stats["pixels"] += drawn

    stats["seconds"] = time.perf_counter() - t0

    # Depth image: (z_ndc + 1) / 2, background (+inf) pinned to 1.0 white.
    depth01 = np.where(np.isfinite(depth_fb), (depth_fb + 1.0) * 0.5, 1.0)
    # Encode to sRGB exactly once, at the very end.
    color_srgb = np.round(
        np.clip(linear_to_srgb(color_fb), 0.0, 1.0) * 255.0
    ).astype(np.uint8)

    return RenderResult(
        color_linear=color_fb,
        color_srgb=color_srgb,
        depth_ndc=depth_fb,
        depth01=depth01,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Command line entry point and output files
# ---------------------------------------------------------------------------

def write_png_preview(ppm_path):
    """Convert a written PPM to a PNG next to it, for quick viewing.

    Input:  ppm_path -- Path of an existing PPM file.
    Output: a .png file with the same stem.

    Pillow is imported HERE, on purpose: in this project it is allowed to
    convert images for display, never to render or filter them. The PNG
    is a convenience copy; the PPM stays the rendered artifact.
    """
    from PIL import Image  # conversion only, see docstring

    _, _, img = ppm_io.read_ppm(ppm_path)
    Image.fromarray(img).save(ppm_path.with_suffix(".png"))


def depth_to_rgb(depth01):
    """Turn the depth image into a gray uint8 RGB image (PPM-ready).

    Input:  depth01 -- (H, W) float in [0, 1];
    Output: (H, W, 3) uint8, same value in all three channels.
    Encoding: dark = near, white = background (there is no depth there).
    """
    gray = np.round(np.clip(depth01, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _print_stats(step, settings, result):
    """Print one line of stage info plus the triangle/pixel counters."""
    s = result.stats
    percent = 100.0 * s["pixels"] / float(settings.width * settings.height)
    print(
        f"stage {step} ({STAGE_NAMES[step]}) {settings.width}x{settings.height} "
        f"in {s['seconds']:.2f} s"
    )
    print(
        f"  triangles: {s['submitted']} submitted, {s['unprojectable']} unprojectable, "
        f"{s['clipped']} near-clipped, {s['culled']} back-culled, "
        f"{s['rasterized']} rasterized"
    )
    print(f"  pixels written: {s['pixels']} ({percent:.1f}% of the frame)")


def render_stage(step, mesh, texture, write_stage=True, write_final=False):
    """Render one stage and write the requested outputs.

    Inputs: step -- 1..9; mesh, texture -- scene; write_stage -- write
            out/step{N}_{name}.ppm (+ depth when the stage has one);
            write_final -- write out/temple_raster.ppm and temple_depth.ppm.
    Output: RenderResult (also printed stats).
    """
    settings = stage_settings(step)
    result = render_scene(mesh, texture, settings)

    label = f"step{step}_{STAGE_NAMES[step]}"
    if write_stage:
        ppm_path = OUT_DIR / f"{label}.ppm"
        ppm_io.write_ppm(ppm_path, result.color_srgb)
        write_png_preview(ppm_path)
        if settings.use_depth:
            depth_path = OUT_DIR / f"{label}_depth.ppm"
            ppm_io.write_ppm(depth_path, depth_to_rgb(result.depth01))
    if write_final:
        final_path = OUT_DIR / "temple_raster.ppm"
        ppm_io.write_ppm(final_path, result.color_srgb)
        write_png_preview(final_path)
        if settings.use_depth:
            ppm_io.write_ppm(OUT_DIR / "temple_depth.ppm", depth_to_rgb(result.depth01))

    _print_stats(step, settings, result)
    return result


def main(argv=None):
    """CLI: final render by default; --step N or --steps for the stages."""
    parser = argparse.ArgumentParser(
        description="Software rasterizer for the stepped temple (phase 3)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--step",
        type=int,
        choices=range(1, 10),
        metavar="N",
        help="render a single build stage (1..9)",
    )
    group.add_argument(
        "--steps",
        action="store_true",
        help="render all nine build stages, then the final frame",
    )
    args = parser.parse_args(argv)

    obj_path = DATA_DIR / "temple.obj"
    tex_path = DATA_DIR / "relief.ppm"
    if not (obj_path.is_file() and tex_path.is_file()):
        print("missing data/ assets -- run: python make_model.py")
        return 1

    mesh = objio.load_obj(obj_path)
    texture = Texture(ppm_io.read_ppm(tex_path)[2])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.steps:
        for step in range(1, 10):
            render_stage(step, mesh, texture, write_stage=True, write_final=(step == 9))
    elif args.step is not None:
        render_stage(args.step, mesh, texture, write_stage=True, write_final=False)
    else:
        # Default: the final frame only, under the pinned output names.
        render_stage(9, mesh, texture, write_stage=False, write_final=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
