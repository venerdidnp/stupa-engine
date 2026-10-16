"""
raytrace.py -- the ray tracer (build plan phase 4).

Where the rasterizer walks triangles and scatters their fragments into the
framebuffer, this renderer walks the framebuffer and casts one ray through
every pixel center instead. For the pinned pinhole camera the two must
agree: same scene, same camera, same texture, same light, same sRGB
encoding, so the phase-5 comparison can diff the two images and expect
differences only where a silhouette edge falls between pixel centers.

Pipeline, per pixel:
    pixel center -> world-space ray (pixel_rays) -> nearest triangle along
    the ray (Moller-Trumbore, first_hit) -> hit point -> UV from the hit
    barycentrics -> shade with the same flat Lambert term the rasterizer
    uses -> framebuffer (linear light), encoded to sRGB once at the end.

How the two renderers stay comparable (on purpose, not by accident):
    - Camera, projection and viewport come from the same modules
      (transforms + raster3d.Camera), so a pixel center maps to the same
      world ray render_scene() implicitly traces.
    - Texture, mip selection (pixel_mip_level), sRGB helpers and the sky /
      sand colors are imported from raster3d instead of duplicated, so a
      change to the shared look affects both renderers at once.
    - Depth uses the same convention: NDC z of the hit point, background
      +inf in depth_ndc and 1.0 in depth01.
    - Back-face handling: the ray test is two-sided, but for a union of
      closed solids the nearest hit is always on a front-facing triangle
      (reaching a back face means the ray already crossed an exit surface
      before), so both renderers draw the same surfaces.

What the RenderSettings switches mean here (they exist for the staged
rasterizer, and not all of them can apply to a ray tracer):
    wireframe, use_depth, clip_near, cull_backface -- ignored. A ray
    tracer never splits triangles (t_min keeps self-hits away, no near
    plane needed), first_hit IS the depth test, and back faces are never
    the nearest surface.
    lambert, use_texture, bilinear, mipmaps -- honored, with the same
    meaning (and the same code) as in the rasterizer.

Memory: rays are cast in horizontal strips of CHUNK_ROWS rows, so the
(P, T, 3) Moller-Trumbore temporaries stay at a few tens of MB instead of
scaling with the whole frame.

Outputs (written by main()):
    out/temple_ray.ppm        color, 800x600
    out/temple_ray_depth.ppm  depth, same encoding as the rasterizer
    out/temple_ray.png        preview (Pillow converts only)

Run:
    python raytrace.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import objio
import ppm_io
from raster3d import (
    Camera,
    RenderResult,
    RenderSettings,
    SAND_SRGB,
    ScreenTriangle,
    Texture,
    barycentric,
    depth_to_rgb,
    linear_to_srgb,
    pixel_mip_level,
    srgb_to_linear,
    write_png_preview,
)
from transforms import look_at, perspective, project, viewport

# Where the generated assets live and where renders go (script-relative,
# so the working directory does not matter).
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "out"

# Rows of rays per strip (see "Memory" above). Four rows at 800 pixels
# wide keeps the biggest (P, T, 3) temporary around 14 MB.
CHUNK_ROWS = 4


def pixel_rays(camera, width, height, xs, ys):
    """World-space rays through a batch of continuous pixel coordinates.

    Inputs:
      camera -- raster3d.Camera (eye, target, up, fovy_deg, ...).
      width, height -- framebuffer size in pixels.
      xs, ys -- (P,) pixel coordinates (0 = left/top frame edge, so pixel
                centers sit at integers + 0.5, exactly the positions
                render_scene() samples).
    Output: origins (P, 3) (the eye, repeated), dirs (P, 3).

    How: pixel -> NDC with the inverse of transforms.viewport
        x_ndc = 2*x/width - 1,   y_ndc = 1 - 2*y/height
    then NDC -> eye-space direction. On the near plane (eye z = -1, w = 1)
    the projection gives x_ndc = (f/aspect)*x_eye and y_ndc = f*y_eye with
    f = 1/tan(fovy/2), so the eye-space direction through that NDC point
    is (x_ndc*aspect/f, y_ndc/f, -1). Rotating it to world space uses the
    rows of the look_at matrix (right, up_cam, back):
        dir = right * (x_ndc*aspect/f) + up_cam * (y_ndc/f) - back

    Why the directions are NOT normalized: every intersection test is
    invariant to the length of dir, and keeping the z component at -1 in
    eye space keeps the numbers small and hand-checkable. t values are
    therefore "distance per unit of this scaling", which is all the
    depth ordering needs.
    """
    eye = np.asarray(camera.eye, dtype=np.float64)
    aspect = width / float(height)
    f = 1.0 / np.tan(np.radians(camera.fovy_deg) * 0.5)

    v_matrix = look_at(camera.eye, camera.target, camera.up)
    right = v_matrix[0, :3]
    up_cam = v_matrix[1, :3]
    back = v_matrix[2, :3]

    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    x_ndc = 2.0 * xs / width - 1.0
    y_ndc = 1.0 - 2.0 * ys / height

    dirs = (
        (x_ndc * (aspect / f))[:, None] * right[None, :]
        + (y_ndc / f)[:, None] * up_cam[None, :]
        - back[None, :]
    )
    origins = np.broadcast_to(eye, dirs.shape).copy()
    return origins, dirs


def camera_rays(camera, width, height):
    """Rays through every pixel center of the frame (row 0 = top).

    Output: origins (width*height, 3), dirs (width*height, 3), row-major
    (index = y*width + x). Used by the tests; ray_trace() itself casts
    strip by strip with pixel_rays() for memory reasons.
    """
    xs = np.arange(width, dtype=np.float64) + 0.5
    ys = np.arange(height, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    return pixel_rays(camera, width, height, grid_x.reshape(-1), grid_y.reshape(-1))


def intersect_triangles(origins, dirs, v0, v1, v2, t_min=1e-9):
    """Moller-Trumbore intersections between rays and triangles.

    Inputs:
      origins, dirs -- (P, 3) ray origins and directions (dirs need not be
                       normalized).
      v0, v1, v2   -- (T, 3) triangle corners, or (3,) for a single one.
      t_min        -- a hit needs t > t_min (keeps self-hits away).
    Output: (t, u, v), each (P, T) -- or (P,) for a single triangle:
      t -- ray parameter of the hit (origin + t*dir); inf on a miss.
      u, v -- barycentric weights of v1 and v2 at the hit point (v0's
           weight is 1 - u - v); 0 on a miss.

    How: solve o + t*d = v0 + u*e1 + v*e2 (e1 = v1 - v0, e2 = v2 - v0)
    with Cramer's rule. p = d x e2 and det = e1 . p; a triangle seen
    edge-on (det = 0, tolerance 1e-12) can never be hit. Then with
    tvec = o - v0 and q = tvec x e1:
        u = (tvec . p) / det,  v = (d . q) / det,  t = (e2 . q) / det
    The hit counts as inside iff u >= 0, v >= 0 and u + v <= 1 -- edges
    count as inside, matching the rasterizer's "all weights >= 0" test,
    so silhouette pixels keep the same side on both renderers.

    The test is deliberately two-sided (no culling): for the closed temple
    the nearest hit is always the outside surface, which is exactly what
    the rasterizer draws after back-face culling, and two-sidedness also
    keeps the function usable for single triangles of any winding.
    """
    origins = np.asarray(origins, dtype=np.float64)
    dirs = np.asarray(dirs, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    single = v0.ndim == 1
    if single:  # a bare (3,) triangle becomes one row, squeezed on return
        v0, v1, v2 = v0[None, :], v1[None, :], v2[None, :]

    e1 = v1 - v0                                     # (T, 3)
    e2 = v2 - v0                                     # (T, 3)
    p = np.cross(dirs[:, None, :], e2[None, :, :])   # (P, T, 3)
    det = np.einsum("ptk,tk->pt", p, e1)             # (P, T)

    hit = np.abs(det) >= 1e-12
    inv_det = np.zeros_like(det)
    np.divide(1.0, det, out=inv_det, where=hit)

    tvec = origins[:, None, :] - v0[None, :, :]      # (P, T, 3)
    u = np.einsum("ptk,ptk->pt", p, tvec) * inv_det
    q = np.cross(tvec, e1[None, :, :])               # (P, T, 3)
    v = np.einsum("ptk,pk->pt", q, dirs) * inv_det
    t = np.einsum("ptk,tk->pt", q, e2) * inv_det

    hit &= (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > t_min)
    t = np.where(hit, t, np.inf)
    u = np.where(hit, u, 0.0)
    v = np.where(hit, v, 0.0)
    if single:
        return t[:, 0], u[:, 0], v[:, 0]
    return t, u, v


def first_hit(origins, dirs, mesh, t_min=1e-9):
    """Nearest triangle of a mesh along each ray.

    Inputs: origins (P, 3), dirs (P, 3); mesh -- objio.ObjMesh.
    Output: (tri, t, u, v, point):
      tri   -- (P,) int triangle index, -1 when the ray misses everything;
      t     -- (P,) hit parameter, inf on a miss;
      u, v  -- (P,) hit barycentrics (0 on a miss);
      point -- (P, 3) hit position (the ray origin on a miss).

    How: one vectorized intersect_triangles call against all T triangles,
    then argmin over the t axis. np.argmin returns the FIRST minimal
    index, which extends the rasterizer's tie rule to coplanar seams (the
    stacked boxes' mating faces): the rasterizer's strict depth test also
    keeps the triangle drawn first, and the OBJ lists lower boxes first,
    so both renderers pick the same mate at a shared boundary plane.
    """
    v0 = mesh.positions[mesh.face_vertex_indices[:, 0]]
    v1 = mesh.positions[mesh.face_vertex_indices[:, 1]]
    v2 = mesh.positions[mesh.face_vertex_indices[:, 2]]
    t, u, v = intersect_triangles(origins, dirs, v0, v1, v2, t_min)

    best = np.argmin(t, axis=1)               # (P,) first minimum
    rows = np.arange(t.shape[0])
    t_best = t[rows, best]
    u_best = u[rows, best]
    v_best = v[rows, best]

    miss = ~np.isfinite(t_best)
    tri = np.where(miss, -1, best)
    t_best = np.where(miss, np.inf, t_best)
    point = origins + np.where(miss, 0.0, t_best)[:, None] * dirs
    return tri, t_best, u_best, v_best, point


def _triangle_light(mesh, settings):
    """Per-triangle flat Lambert factor, word for word the rasterizer's.

    Inputs: mesh -- ObjMesh; settings -- RenderSettings (lambert, ambient,
            diffuse, light_dir).
    Output: (T,) float factors (all 1.0 when settings.lambert is False).

    Why per triangle and not per pixel: the rasterizer shades a triangle
    with one factor from its world-space normal, and this renderer must
    produce the same numbers, so it computes the identical expression
    once per triangle instead of once per hit.
    """
    face = mesh.face_vertex_indices
    if not settings.lambert:
        return np.ones(len(face), dtype=np.float64)
    p0 = mesh.positions[face[:, 0]]
    p1 = mesh.positions[face[:, 1]]
    p2 = mesh.positions[face[:, 2]]
    light_dir = np.asarray(settings.light_dir, dtype=np.float64)
    light_dir = light_dir / np.linalg.norm(light_dir)
    normal = np.cross(p1 - p0, p2 - p0)
    norm_len = np.linalg.norm(normal, axis=1)
    ndotl = np.zeros(len(face), dtype=np.float64)
    ok = norm_len > 0.0
    ndotl[ok] = (normal[ok] @ light_dir) / norm_len[ok]
    return settings.ambient + settings.diffuse * np.maximum(0.0, ndotl)


def ray_trace(mesh, texture, settings, camera=Camera()):
    """Render a whole mesh by casting one ray per pixel center.

    Inputs:
      mesh     -- objio.ObjMesh in world space (same input as
                  render_scene: the temple needs no model transform).
      texture  -- Texture, used when settings.use_texture (may be None).
      settings -- RenderSettings: width/height, background, lambert,
                  use_texture, bilinear, mipmaps. The geometric switches
                  are rasterizer-only, see the module docstring.
      camera   -- Camera (defaults to the pinned three-quarter view).
    Output: RenderResult with the same conventions as render_scene:
      color_linear (H, W, 3) linear light, color_srgb (H, W, 3) uint8,
      depth_ndc (background +inf), depth01 (background 1.0), and a stats
      dict with keys rays, hits, seconds.

    How: pre-project every vertex once (depth of hits, screen-space mip
    levels), precompute the per-triangle light, then walk the frame in
    CHUNK_ROWS-row strips. Per strip: cast the rays through the strip's
    pixel centers, find the nearest hit, and for the hit pixels evaluate
    exactly what a rasterizer evaluates at that center:
      - UV from the hit barycentrics (an exact surface UV; there is no
        1/w machinery in a ray tracer),
      - mip level from pixel_mip_level fed with the pixel's SCREEN-space
        barycentrics (the same input render_scene gives it; the 3D
        barycentrics would give a subtly wrong level wherever perspective
        is strong),
      - albedo * flat light, and NDC depth from projecting the hit point.
    """
    width, height = settings.width, settings.height
    if width < 1 or height < 1:
        raise ValueError(f"ray_trace: bad size {width}x{height}")

    # Clear color: decode the sky to linear ONCE, mirroring render_scene,
    # so the final encode round-trips it to the authored sRGB value.
    color_fb = np.empty((height, width, 3), dtype=np.float64)
    color_fb[:] = srgb_to_linear(np.array(settings.background_srgb))
    depth_fb = np.full((height, width), np.inf, dtype=np.float64)

    stats = {"rays": 0, "hits": 0, "seconds": 0.0}
    t0 = time.perf_counter()

    p_matrix = perspective(camera.fovy_deg, width / height, camera.near, camera.far)
    v_matrix = look_at(camera.eye, camera.target, camera.up)

    # Vertex pre-pass: clip/NDC (depth of hits) and screen pixels (mip
    # levels). Vertices behind the eye produce garbage here; triangles
    # touching them fall back to mip level 0 below.
    clip_all, ndc_all = project(p_matrix, v_matrix, np.eye(4), mesh.positions)
    w_all = clip_all[:, 3]
    screen_all = viewport(ndc_all, width, height)          # (N, 3): x, y, depth01

    light_all = _triangle_light(mesh, settings)            # (T,)
    sand_linear = srgb_to_linear(np.array(SAND_SRGB))

    face_idx = mesh.face_vertex_indices                    # (T, 3)
    face_uv_idx = mesh.face_uv_indices                     # (T, 3)
    if mesh.uvs.size > 0:
        uv_ok_tri = (face_uv_idx >= 0).all(axis=1)         # (T,)
    else:
        uv_ok_tri = np.zeros(len(face_idx), dtype=bool)
    textured = settings.use_texture and texture is not None
    tex_size = max(texture.width, texture.height) if texture is not None else 1
    n_levels = len(texture.levels) if texture is not None else 1

    col_centers = np.arange(width, dtype=np.float64) + 0.5
    for y0 in range(0, height, CHUNK_ROWS):
        y1 = min(y0 + CHUNK_ROWS, height)
        row_centers = np.arange(y0, y1, dtype=np.float64) + 0.5
        grid_x, grid_y = np.meshgrid(col_centers, row_centers)

        origins, dirs = pixel_rays(
            camera, width, height, grid_x.reshape(-1), grid_y.reshape(-1)
        )
        tri, _, u_hit, v_hit, point = first_hit(origins, dirs, mesh)

        stats["rays"] += int(grid_x.size)
        hit = tri >= 0
        n_hits = int(np.count_nonzero(hit))
        stats["hits"] += n_hits
        if n_hits == 0:
            continue

        flat = np.flatnonzero(hit.ravel())     # row-major ray indices
        ti = tri[flat]                         # (H,) hit triangle per pixel
        px = flat % width
        py = y0 + flat // width

        # Depth: project the hit points with the same matrices the
        # rasterizer uses, so NDC z agrees to float precision.
        _, ndc_pts = project(p_matrix, v_matrix, np.eye(4), point[flat])
        depth_fb[py, px] = ndc_pts[:, 2]

        # Barycentric weights of the hit point relative to its triangle:
        # MT's (u, v) ARE the weights of v1 and v2, so v0's weight is
        # simply 1 - u - v.
        bary = np.stack(
            [1.0 - u_hit[flat] - v_hit[flat], u_hit[flat], v_hit[flat]], axis=1
        )

        # Albedo: flat sand by default; textured triangles get the exact
        # surface UV at the hit point.
        albedo = np.empty((n_hits, 3), dtype=np.float64)
        albedo[:] = sand_linear
        if textured:
            sel = np.flatnonzero(uv_ok_tri[ti])   # hits with a full UV set
            if sel.size:
                uv_tri = mesh.uvs[face_uv_idx[ti[sel]]]             # (S, 3, 2)
                uv_px = np.einsum("si,sik->sk", bary[sel], uv_tri)  # (S, 2)
                level = 0
                if settings.mipmaps:
                    level = np.zeros(sel.size, dtype=np.intp)
                    sx = (flat[sel] % width) + 0.5
                    sy = (y0 + flat[sel] // width) + 0.5
                    # One group per triangle: pixel_mip_level derives the
                    # level from screen-space affine gradients, so it
                    # needs the SCREEN barycentrics of the pixels, which
                    # generally differ from the 3D ones (perspective).
                    for t_idx in np.unique(ti[sel]):
                        here = np.flatnonzero(ti[sel] == t_idx)
                        idx3 = face_idx[t_idx]
                        if np.any(w_all[idx3] <= 0.0):
                            continue  # vertex behind the eye: keep level 0
                        tri_s = ScreenTriangle(
                            pos=screen_all[idx3, :2],
                            z_ndc=ndc_all[idx3, 2],
                            inv_w=1.0 / w_all[idx3],
                            uv=mesh.uvs[face_uv_idx[t_idx]],
                            color=sand_linear,
                            light=1.0,
                        )
                        weights = barycentric(sx[here], sy[here], tri_s.pos)
                        level[here] = pixel_mip_level(
                            tri_s,
                            weights,
                            uv_px[here, 0],
                            uv_px[here, 1],
                            tex_size,
                            n_levels,
                        )
                albedo[sel] = texture.fetch(
                    uv_px[:, 0], uv_px[:, 1], level, settings.bilinear
                )

        color_fb[py, px] = albedo * light_all[ti][:, None]

    stats["seconds"] = time.perf_counter() - t0

    depth01 = np.where(np.isfinite(depth_fb), (depth_fb + 1.0) * 0.5, 1.0)
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


def _print_stats(settings, result):
    """Print one line of render info plus the ray counters."""
    s = result.stats
    percent = 100.0 * s["hits"] / float(settings.width * settings.height)
    print(f"ray trace {settings.width}x{settings.height} in {s['seconds']:.2f} s")
    print(
        f"  rays: {s['rays']} cast, {s['hits']} hit geometry "
        f"({percent:.1f}% of the frame), {s['rays'] - s['hits']} hit sky"
    )


def main(argv=None):
    """CLI: render the final frame with the ray tracer."""
    parser = argparse.ArgumentParser(
        description="Ray tracer for the stepped temple (phase 4)."
    )
    parser.parse_args(argv)

    obj_path = DATA_DIR / "temple.obj"
    tex_path = DATA_DIR / "relief.ppm"
    if not (obj_path.is_file() and tex_path.is_file()):
        print("missing data/ assets -- run: python make_model.py")
        return 1

    mesh = objio.load_obj(obj_path)
    texture = Texture(ppm_io.read_ppm(tex_path)[2])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    settings = RenderSettings()  # defaults = the final-stage look
    result = ray_trace(mesh, texture, settings)

    ppm_path = OUT_DIR / "temple_ray.ppm"
    ppm_io.write_ppm(ppm_path, result.color_srgb)
    write_png_preview(ppm_path)
    ppm_io.write_ppm(OUT_DIR / "temple_ray_depth.ppm", depth_to_rgb(result.depth01))

    _print_stats(settings, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
