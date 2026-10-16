"""
tests.py -- all numeric checks for this project, one section per phase.

Every check prints the actual numbers next to the expected ones, so a
failure is readable without a debugger. Expected values come from the
reference tables in PLAN.md; nothing here is tuned to make a test pass.

Run:  python tests.py   (exit code 0 = all checks passed)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

import make_model
import objio
import ppm_io
import raster3d
from transforms import (
    look_at,
    perspective,
    project,
    rot_x,
    rot_y,
    rot_z,
    scale,
    translate,
    viewport,
)

# Bookkeeping for the final summary line.
_TOTAL = 0
_FAILURES = []


def fmt(v):
    """Format a number or array for printing (ints as ints, floats 6 dp)."""
    a = np.asarray(v)
    if np.issubdtype(a.dtype, np.integer) or a.dtype == np.bool_:
        return np.array2string(a.astype(np.int64), separator=", ")
    a = a.astype(np.float64)
    if a.ndim == 0:
        return f"{float(a):.6f}"
    return np.array2string(a, precision=6, separator=", ")


def check(name, actual, expected, atol=1e-9):
    """Print actual vs expected for one value and record pass/fail.

    Why atol with rtol=0: the reference numbers in the plan are exact
    decimals of determinate formulas; a relative tolerance could hide
    errors at zero (e.g. an "almost zero" coordinate that should be 0).
    """
    global _TOTAL
    _TOTAL += 1

    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    ok = a.shape == e.shape and np.allclose(a, e, rtol=0.0, atol=atol)

    print(f"[{'ok  ' if ok else 'FAIL'}] {name}")
    print(f"        actual   = {fmt(actual)}")
    print(f"        expected = {fmt(expected)}")
    if not ok:
        _FAILURES.append(name)
    return ok


def phase1():
    """Phase 1 checks: transforms, projection, viewport (PLAN.md section 2)."""
    print("=== Phase 1: math core (transforms) ===")

    p = np.array([2.0, 3.0, 4.0, 1.0])  # the plan's test point

    # --- Reference row 1: translate first, then rotate -------------------
    # "A then B" composes as B @ A @ p, so the rotation goes on the left.
    M1 = rot_y(90) @ translate(1, 0, -1)
    print("  M1 = rot_y(90) @ translate(1, 0, -1) =")
    print(np.array2string(M1, precision=0, suppress_small=True))
    world1 = M1 @ p
    check(
        "(2,3,4) translate(1,0,-1) then rot_y(90) -> (3,3,-3)",
        world1[:3],
        [3.0, 3.0, -3.0],
    )

    # --- Reference row 2: perspective divide with focal distance d = 2 ---
    # Route it through the real perspective() function: a focal length of
    # d in NDC units is f = 1/tan(fovy/2) = 2, i.e. fovy = 2*atan(1/2).
    # For the point (3,3,-3): x_ndc = d * 3 / 3 = 2, same for y.
    fovy_deg = float(np.degrees(2.0 * np.arctan(0.5)))
    P_d2 = perspective(fovy_deg, 1.0, 1.0, 50.0)
    _, ndc_d2 = project(P_d2, np.eye(4), M1, p[None, :3])
    check(
        "perspective divide of (3,3,-3) with d=2 -> (2,2)",
        ndc_d2[0, :2],
        [2.0, 2.0],
    )

    # --- Reference row 3: rotate first, then translate -------------------
    # Same two matrices, reversed order: the translation now happens in
    # world space, so the point lands at (3+1+... ) -> (5,3,-3).
    M2 = translate(1, 0, -1) @ rot_y(90)
    print("  M2 = translate(1, 0, -1) @ rot_y(90) =")
    print(np.array2string(M2, precision=0, suppress_small=True))
    world2 = M2 @ p
    check(
        "(2,3,4) rot_y(90) then translate(1,0,-1) -> (5,3,-3)",
        world2[:3],
        [5.0, 3.0, -3.0],
    )

    # --- Reference row 4: moving the camera vs moving the model ----------
    # Camera at (1,0,0) still looking down -z (target one step ahead) is
    # equivalent to translating the whole world by (-1,0,0).
    p5 = np.array([5.0, 0.0, 0.0, 1.0])
    V_shifted = look_at([1.0, 0.0, 0.0], [1.0, 0.0, -1.0], [0.0, 1.0, 0.0])
    in_moved_camera = V_shifted @ p5
    in_moved_model = translate(-1, 0, 0) @ p5
    check(
        "camera moved +1 on x: world (5,0,0) -> eye (4,0,0)",
        in_moved_camera[:3],
        [4.0, 0.0, 0.0],
    )
    check(
        "model moved -1 on x: world (5,0,0) -> eye (4,0,0)",
        in_moved_model[:3],
        [4.0, 0.0, 0.0],
    )

    # --- Reference rows 5-6: depth mapping of perspective(n=1, f=9) ------
    P = perspective(60.0, 1.0, 1.0, 9.0)
    check("perspective n=1 f=9: A = P[2,2]", float(P[2, 2]), -1.25)
    check("perspective n=1 f=9: B = P[2,3]", float(P[2, 3]), -2.25)
    depth_pts = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -9.0], [0.0, 0.0, -3.0]])
    _, ndc_z = project(P, np.eye(4), np.eye(4), depth_pts)
    check(
        "perspective n=1 f=9: eye z=-1,-9,-3 -> ndc z=-1,+1,0.5",
        ndc_z[:, 2],
        [-1.0, 1.0, 0.5],
    )

    # --- Extra: the Phase 6 worked example, end to end -------------------
    # Same scene the plan uses for the worked example. All values below
    # are its reference numbers; this exercises look_at + perspective +
    # project + viewport as one chain.
    eye = [0.0, 0.0, 8.0]
    V = look_at(eye, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    P60 = perspective(60.0, 1.0, 1.0, 50.0)
    clip8, ndc8 = project(P60, V, M1, p[None, :3])
    check("worked example: world point -> (3,3,-3)", (M1 @ p)[:3], [3.0, 3.0, -3.0])
    check(
        "worked example: eye point -> (3,3,-11)",
        (V @ (M1 @ p))[:3],
        [3.0, 3.0, -11.0],
    )
    check(
        "worked example: clip -> (5.196, 5.196, 9.408, 11)",
        np.round(clip8[0], 3),
        [5.196, 5.196, 9.408, 11.0],
    )
    check(
        "worked example: ndc -> (0.472, 0.472, 0.855)",
        np.round(ndc8[0], 3),
        [0.472, 0.472, 0.855],
    )
    screen = viewport(ndc8, 800, 600)[0]
    check("worked example: screen pixel x -> 589", np.round(screen[0]), 589.0)
    check("worked example: screen pixel y -> 158", np.round(screen[1]), 158.0)
    check("worked example: depth -> 0.928", np.round(screen[2], 3), 0.928)

    # --- Extra sanity: the remaining Phase 1 functions -------------------
    # Hand-derivable right-hand-rule facts, cheap insurance against a
    # sign flip in the unused-until-now rotations and scale.
    check("rot_x(90) maps +y to +z", (rot_x(90) @ [0, 1, 0, 1])[:3], [0.0, 0.0, 1.0])
    check("rot_z(90) maps +x to +y", (rot_z(90) @ [1, 0, 0, 1])[:3], [0.0, 1.0, 0.0])
    check("scale(2,3,4) on (1,1,1) -> (2,3,4)", (scale(2, 3, 4) @ [1, 1, 1, 1])[:3], [2.0, 3.0, 4.0])


def phase2():
    """Phase 2 checks: PPM I/O, OBJ loader, generated temple and texture."""
    print("=== Phase 2: I/O and model ===")

    # --- PPM write/read round trip on an exactly-known byte pattern -----
    # (3, 4, 3) = 4x3 pixel image, three RGB bytes per pixel, all
    # distinct-ish values so a row/column mix-up cannot pass unnoticed.
    pattern = ((np.arange(36) * 7) % 251).astype(np.uint8).reshape(3, 4, 3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pattern.ppm"
        ppm_io.write_ppm(path, pattern)
        width, height, back = ppm_io.read_ppm(path)
        check("ppm_io: round trip size (w, h) = (4, 3)", [width, height], [4, 3])
        check("ppm_io: round trip, every byte identical", back.ravel(), pattern.ravel())

    # --- Loader on a hand-written OBJ: all index forms, negatives, quads -
    # Face 1 is a 'v/vt/vn' quad -> must fan to 2 triangles, keeping the
    # per-corner vt indices. Face 2 is 'v//vn' with negative vertex
    # indices -> must resolve to (0, 1, 2) and carry no UVs (-1).
    obj_lines = [
        "v -1 -1 0",
        "v  1 -1 0",
        "v  1  1 0",
        "v -1  1 0",
        "vt 0 0",
        "vt 1 0",
        "vt 1 1",
        "vt 0 1",
        "vn 0 0 1",
        "f 1/1/1 2/2/1 3/3/1 4/4/1",
        "f -4//1 -3//1 -2//1",
    ]
    mesh = objio.load_obj_lines(obj_lines)
    check("loader: positions parsed", len(mesh.positions), 4)
    check("loader: uvs parsed", len(mesh.uvs), 4)
    check("loader: triangles (quad fan + plain triangle)", len(mesh.face_vertex_indices), 3)
    check(
        "loader: fan order (a,b,c)+(a,c,d), negatives resolved",
        mesh.face_vertex_indices.ravel(),
        [0, 1, 2, 0, 2, 3, 0, 1, 2],
    )
    check(
        "loader: per-corner vt indices kept, v//vn -> -1",
        mesh.face_uv_indices.ravel(),
        [0, 1, 2, 0, 2, 3, -1, -1, -1],
    )
    check(
        "loader: v flipped once (vt v=0 -> uv v=1)",
        mesh.uvs.ravel(),
        [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    )

    # --- Generate the real assets (deterministic) and load them back -----
    obj_path, tex_path = make_model.build()
    temple = objio.load_obj(obj_path)
    check("temple.obj: vertices = 120", len(temple.positions), 120)
    check("temple.obj: uvs = 360 (24 per box, one planar set per face)", len(temple.uvs), 360)
    check("temple.obj: triangles = 180", len(temple.face_vertex_indices), 180)
    check(
        "temple.obj: every face corner references a uv",
        int((temple.face_uv_indices >= 0).all()),
        1,
    )

    # The temple must sit in a known box: 6 wide/deep on the base and
    # 10.6 tall (base_1 bottom at y=0 to the tip's top).
    mins = temple.positions.min(axis=0)
    maxs = temple.positions.max(axis=0)
    check(
        "temple.obj: bounding box x [-3,3], y [0,10.6], z [-3,3]",
        np.column_stack([mins, maxs]),
        [[-3.0, 3.0], [0.0, 10.6], [-3.0, 3.0]],
    )

    # CCW-from-outside: for a convex box, the triangle normal must point
    # away from the box center. This proves the generator's winding AND
    # that the loader's fan triangulation preserved it.
    pos = temple.positions
    tri = temple.face_vertex_indices
    a = pos[tri[:, 0]]
    b = pos[tri[:, 1]]
    c = pos[tri[:, 2]]
    normals = np.cross(b - a, c - a)                    # (T, 3)
    centroids = (a + b + c) / 3.0
    outward_dot = np.empty(len(tri), dtype=np.float64)
    for box_index in range(15):
        lo, hi = box_index * 12, (box_index + 1) * 12  # 12 tris per box
        box_center = pos[box_index * 8 : (box_index + 1) * 8].mean(axis=0)
        outward_dot[lo:hi] = np.einsum("ij,ij->i", normals[lo:hi], centroids[lo:hi] - box_center)
    check(
        "temple.obj: all 180 triangles wound CCW seen from outside",
        int((outward_dot > 0.0).sum()),
        180,
    )

    # --- The texture file -------------------------------------------------
    tex_w, tex_h, tex = ppm_io.read_ppm(tex_path)
    check("relief.ppm: 256x256", [tex_w, tex_h], [256, 256])
    check("relief.ppm: three channels", int(tex.shape[2]), 3)
    check("relief.ppm: not a flat color (std > 1)", int(tex.std() > 1.0), 1)


def phase3():
    """Phase 3 checks: the rasterizer, against the plan's reference values."""
    print("=== Phase 3: rasterizer ===")

    # --- Plan row 1: barycentric weights drive linear attributes ----------
    # The angle is turned back exactly: build the point from weights
    # (0.25, 0.25, 0.5), ask for its weights, interpolate depths.
    tri = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    p = 0.25 * tri[0] + 0.25 * tri[1] + 0.5 * tri[2]  # (1, 2)
    w = raster3d.barycentric(np.array([p[0]]), np.array([p[1]]), tri)[0]
    check(
        "barycentric of the 0.25/0.25/0.5 point -> (0.25, 0.25, 0.5)",
        w,
        [0.25, 0.25, 0.5],
    )
    check(
        "barycentric (0.25,0.25,0.5) with depths 0.20/0.50/0.90 -> 0.625",
        raster3d.interp_linear(w, [0.20, 0.50, 0.90]),
        0.625,
    )
    outside = raster3d.barycentric(np.array([5.0]), np.array([5.0]), tri)[0]
    check(
        "barycentric outside point (5,5) -> (-1.5, 1.25, 1.25)",
        outside,
        [-1.5, 1.25, 1.25],
    )

    # --- Plan row 2: perspective-correct UV at the floor edge -------------
    # Screen midpoint of an edge from w = 1 (u = 0) to w = 9 (u = 1):
    # naive linear gives 0.5, the correct answer is 0.1.
    w_mid = np.array([0.5, 0.5, 0.0])
    u_vals = np.array([0.0, 1.0, 0.0])
    inv_ws = np.array([1.0, 1.0 / 9.0, 1.0])
    check(
        "floor edge midpoint: naive u -> 0.5",
        raster3d.interp_linear(w_mid, u_vals),
        0.5,
    )
    check(
        "floor edge midpoint: perspective-correct u -> 0.1",
        raster3d.interp_perspective(w_mid, u_vals, inv_ws),
        0.1,
        atol=1e-12,
    )

    # --- Plan row 3: two overlapping triangles at depth 0.3 and 0.7 -------
    # NDC z = -0.4 maps to depth (z + 1) / 2 = 0.3, +0.4 to 0.7. Near is
    # red, far is blue; whichever is drawn first, the overlap pixel must
    # keep the near color and depth.
    cfg = raster3d.RenderSettings(
        width=100,
        height=100,
        use_depth=True,
        clip_near=False,
        cull_backface=False,
        lambert=False,
        use_texture=False,
        bilinear=False,
        mipmaps=False,
    )

    def make_tri(p0, p1, p2, z, color):
        return raster3d.ScreenTriangle(
            pos=np.array([p0, p1, p2], dtype=np.float64),
            z_ndc=np.full(3, z),
            inv_w=np.ones(3),
            uv=None,
            color=np.array(color, dtype=np.float64),
            light=1.0,
        )

    near_color = [1.0, 0.25, 0.25]
    far_color = [0.25, 0.25, 1.0]
    near_tri = make_tri((10, 10), (70, 10), (10, 70), -0.4, near_color)
    far_tri = make_tri((20, 20), (80, 20), (20, 80), 0.4, far_color)

    for order in (("near", "far"), ("far", "near")):
        color = np.zeros((100, 100, 3))
        depth = np.full((100, 100), np.inf)
        for which in order:
            here = near_tri if which == "near" else far_tri
            raster3d.rasterize_triangle(color, depth, here, None, cfg)
        tag = "+".join(order)
        check(
            f"z-buffer {tag}: overlap pixel (30,30) shows near",
            color[30, 30],
            near_color,
        )
        check(
            f"z-buffer {tag}: near-only pixel (15,15) shows near",
            color[15, 15],
            near_color,
        )
        check(
            f"z-buffer {tag}: far-only pixel (30,65) shows far",
            color[30, 65],
            far_color,
        )
        check(
            f"z-buffer {tag}: near depth at the overlap stays -0.4",
            depth[30, 30],
            -0.4,
        )
        check(
            f"z-buffer {tag}: depth01 of the overlap = 0.3 (back never wins)",
            (depth[30, 30] + 1.0) / 2.0,
            0.3,
        )

    # --- Near clip: exact cuts, UVs and the safety invariant --------------
    # v0 and v2 are in front of the plane, v1 is behind it. The two cut
    # points land exactly at the midpoints of v0-v1 and v1-v2.
    clip_pts = np.array(
        [[0.0, 0.0, -1.0, 2.0], [0.0, 0.0, -3.0, 2.0], [2.0, 0.0, -1.0, 2.0]]
    )
    clip_uvs = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    pieces = raster3d.clip_near_polygon(clip_pts, clip_uvs)
    check("near clip: crossing triangle fans into 2 triangles", len(pieces), 2)
    all_pts = np.concatenate([piece_pts for piece_pts, _ in pieces])
    all_uvs = np.concatenate([piece_uv for _, piece_uv in pieces])
    expected_pts = np.unique(
        np.array([[0, 0, -1, 2], [0, 0, -2, 2], [1, 0, -2, 2], [2, 0, -1, 2]], float),
        axis=0,
    )
    check(
        "near clip: output vertices exactly {v0, cuts, v2}",
        np.unique(all_pts, axis=0),
        expected_pts,
    )
    expected_uvs = np.array([[0.0, 0.0], [0.25, 0.0], [0.75, 0.0], [1.0, 0.0]])
    check(
        "near clip: crossing UVs interpolate to 0.25 and 0.75",
        np.unique(all_uvs, axis=0),
        expected_uvs,
    )
    check(
        "near clip: every output vertex has z + w >= 0",
        int((all_pts[:, 2] + all_pts[:, 3] >= 0.0).all()),
        1,
    )

    inside_pts = np.array(
        [[0.0, 0.0, -1.0, 2.0], [2.0, 0.0, -1.0, 2.0], [0.0, 2.0, -1.0, 2.0]]
    )
    check(
        "near clip: fully inside -> 1 unchanged triangle",
        len(raster3d.clip_near_polygon(inside_pts)),
        1,
    )
    outside_pts = inside_pts.copy()
    outside_pts[:, 2] = -3.0
    check(
        "near clip: fully behind -> 0 triangles",
        len(raster3d.clip_near_polygon(outside_pts)),
        0,
    )

    # --- Back-face predicate ----------------------------------------------
    check(
        "cull: CCW NDC triangle is front-facing",
        int(raster3d.is_front_facing(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))),
        1,
    )
    check(
        "cull: reversed winding is back-facing",
        int(raster3d.is_front_facing(np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]))),
        0,
    )

    # --- Mip pyramid: sizes and exact box means ---------------------------
    small = np.arange(16.0).reshape(4, 4)
    pyramid = raster3d.build_mip_pyramid(small, levels=3)
    check(
        "mip pyramid: 4x4 with 3 levels -> sizes 4/2/1",
        np.array([lvl.shape for lvl in pyramid]),
        [[4, 4], [2, 2], [1, 1]],
    )
    check(
        "mip pyramid: level 1 = the four 2x2 means",
        pyramid[1],
        [[2.5, 4.5], [10.5, 12.5]],
    )
    check("mip pyramid: level 2 = the mean of level 1", pyramid[2], [[7.5]])

    tex_path = make_model.DATA_DIR / "relief.ppm"
    if not tex_path.is_file():
        make_model.build()
    texture = raster3d.Texture(ppm_io.read_ppm(tex_path)[2])
    check(
        "texture pyramid: 256/128/64/32",
        np.array([lvl.shape[:2] for lvl in texture.levels]),
        [[256, 256], [128, 128], [64, 64], [32, 32]],
    )
    level0, level1 = texture.levels[0], texture.levels[1]
    check(
        "texture pyramid: level 1 equals the linear 2x2 mean of level 0",
        level1[3, 7],
        level0[6:8, 14:16].mean(axis=(0, 1)),
        atol=1e-12,
    )

    # --- sRGB helpers ------------------------------------------------------
    check(
        "srgb_to_linear(0.5) -> 0.214041 (power-law branch)",
        raster3d.srgb_to_linear(0.5),
        0.214041,
        atol=1e-5,
    )
    check(
        "srgb_to_linear(0.02) = 0.02 / 12.92 (linear branch)",
        raster3d.srgb_to_linear(0.02),
        0.02 / 12.92,
        atol=1e-15,
    )
    check(
        "sRGB round trip of 0.5 stays 0.5",
        raster3d.linear_to_srgb(raster3d.srgb_to_linear(0.5)),
        0.5,
        atol=1e-12,
    )

    # --- Full render sanity at a small size -------------------------------
    temple = objio.load_obj(make_model.DATA_DIR / "temple.obj")
    settings_small = raster3d.RenderSettings(width=160, height=120)
    result = raster3d.render_scene(temple, texture, settings_small)

    bg_linear = raster3d.srgb_to_linear(np.array(raster3d.SKY_SRGB))
    covered = np.any(np.abs(result.color_linear - bg_linear) > 1e-9, axis=2)
    coverage = float(covered.mean())
    print(f"  full render: coverage = {coverage:.4f} of the 160x120 frame")
    check(
        "full render: coverage inside the sanity band (0.03, 0.7)",
        int(0.03 < coverage < 0.7),
        1,
    )
    check(
        "full render: corner pixel is background sky",
        result.color_linear[0, 0],
        bg_linear,
    )
    check(
        "full render: corner pixel depth is background (1.0)",
        result.depth01[0, 0],
        1.0,
    )
    check("full render: 180 triangles submitted", result.stats["submitted"], 180)
    check(
        "full render: 0 near-clipped (camera is far from the model)",
        result.stats["clipped"],
        0,
    )
    check(
        "full render: no unprojectable triangles",
        result.stats["unprojectable"],
        0,
    )

    # Independent back-face count: a planar triangle is front-facing
    # exactly when its outward normal points toward the eye. If this
    # count matches the rasterizer's culled counter, the NDC area sign
    # test and the geometric test agree on all 180 triangles.
    a = temple.positions[temple.face_vertex_indices[:, 0]]
    b = temple.positions[temple.face_vertex_indices[:, 1]]
    c = temple.positions[temple.face_vertex_indices[:, 2]]
    normals = np.cross(b - a, c - a)
    eye = np.array(raster3d.Camera().eye)
    facing = np.einsum("ij,ij->i", normals, eye[None, :] - a)
    expected_culled = int((facing <= 0.0).sum())
    print(f"  independent back-face count = {expected_culled} of 180")
    check(
        "full render: culled matches the independent back-face count",
        result.stats["culled"],
        expected_culled,
    )

    # Depth at the front body face: project world (0, 3.4, 1.6), exactly
    # on the front wall of the body box, and require the depth image at
    # the pixel containing it to match the analytic depth. The tolerance
    # covers the half-pixel offset between the projected point and the
    # pixel center (about 4e-4 of depth per pixel at this resolution).
    cam = raster3d.Camera()
    p_mat = perspective(cam.fovy_deg, 160.0 / 120.0, cam.near, cam.far)
    v_mat = look_at(cam.eye, cam.target, cam.up)
    _, ndc_pt = project(p_mat, v_mat, np.eye(4), np.array([[0.0, 3.4, 1.6]]))
    scr_pt = viewport(ndc_pt, 160, 120)[0]
    px, py = int(round(scr_pt[0])), int(round(scr_pt[1]))
    expected_depth = float((ndc_pt[0, 2] + 1.0) * 0.5)
    check(
        "full render: depth at the body-front pixel matches projection",
        result.depth01[py, px],
        expected_depth,
        atol=2e-3,
    )

    # Determinism: identical inputs must produce a byte-identical frame.
    result2 = raster3d.render_scene(temple, texture, settings_small)
    check(
        "full render: deterministic (two runs byte-identical)",
        int(np.array_equal(result.color_srgb, result2.color_srgb)),
        1,
    )


def main():
    """Run every phase section and print a summary; exit code 0 on success."""
    print("stupa-engine -- numeric checks")
    phase1()
    phase2()
    phase3()
    # Later phases append their own section here as they are built.

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} of {_TOTAL} checks FAILED:")
        for name in _FAILURES:
            print(f"  - {name}")
        return 1
    print(f"all {_TOTAL} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
