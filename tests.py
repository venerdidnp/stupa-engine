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


def main():
    """Run every phase section and print a summary; exit code 0 on success."""
    print("stupa-engine -- numeric checks")
    phase1()
    phase2()
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
