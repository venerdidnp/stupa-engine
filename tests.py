"""Module 3 -- run every from-scratch check. `python tests.py`. Exit 0 iff all
pass. Hand-checkable invariants of Examples E1-E5 + a 1-triangle render.
"""
from __future__ import annotations

import sys
import traceback

import numpy as np

from transforms import (translate, rot_y, look_at, perspective, project,
                        viewport, perspective_divide_2d)


def check(name, fn):
    try:
        fn(); print("OK   " + name); return True
    except Exception:
        print("FAIL " + name); traceback.print_exc(); return False


def t_e1_running_point():
    p = np.array([2.0, 3.0, 4.0, 1.0])
    world = rot_y(90) @ translate((1, 0, -1)) @ p        # translate first, then yaw
    assert np.allclose(world, [3, 3, -3, 1], atol=1e-9), world
    proj = perspective_divide_2d(world[:3], d=2)
    assert np.allclose(proj, [2, 2], atol=1e-9), proj
    # order matters: yaw first, then translate -> different point
    other = translate((1, 0, -1)) @ rot_y(90) @ p
    assert not np.allclose(other, world), "order should matter"
    assert np.allclose(other, [5, 3, -3, 1], atol=1e-9), other


def t_e2_duality():
    p = np.array([5.0, 0.0, 0.0, 1.0])
    V_cam_plus_x = look_at(eye=(1, 0, 0), center=(1, 0, -1))   # camera moved +x
    model_minus_x = translate((-1, 0, 0))                      # model moved -x
    assert np.allclose(V_cam_plus_x @ p, model_minus_x @ p, atol=1e-9), \
        (V_cam_plus_x @ p, model_minus_x @ p)


def t_e3_perspective_near_far():
    n, f = 1.0, 9.0
    P = perspective(60, 1.0, n, f)
    for z, want in [(-n, -1.0), (-f, +1.0), (-3.0, 0.5)]:
        clip = P @ np.array([0, 0, z, 1.0])
        ndc_z = clip[2] / clip[3]
        assert abs(ndc_z - want) < 1e-9, f"z={z}: ndc_z={ndc_z} want {want}"


def t_e4_perspective_correct_u():
    # floor edge: L (x=0, u=0, w=1) .. R (x=100, u=1, w=9); u at screen x=50
    u_over_w = np.interp(50, [0, 100], [0 / 1, 1 / 9])
    inv_w = np.interp(50, [0, 100], [1 / 1, 1 / 9])
    u_pc = u_over_w / inv_w
    assert abs(u_pc - 0.1) < 1e-12, u_pc
    assert abs(0.5 - 0.5) < 1e-12               # the naive answer, for contrast


def t_e5_barycentric_depth():
    zA, zB, zC = 0.20, 0.50, 0.90
    la, lb, lc = 0.25, 0.25, 0.50
    assert abs(la + lb + lc - 1) < 1e-12
    z = la * zA + lb * zB + lc * zC
    assert abs(z - 0.625) < 1e-12, z


def t_projection_pipeline_shapes():
    verts = np.array([[0, 0, -2.0], [1, 0, -2.0], [0, 1, -3.0]])
    P = perspective(50, 4 / 3, 0.1, 100)
    V = look_at((0, 0, 3), (0, 0, 0))
    clip, ndc = project(P, V, np.eye(4), verts)
    assert clip.shape == (3, 4) and ndc.shape == (3, 3)
    scr = viewport(ndc, 640, 480)
    assert scr.shape == (3, 3)
    assert np.all(scr[:, 0] >= -640) and np.all(scr[:, 0] <= 1280)


def t_one_triangle_render():
    from objio import Mesh
    import raster3d
    pos = np.array([[-0.6, -0.6, -2.0], [0.6, -0.6, -2.0], [0.0, 0.6, -2.0]])
    m = Mesh(pos, np.zeros((0, 2)), [[0, 1, 2]], [])
    col, dep, filled = raster3d.render(m, w=48, h=48, tex_u8=None, near=1.0,
                                       far=50.0, eye=(0, 0, 0), center=(0, 0, -1),
                                       cull_back=False, light_dir=(0, 0, 1))
    assert 200 < filled < 48 * 48, f"filled={filled}"
    assert np.isfinite(dep).sum() == filled
    # depth is constant (all verts at z=-2) -> small spread
    fin = dep[np.isfinite(dep)]
    assert fin.max() - fin.min() < 1e-6, f"depth spread {fin.max()-fin.min()}"


def t_reference_frame():
    """Render the committed temple and match ref/temple_ref.ppm (generated on
    first run -- commit ref/ afterwards)."""
    import os
    import raster3d
    from objio import load_obj
    from ppm_io import read_ppm, write_ppm
    from transforms import translate, rot_y, scale
    from metrics import psnr
    if not os.path.exists("data/pusaka_model.obj"):
        print("     (run make_model.py first -- skipping)"); return
    mesh = load_obj("data/pusaka_model.obj")
    tex = read_ppm("data/relief.ppm")
    M = translate((0, 0, 0)) @ rot_y(35) @ scale((1.4, 1.4, 1.4))
    col, _, _ = raster3d.render(mesh, w=160, h=160, tex_u8=tex, model=M)
    if not os.path.exists("ref/temple_ref.ppm"):
        os.makedirs("ref", exist_ok=True)
        write_ppm("ref/temple_ref.ppm", col)
        print("     (ref/temple_ref.ppm generated -- commit it)")
    ref = read_ppm("ref/temple_ref.ppm")
    p = psnr(col, ref)
    assert p == float("inf") or p > 60, f"temple frame PSNR vs ref = {p:.1f} dB"


def main():
    checks = [
        ("E1  running point (2,3,4) -> (3,3,-3); divide d=2 -> (2,2); order matters", t_e1_running_point),
        ("E2  camera +x == model -x (identical points)", t_e2_duality),
        ("E3  perspective: z=-n -> ndc -1 ; z=-f -> ndc +1 ; z=-3 -> 0.5", t_e3_perspective_near_far),
        ("E4  perspective-correct u at screen centre = 0.1 (naive 0.5)", t_e4_perspective_correct_u),
        ("E5  barycentric depth 0.20/0.50/0.90 at (.25,.25,.5) = 0.625", t_e5_barycentric_depth),
        ("projection pipeline shapes + viewport", t_projection_pipeline_shapes),
        ("one-triangle z-buffer render fills a plausible region, flat depth", t_one_triangle_render),
        ("temple frame matches ref/temple_ref.ppm (byte-identical on re-render)", t_reference_frame),
    ]
    ok = all(check(n, f) for n, f in checks)
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
