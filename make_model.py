"""Module 3 -- deterministic Pusaka3D temple: a stacked-tier candi.

No RNG, no binary assets in Git. Writes:
  data/pusaka_model.obj   180 triangles (15 stacked boxes, planar-projected UVs)
  data/relief.ppm         256x256 procedural sandstone texture (Module 1 style)
  data/mymodel.obj        312 triangles (26 boxes: student custom candi with corner stupas & stairs)

Run:  python make_model.py [--out data]
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np

from objio import save_obj
from ppm_io import write_ppm
from srgb import linear_to_srgb, to_u8


def box(cx, cz, half_x, y0, y1, half_z=None):
    """Axis-aligned box, centre (cx,*,cz), horizontal half-size (half_x, half_z),
    vertical span y0..y1. Returns (8 positions, 12 triangles (as local idx),
    per-face-vertex uvs). CCW-front (outward-facing)."""
    if half_z is None:
        half_z = half_x
    x0, x1 = cx - half_x, cx + half_x
    z0, z1 = cz - half_z, cz + half_z
    P = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1],   # bottom 0..3
        [x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1],   # top    4..7
    ])
    # each face: 2 triangles, wound CCW seen from outside, i.e. the
    # right-handed normal (v1-v0)x(v2-v0) points AWAY from the box centre
    # (tests.py::t_winding_outward counts this).
    faces = [
        (4, 6, 5), (4, 7, 6),        # +y top
        (0, 1, 2), (0, 2, 3),        # -y bottom
        (0, 5, 1), (0, 4, 5),        # -z front
        (2, 7, 3), (2, 6, 7),        # +z back
        (3, 4, 0), (3, 7, 4),        # -x left
        (1, 6, 2), (1, 5, 6),        # +x right
    ]
    # simple planar UVs: project each vertex onto x-y, scaled into [0,1] per box
    uv = np.c_[(P[:, 0] - x0) / max(2 * half_x, 1e-6),
               (P[:, 1] - y0) / max(y1 - y0, 1e-6)]
    face_uv = [(a, b, c) for (a, b, c) in faces]
    return P, faces, uv, face_uv


def build_temple():
    """Official 15-box reference temple (180 triangles, 120 vertices)."""
    positions, faces, uvs, face_uvs = [], [], [], []
    voff = 0
    # 7 stacked tiers, shrinking; a finial box on top
    tiers = [(0.0, 0.0, 1.00, 0.00, 0.35),
             (0.0, 0.0, 0.82, 0.35, 0.62),
             (0.0, 0.0, 0.66, 0.62, 0.85),
             (0.0, 0.0, 0.50, 0.85, 1.05),
             (0.0, 0.0, 0.34, 1.05, 1.22),
             (0.0, 0.0, 0.20, 1.22, 1.36),
             (0.0, 0.0, 0.07, 1.36, 1.52)]
    # four corner shrines (small boxes)
    for (sx, sz) in [(-0.8, -0.8), (0.8, -0.8), (-0.8, 0.8), (0.8, 0.8)]:
        tiers.append((sx, sz, 0.16, 0.0, 0.30))
        tiers.append((sx, sz, 0.10, 0.30, 0.44))
    for (cx, cz, half, y0, y1) in tiers:
        P, F, uv, fuv = box(cx, cz, half, y0, y1)
        for (a, b, c) in F:
            faces.append((a + voff, b + voff, c + voff))
        for (a, b, c) in fuv:
            face_uvs.append((a + voff, b + voff, c + voff))
        positions.extend(P.tolist())
        uvs.extend(uv.tolist())
        voff += len(P)
    return np.array(positions), np.array(uvs), faces, face_uvs


def build_custom_model():
    """Student custom candi model (26 boxes = 312 triangles, 208 vertices).
    Fulfills Lab 1 requirement: custom model >= 200 triangles."""
    positions, faces, uvs, face_uvs = [], [], [], []
    voff = 0

    # 15 central stacked boxes (base terraces, body, roof, spire)
    boxes_def = [
        # (cx, cz, hx, y0, y1, hz)
        (0.0, 0.0, 1.20, 0.00, 0.25, 1.20),   # base_1
        (0.0, 0.0, 1.05, 0.25, 0.50, 1.05),   # base_2
        (0.0, 0.0, 0.90, 0.50, 0.75, 0.90),   # base_3
        (0.0, 0.0, 0.70, 0.75, 1.20, 0.70),   # body
        (0.0, 0.0, 0.78, 1.20, 1.30, 0.78),   # cornice
        (0.0, 0.0, 0.65, 1.30, 1.45, 0.65),   # roof_1
        (0.0, 0.0, 0.55, 1.45, 1.60, 0.55),   # roof_2
        (0.0, 0.0, 0.45, 1.60, 1.73, 0.45),   # roof_3
        (0.0, 0.0, 0.35, 1.73, 1.85, 0.35),   # roof_4
        (0.0, 0.0, 0.27, 1.85, 1.95, 0.27),   # roof_5
        (0.0, 0.0, 0.20, 1.95, 2.03, 0.20),   # roof_6
        (0.0, 0.0, 0.14, 2.03, 2.10, 0.14),   # roof_7
        (0.0, 0.0, 0.10, 2.10, 2.18, 0.10),   # spire_base
        (0.0, 0.0, 0.06, 2.18, 2.26, 0.06),   # spire
        (0.0, 0.0, 0.03, 2.26, 2.32, 0.03),   # tip
    ]

    # 4 corner stupas on base_1 (outer terrace corners)
    for (sx, sz) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]:
        boxes_def.append((sx, sz, 0.12, 0.25, 0.45, 0.12))

    # 4 corner stupas on base_2 (middle terrace corners)
    for (sx, sz) in [(-0.85, -0.85), (0.85, -0.85), (-0.85, 0.85), (0.85, 0.85)]:
        boxes_def.append((sx, sz, 0.09, 0.50, 0.68, 0.09))

    # 3 front steps leading up on +z face
    boxes_def.append((0.0, 1.30, 0.25, 0.00, 0.08, 0.10))
    boxes_def.append((0.0, 1.20, 0.22, 0.08, 0.16, 0.10))
    boxes_def.append((0.0, 1.10, 0.19, 0.16, 0.25, 0.10))

    for item in boxes_def:
        cx, cz, hx, y0, y1, hz = item
        P, F, uv, fuv = box(cx, cz, hx, y0, y1, hz)
        for (a, b, c) in F:
            faces.append((a + voff, b + voff, c + voff))
        for (a, b, c) in fuv:
            face_uvs.append((a + voff, b + voff, c + voff))
        positions.extend(P.tolist())
        uvs.extend(uv.tolist())
        voff += len(P)
    return np.array(positions), np.array(uvs), faces, face_uvs


def relief_texture(n=256):
    """Procedural sandstone relief -- matches Module 1 and reference frame."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    fx = 0.22
    raw = np.sin(2 * np.pi * fx * (x + 0.5 * y))
    ridges = 0.5 + 0.40 * np.tanh(2.5 * raw)
    slow = 0.10 * np.sin(2 * np.pi * 0.02 * x) * np.cos(2 * np.pi * 0.015 * y)
    lum = np.clip(ridges + slow, 0.05, 0.98)
    lum[int(0.33 * n):int(0.37 * n), :] *= 0.6      # a carved band
    tint = (1.00, 0.82, 0.60)
    lin = np.dstack([lum * tint[0], lum * tint[1], lum * tint[2]])
    return to_u8(linear_to_srgb(np.clip(lin, 0, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1. Official reference model (180 tris)
    pos, uvs, faces, fuvs = build_temple()
    save_obj(os.path.join(args.out, "pusaka_model.obj"), pos, uvs, faces, fuvs)
    print(f"wrote {os.path.join(args.out, 'pusaka_model.obj')}  ({len(pos)} verts, {len(faces)} triangles)")

    # 2. Student custom model for Lab 1 (312 tris >= 200 tris)
    c_pos, c_uvs, c_faces, c_fuvs = build_custom_model()
    save_obj(os.path.join(args.out, "mymodel.obj"), c_pos, c_uvs, c_faces, c_fuvs)
    save_obj(os.path.join(args.out, "temple.obj"), c_pos, c_uvs, c_faces, c_fuvs)
    print(f"wrote {os.path.join(args.out, 'mymodel.obj')}  ({len(c_pos)} verts, {len(c_faces)} triangles - Lab 1 custom model)")

    # 3. Texture
    tex = relief_texture()
    tex_path = os.path.join(args.out, "relief.ppm")
    write_ppm(tex_path, tex)
    print(f"wrote {tex_path}  (256x256, {os.path.getsize(tex_path)} bytes)")

    # Copy to web/data/ for standalone web viewer
    web_data = os.path.join("web", "data")
    os.makedirs(web_data, exist_ok=True)
    for fname in ["pusaka_model.obj", "mymodel.obj", "temple.obj", "relief.ppm"]:
        shutil.copy2(os.path.join(args.out, fname), os.path.join(web_data, fname))


if __name__ == "__main__":
    main()
