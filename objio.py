"""Module 3 -- a minimal Wavefront OBJ loader, from scratch (no library).

Supports:  v x y z | vt u v | f a/b a/b a/b (also a//c and a).
Triangulates polygons with a fan. Returns a small mesh object.

UV origin: OBJ / OpenGL put (0,0) at bottom-left. Image files are top-left.
load_obj(flip_v=True) flips v once, here, so the rest of the code never has
to think about it (Pitfall: UV origin flip).
"""
from __future__ import annotations

import numpy as np


class Mesh:
    def __init__(self, positions, uvs, faces, face_uvs):
        self.positions = np.asarray(positions, float)      # (V,3)
        self.uvs = np.asarray(uvs, float) if len(uvs) else np.zeros((0, 2))
        self.faces = np.asarray(faces, np.int64)           # (F,3) -> position idx
        self.face_uvs = np.asarray(face_uvs, np.int64) if len(face_uvs) else None

    @property
    def n_tris(self):
        return len(self.faces)

    def triangle_positions(self):
        return self.positions[self.faces]                  # (F,3,3)

    def triangle_uvs(self):
        if self.face_uvs is None:
            return np.zeros((len(self.faces), 3, 2))
        return self.uvs[self.face_uvs]                     # (F,3,2)


def _parse_index(tok, n_pos, n_uv):
    """'a/b/c' -> (pos_idx, uv_idx) as 0-based; negatives are relative."""
    parts = (tok.split("/") + ["", ""])[:3]
    pi = int(parts[0])
    pi = pi - 1 if pi > 0 else n_pos + pi
    ui = None
    if parts[1] != "":
        u = int(parts[1])
        ui = u - 1 if u > 0 else n_uv + u
    return pi, ui


def load_obj(path, flip_v=True):
    positions, uvs, faces, face_uvs = [], [], [], []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                positions.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                u, v = (float(x) for x in line.split()[1:3])
                uvs.append([u, 1.0 - v if flip_v else v])
            elif line.startswith("f "):
                toks = line.split()[1:]
                idx = [_parse_index(t, len(positions), len(uvs)) for t in toks]
                for k in range(1, len(idx) - 1):            # fan triangulation
                    faces.append([idx[0][0], idx[k][0], idx[k + 1][0]])
                    if idx[0][1] is not None:
                        face_uvs.append([idx[0][1], idx[k][1], idx[k + 1][1]])
    return Mesh(positions, uvs, faces, face_uvs if face_uvs else [])


def save_obj(path, positions, uvs, faces, face_uvs=None):
    with open(path, "w") as fh:
        fh.write("# Module 3 -- deterministic Pusaka3D temple (make_model.py)\n")
        for p in positions:
            fh.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
        for t in uvs:
            fh.write(f"vt {t[0]:.5f} {t[1]:.5f}\n")
        for i, f in enumerate(faces):
            if face_uvs is not None:
                a, b, c = f; ua, ub, uc = face_uvs[i]
                fh.write(f"f {a+1}/{ua+1} {b+1}/{ub+1} {c+1}/{uc+1}\n")
            else:
                fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
