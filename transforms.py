"""Module 3 -- the transform stack, from scratch (NumPy).

CONVENTIONS (pinned -- never mix another in):
  * column vectors, pre-multiply:  p' = M @ p
  * a chain "first A, then B" is  p' = B @ A @ p   (right to left)
  * right-handed axes; the camera looks down -z
  * NDC is [-1, 1]^3, near plane at z = -1, far plane at z = +1 (OpenGL style)
  * viewport: NDC x in [-1,1] -> pixel columns; NDC y is FLIPPED (screen y down)

Reproduces:
  E1  (2,3,4) --T(1,0,-1)--> (3,3,3) --Ry(90)--> (3,3,-3);  divide d=2 -> (2,2)
  E2  camera +x by 1  ==  model -x by 1   (identical points)
  E3  perspective(.,.,n,f): z=-n -> ndc_z -1 ;  z=-f -> ndc_z +1
"""
from __future__ import annotations

import numpy as np


def translate(t):
    M = np.eye(4)
    M[:3, 3] = t
    return M


def scale(s):
    s = np.asarray(s, dtype=float)
    return np.diag([s[0], s[1], s[2], 1.0])


def rot_x(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def rot_y(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])


def rot_z(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def perspective(fovy_deg, aspect, near, far):
    """OpenGL-style perspective. Maps eye z=-near -> ndc z=-1, z=-far -> +1."""
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2.0)
    M = np.zeros((4, 4))
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M


def look_at(eye, center, up=(0, 1, 0)):
    """Right-handed view matrix. Camera at `eye` looking at `center`."""
    eye = np.asarray(eye, float); center = np.asarray(center, float)
    up = np.asarray(up, float)
    f = center - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    V = np.eye(4)
    V[0, :3] = s
    V[1, :3] = u
    V[2, :3] = -f
    V[0, 3] = -s @ eye
    V[1, 3] = -u @ eye
    V[2, 3] = f @ eye
    return V


def to_homogeneous(verts):
    verts = np.asarray(verts, float)
    return np.c_[verts, np.ones(len(verts))]


def project(P, V, M, verts):
    """verts (N,3) -> (clip (N,4), ndc (N,3)). ndc = clip / clip.w."""
    p = to_homogeneous(verts)
    clip = (P @ V @ M @ p.T).T
    w = clip[:, 3:4]
    ndc = clip[:, :3] / w
    return clip, ndc


def viewport(ndc, w, h):
    """NDC [-1,1] -> screen pixels, y DOWN. Returns (N,3): x, y, depth(ndc z)."""
    ndc = np.asarray(ndc, float)
    x = (ndc[:, 0] * 0.5 + 0.5) * w
    y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * h
    return np.c_[x, y, ndc[:, 2]]


def perspective_divide_2d(p3, d):
    """The bare similar-triangles projection used in Example E1: a point in
    front of a pinhole at the origin (camera looks down -z, so p3.z < 0)
    projects to (d*x/|z|, d*y/|z|) on an image plane at distance d."""
    x, y, z = p3
    return np.array([d * x / abs(z), d * y / abs(z)])
