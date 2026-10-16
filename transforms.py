"""
transforms.py -- all 4x4 matrix math for the renderer.

=====================================================================
PINNED CONVENTIONS (never mix these; the whole project depends on them)
=====================================================================

1. Column vectors.
   A point is a 4x1 column [x, y, z, 1]^T and every transform is
   applied on the left:  p' = M @ p.  A chain "first A, then B" is
   therefore written  p' = B @ A @ p  (the right-most matrix acts
   first).

2. Right-handed coordinates, +y up.
   Rotations follow the right-hand rule around +x, +y, +z. The camera
   looks down its own -z axis ("eye space"), so visible points have
   z_eye < 0.

3. NDC (normalized device coordinates) is the cube [-1, +1]^3.
   The near plane maps to NDC z = -1 and the far plane to NDC z = +1
   (classic OpenGL range). The w-divide happens AFTER clipping, never
   before: points behind the camera produce w <= 0 and would fold onto
   the screen if divided early.

4. Front faces are counter-clockwise (CCW) seen from outside; the
   rasterizer culls back faces.

5. UV origin is bottom-left. The single v-flip happens once, in the
   OBJ loader (objio.py); nothing in this file touches texture coords.

6. All angle parameters end in `_deg`. They are converted to radians
   here, at the boundary, and degrees are never passed around inside
   the math.

7. The full model-to-clip chain is  clip = P @ V @ M @ p
   (model first, then view, then projection).
"""

import numpy as np


def translate(tx, ty, tz):
    """Build a 4x4 translation matrix.

    What: returns T such that T @ [x, y, z, 1]^T = [x+tx, y+ty, z+tz, 1]^T.
    Inputs: tx, ty, tz -- offsets in world units.
    Output: (4, 4) float64 array.

    Why this layout: with column vectors the translation lives in the
    last COLUMN (rows 0..2 of column 3). The last row stays (0,0,0,1)
    so the homogeneous w stays 1 and points stay points.
    """
    return np.array(
        [
            [1.0, 0.0, 0.0, float(tx)],
            [0.0, 1.0, 0.0, float(ty)],
            [0.0, 0.0, 1.0, float(tz)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def scale(sx, sy, sz):
    """Build a 4x4 scale matrix.

    What: returns S such that S @ [x, y, z, 1]^T = [sx*x, sy*y, sz*z, 1]^T.
    Inputs: sx, sy, sz -- scale factors along each axis.
    Output: (4, 4) float64 array.

    Why the bottom-right entry is 1.0: w must stay 1.0, otherwise every
    later divide by w would scale the whole point a second time.
    """
    return np.array(
        [
            [float(sx), 0.0, 0.0, 0.0],
            [0.0, float(sy), 0.0, 0.0],
            [0.0, 0.0, float(sz), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rot_x(deg):
    """Rotation about the +x axis by `deg` degrees (right-hand rule).

    What: returns R such that a point rotates counter-clockwise around +x
          when looking from +x toward the origin; +y leans toward +z.
    Inputs: deg -- angle in degrees.
    Output: (4, 4) float64 array.

    Why these signs: with column vectors the standard right-handed
    matrix is [[1,0,0],[0,c,-s],[0,s,c]]. Check: rot_x(90) maps
    (0,1,0) -> (0,0,1), i.e. c=0, s=1 gives y'=0, z'=1.
    """
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rot_y(deg):
    """Rotation about the +y axis by `deg` degrees (right-hand rule).

    What: returns R such that a point rotates counter-clockwise around +y
          when looking from +y toward the origin; +z leans toward +x.
    Inputs: deg -- angle in degrees.
    Output: (4, 4) float64 array.

    Why these signs: the standard right-handed column-vector matrix is
    [[c,0,s],[0,1,0],[-s,0,c]]. Check against the plan's reference:
    rot_y(90) @ (3,3,3) = (c*3 + s*3, 3, -s*3 + c*3) = (3, 3, -3),
    which is exactly the expected value for the first check.
    """
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rot_z(deg):
    """Rotation about the +z axis by `deg` degrees (right-hand rule).

    What: returns R such that a point rotates counter-clockwise around +z
          when looking from +z toward the origin; +x leans toward +y.
    Inputs: deg -- angle in degrees.
    Output: (4, 4) float64 array.

    Why these signs: the standard right-handed column-vector matrix is
    [[c,-s,0],[s,c,0],[0,0,1]]. Check: rot_z(90) maps (1,0,0) ->
    (0,1,0).
    """
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def perspective(fovy_deg, aspect, near, far):
    """Build the perspective projection matrix P (eye space -> clip space).

    Inputs:
      fovy_deg -- vertical field of view in degrees (0 < fovy < 180).
      aspect   -- viewport width / height, > 0.
      near     -- distance to the near plane, in world units, > 0.
      far      -- distance to the far plane, > near.
    Output: (4, 4) float64 array.

    How it works:
      f = 1 / tan(fovy / 2) is the focal length expressed in NDC units
      (f = 1 gives a 90 degree vertical field of view).

      x_clip = (f / aspect) * x_eye
      y_clip = f * y_eye
      z_clip = A * z_eye + B
      w_clip = -z_eye          <- the source of perspective foreshortening

      The w-divide happens later (in the rasterizer, after clipping), so
      for visible geometry:
        z_ndc = (A * z_eye + B) / (-z_eye)

      Two constraints pin A and B:
        near plane must map to z_ndc = -1:  (A*(-n) + B) / n   = -1
        far  plane must map to z_ndc = +1:  (A*(-f) + B) / f   = +1
      Solving the 2x2 system:
        A = -(far + near) / (far - near)
        B = -(2 * far * near) / (far - near)
      e.g. n = 1, f = 9 gives A = -1.25, B = -2.25, which is one of the
      plan's reference values.

    Why w_clip = -z_eye: dividing by -z_eye is exactly the pinhole
    projection x_img = x / (-z) -- farther points (more negative z)
    shrink on screen. Because -z_eye > 0 only for points in front of
    the camera, geometry behind the camera must be clipped BEFORE this
    divide or it would mirror into view.
    """
    if not 0.0 < fovy_deg < 180.0:
        raise ValueError(f"fovy_deg must be in (0, 180), got {fovy_deg}")
    if aspect <= 0.0:
        raise ValueError(f"aspect must be > 0, got {aspect}")
    if not 0.0 < near < far:
        raise ValueError(f"need 0 < near < far, got near={near}, far={far}")

    # Focal length in NDC units: the half-height of the NDC cube (1.0)
    # divided by tan of the half field of view.
    f = 1.0 / np.tan(np.radians(fovy_deg) * 0.5)

    # Depth-mapping coefficients, derived above from the near/far pins.
    A = -(far + near) / (far - near)
    B = -(2.0 * far * near) / (far - near)

    return np.array(
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, A, B],
            [0.0, 0.0, -1.0, 0.0],  # w_clip = -z_eye
        ],
        dtype=np.float64,
    )


def look_at(eye, target, up):
    """Build the view matrix V (world space -> eye space).

    Inputs:
      eye    -- (3,) camera position in world space.
      target -- (3,) point the camera looks at.
      up     -- (3,) approximate up direction (need not be normalized).
    Output: (4, 4) float64 array.

    How it works (inverse of the camera pose):
      The camera pose maps camera-space points to world space:
          X_world = R @ X_cam + c
      where c = eye and the COLUMNS of R are the camera's own axes,
      expressed in world coordinates:
          z_cam = normalize(eye - target)   (backward: camera looks -z)
          x_cam = normalize(up x z_cam)     (right)
          y_cam = z_cam x x_cam             (up, re-orthogonalized)

      The view matrix is the inverse of that rigid pose. For a pure
      rotation + translation the inverse is
          V = [ R^T   -R^T @ c ]
              [ 0      1       ]
      R^T appears (not R) because the inverse of a rotation is its
      transpose; -R^T @ c moves the eye to the origin. This is the
      "R^T and -R^T c" form used everywhere in this project.

    Why x = up x z and then y = z x x: the cross products guarantee
    x_cam and y_cam are perpendicular to the view direction and to
    each other, so R stays a proper rotation (no shear) even if `up`
    is not exactly perpendicular to the view direction.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)

    back = eye - target
    if np.linalg.norm(back) < 1e-12:
        raise ValueError("look_at: eye and target are the same point")
    z_cam = back / np.linalg.norm(back)

    # If `up` is parallel to the view direction the cross product is
    # zero and any "right" direction is equally valid -- but silently
    # returning a broken matrix is worse than failing loudly.
    x_cam = np.cross(up, z_cam)
    if np.linalg.norm(x_cam) < 1e-12:
        raise ValueError("look_at: up vector is parallel to the view direction")
    x_cam = x_cam / np.linalg.norm(x_cam)

    y_cam = np.cross(z_cam, x_cam)  # already unit length (z, x are orthonormal)

    # R has the camera axes as columns; V[:3,:3] = R^T puts them in rows.
    V = np.eye(4, dtype=np.float64)
    V[0, :3] = x_cam
    V[1, :3] = y_cam
    V[2, :3] = z_cam
    V[:3, 3] = -(V[:3, :3] @ eye)
    return V


def viewport(ndc, w, h):
    """Map NDC points to screen pixels, flipping y, and to a depth image value.

    Inputs:
      ndc -- (N, 3) array; x, y in [-1, 1], z in [-1, 1].
      w   -- target width in pixels.
      h   -- target height in pixels.
    Output: (N, 3) array [x_pix, y_pix, depth] where
      x_pix = (x_ndc + 1) * 0.5 * w   (NDC -1 -> pixel 0, left edge)
      y_pix = (1 - y_ndc) * 0.5 * h   (y flipped: screen y grows DOWN,
                                       NDC +1 is the top of the screen)
      depth = (z_ndc + 1) * 0.5       (in [0, 1], for the depth image)

    Why flip y only: NDC y points up (math convention) while image rows
    grow downward (file convention). x needs no flip because both point
    right. The coordinate here is continuous (e.g. 588.95); rounding to
    whole pixels happens where pixels are actually written.

    Check against the plan's worked example: ndc (0.472, 0.472, 0.855)
    at 800x600 gives (588.95, 158.29) -> pixel (589, 158), and depth
    (0.855 + 1) * 0.5 = 0.9275 -> 0.928.
    """
    ndc = np.atleast_2d(np.asarray(ndc, dtype=np.float64))
    if ndc.shape[1] != 3:
        raise ValueError(f"ndc must have shape (N, 3), got {ndc.shape}")

    x_pix = (ndc[:, 0] + 1.0) * 0.5 * w
    y_pix = (1.0 - ndc[:, 1]) * 0.5 * h
    depth = (ndc[:, 2] + 1.0) * 0.5
    return np.stack([x_pix, y_pix, depth], axis=1)


def project(P, V, M, verts):
    """Transform object-space vertices to clip space and NDC.

    Inputs:
      P     -- (4, 4) projection matrix (eye -> clip).
      V     -- (4, 4) view matrix (world -> eye).
      M     -- (4, 4) model matrix (object -> world).
      verts -- (N, 3) or (N, 4) vertices; if (N, 3), w = 1 is appended.
    Output: (clip, ndc)
      clip -- (N, 4) homogeneous clip coordinates. NOT clipped here:
              points with w <= 0 are returned as-is and must be rejected
              by the rasterizer's near-plane clip before dividing.
      ndc  -- (N, 3) = clip.xyz / clip.w (one broadcast divide, column 3
              once). Only meaningful for points with w > 0.

    Why this order: the chain is P @ V @ M @ p -- model first (right-
    most), then view, then projection. Composing the three matrices
    once into MVP and multiplying the whole vertex batch by it is the
    same result at a fraction of the matrix work.
    """
    verts = np.asarray(verts, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] not in (3, 4):
        raise ValueError(f"verts must have shape (N, 3) or (N, 4), got {verts.shape}")
    if verts.shape[1] == 3:
        ones = np.ones((verts.shape[0], 1), dtype=np.float64)
        verts = np.hstack([verts, ones])

    mvp = P @ V @ M                      # (4, 4) combined transform
    clip = (mvp @ verts.T).T             # (N, 4): rows are points, as always
    w = clip[:, 3:4]                     # keep 2D so the divide broadcasts
    ndc = clip[:, :3] / w
    return clip, ndc
