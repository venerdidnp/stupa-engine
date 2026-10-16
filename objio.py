"""
objio.py -- minimal Wavefront OBJ loader (vertices, UVs, faces).

Scope: only the 'v', 'vt' and 'f' records are read. Everything else that
normally appears in an OBJ file ('vn', 'o', 'g', 's', 'mtllib', 'usemtl',
...) is skipped without error.

Behaviour pinned by the project's conventions:
- Polygons are fan-triangulated: (a, b, c, d) -> (a,b,c) + (a,c,d).
  The fan preserves the original corner order, and OBJ mandates that
  faces are wound counter-clockwise seen from outside, so the CCW
  convention survives loading untouched.
- All four OBJ index forms are accepted: 'v', 'v/vt', 'v//vn' and
  'v/vt/vn'. An absent vt index is stored as -1.
- Negative (relative) OBJ indices are resolved when the face is read,
  relative to the counts of vertices/UVs seen so far.
- The v of every UV is flipped HERE, exactly once, so all other files
  can assume a bottom-left UV origin.
- All returned indices are 0-based (OBJ files store them 1-based).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ObjMesh:
    """A loaded OBJ mesh, indices left un-expanded (no vertex duplication).

    Why keep separate position and UV index streams: OBJ decouples them
    (each face corner may point at different v and vt records), which is
    how a box can share 8 corner positions yet have per-face planar UVs.
    The rasterizer expands corners into a triangle soup when it needs to.

    Fields:
      positions           -- (N, 3) float64 vertex positions.
      uvs                 -- (M, 2) float64, v already flipped (bottom-left).
      face_vertex_indices -- (T, 3) int32, 0-based indices into positions.
      face_uv_indices     -- (T, 3) int32, 0-based indices into uvs,
                             or -1 where the file's face had no vt.
    """

    positions: np.ndarray
    uvs: np.ndarray
    face_vertex_indices: np.ndarray
    face_uv_indices: np.ndarray


def _resolve_index(text, count, kind, line_no):
    """Turn one OBJ index token into a 0-based index.

    Inputs: text -- the token; count -- how many records of that kind
            exist so far (needed for negative indices); kind -- 'vertex'
            or 'uv', for error messages; line_no -- for error messages.
    Output: 0-based int index.
    Why resolve early: faces may legally refer backwards with negative
    indices, but downstream code should never have to know that.
    """
    try:
        index = int(text)
    except ValueError:
        raise ValueError(f"objio: bad index {text!r} on line {line_no}") from None
    if index > 0:
        resolved = index - 1
    elif index < 0:
        resolved = count + index  # -1 = last record seen so far
    else:
        raise ValueError(f"objio: index 0 is not valid in OBJ (line {line_no})")
    if not 0 <= resolved < count:
        raise ValueError(
            f"objio: {kind} index {index} out of range on line {line_no}"
        )
    return resolved


def _parse_face_corner(token, vertex_count, uv_count, line_no):
    """Parse one 'v', 'v/vt', 'v//vn' or 'v/vt/vn' corner token.

    Output: (vertex_index, uv_index); uv_index is -1 when absent.
    """
    parts = token.split("/")
    vertex_index = _resolve_index(parts[0], vertex_count, "vertex", line_no)
    uv_index = -1
    if len(parts) >= 2 and parts[1] != "":
        uv_index = _resolve_index(parts[1], uv_count, "uv", line_no)
    return vertex_index, uv_index


def load_obj_lines(lines):
    """Load an OBJ mesh from an iterable of text lines (used by tests too).

    Input:  lines -- iterable of str; '#' starts a comment to end of line.
    Output: ObjMesh (see the dataclass for shapes).
    """
    positions = []
    uvs = []
    face_vertices = []
    face_uvs = []

    for line_no, raw in enumerate(lines, start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        record = fields[0]

        try:
            if record == "v":
                positions.append((float(fields[1]), float(fields[2]), float(fields[3])))
            elif record == "vt":
                # OBJ stores (u, v) with v up; third field (w) is ignored.
                uvs.append((float(fields[1]), float(fields[2])))
            elif record == "f":
                corners = [
                    _parse_face_corner(tok, len(positions), len(uvs), line_no)
                    for tok in fields[1:]
                ]
                if len(corners) < 3:
                    raise ValueError(
                        f"objio: face with fewer than 3 corners (line {line_no})"
                    )
                # Fan triangulation around the first corner: preserves
                # winding for the convex polygons OBJ files use in practice.
                first = corners[0]
                for i in range(1, len(corners) - 1):
                    for vertex_index, uv_index in (first, corners[i], corners[i + 1]):
                        face_vertices.append(vertex_index)
                        face_uvs.append(uv_index)
            # Any other record type (vn, o, g, s, ...) is skipped silently.
        except IndexError:
            raise ValueError(
                f"objio: malformed record on line {line_no}: {raw!r}"
            ) from None

    positions_arr = np.array(positions, dtype=np.float64).reshape(-1, 3)
    uvs_arr = np.array(uvs, dtype=np.float64).reshape(-1, 2)

    # Flip v exactly once, at the boundary between file and project.
    # Rationale: OBJ puts v = 0 at the bottom of the texture, image files
    # store row 0 at the top. Flipping now means everything downstream
    # (sampling included) uses one consistent bottom-left origin.
    if uvs_arr.size:
        uvs_arr[:, 1] = 1.0 - uvs_arr[:, 1]

    return ObjMesh(
        positions=positions_arr,
        uvs=uvs_arr,
        face_vertex_indices=np.array(face_vertices, dtype=np.int32).reshape(-1, 3),
        face_uv_indices=np.array(face_uvs, dtype=np.int32).reshape(-1, 3),
    )


def load_obj(path):
    """Load an OBJ mesh from a file.

    Input:  path -- file path (str or Path).
    Output: ObjMesh (see the dataclass for shapes).
    Why errors='replace': OBJ is a text format; a stray non-UTF-8 byte in
    a comment must not take down the whole load.
    """
    with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
        return load_obj_lines(fh)
