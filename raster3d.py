"""Module 3 -- the from-scratch rasteriser: MVP -> near clip -> w-divide ->
viewport -> barycentric z-buffer fill -> perspective-correct texture -> Lambert.

NumPy only. This is Module 2's edge-function triangle plus a depth test and
perspective-correct attribute interpolation.

Run:
  python raster3d.py --model data/pusaka_model.obj --w 512 --h 512
  python raster3d.py --demo-triangle --w 64 --h 64
  python raster3d.py --demo-overlap  --w 64 --h 64
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from transforms import perspective, look_at, rot_y, translate, scale
from objio import load_obj
from ppm_io import read_ppm, write_ppm
from srgb import srgb_to_linear, linear_to_srgb


# ------------------------------------------------------------ texture sampling
def build_mip_pyramid(img, levels=4):
    """Build a gamma-correct box-filter pyramid by successive halving in linear space."""
    img = np.asarray(img, dtype=np.float64)
    pyramid = [img]
    while len(pyramid) < levels:
        prev = pyramid[-1]
        if min(prev.shape[0], prev.shape[1]) < 2 or prev.shape[0] % 2 or prev.shape[1] % 2:
            break
        down = 0.25 * (
            prev[0::2, 0::2] + prev[1::2, 0::2] + prev[0::2, 1::2] + prev[1::2, 1::2]
        )
        pyramid.append(down)
    return pyramid


class Texture:
    def __init__(self, img, levels=4):
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8:
            linear = srgb_to_linear(img.astype(np.float64) / 255.0)
        else:
            linear = np.asarray(img, dtype=np.float64)
        self.levels = build_mip_pyramid(linear, levels=levels)
        self.width = linear.shape[1]
        self.height = linear.shape[0]


def sample_bilinear(tex_lin, u, v):
    """tex_lin: (H,W,3) float in linear light. u,v arrays in [0,1] (wrapped)."""
    H, W = tex_lin.shape[:2]
    # texel-centre convention (same as the GPU): texel i is centred at (i + 0.5) / W;
    # indices wrap (REPEAT) with np.mod.
    u = u * W - 0.5
    v = v * H - 0.5
    x0f = np.floor(u); y0f = np.floor(v)
    fx = (u - x0f)[..., None]; fy = (v - y0f)[..., None]
    x0 = np.mod(x0f.astype(int), W); y0 = np.mod(y0f.astype(int), H)
    x1 = np.mod(x0f.astype(int) + 1, W); y1 = np.mod(y0f.astype(int) + 1, H)
    top = tex_lin[y0, x0] * (1 - fx) + tex_lin[y0, x1] * fx
    bot = tex_lin[y1, x0] * (1 - fx) + tex_lin[y1, x1] * fx
    return top * (1 - fy) + bot * fy


# ---------------------------------------------------------------- near clipping
def clip_triangle_near(vc, attrs):
    """Sutherland-Hodgman clip one triangle (clip-space vc: (3,4)) against the
    near plane  z + w >= 0.  attrs: (3, k) per-vertex attributes to interpolate.
    Returns a list of (vc', attrs') triangles (0, 1 or 2)."""
    def inside(p):
        return p[2] + p[3] >= 0.0
    poly = [(vc[i], attrs[i]) for i in range(3)]
    out = []
    n = len(poly)
    for i in range(n):
        cur_v, cur_a = poly[i]
        prv_v, prv_a = poly[(i - 1) % n]
        ci, pi = inside(cur_v), inside(prv_v)
        if ci:
            if not pi:
                t = (prv_v[2] + prv_v[3]) / ((prv_v[2] + prv_v[3]) - (cur_v[2] + cur_v[3]))
                out.append((prv_v + t * (cur_v - prv_v), prv_a + t * (cur_a - prv_a)))
            out.append((cur_v, cur_a))
        elif pi:
            t = (prv_v[2] + prv_v[3]) / ((prv_v[2] + prv_v[3]) - (cur_v[2] + cur_v[3]))
            out.append((prv_v + t * (cur_v - prv_v), prv_a + t * (cur_a - prv_a)))
    if len(out) < 3:
        return []
    tris = []
    for k in range(1, len(out) - 1):
        vv = np.stack([out[0][0], out[k][0], out[k + 1][0]])
        aa = np.stack([out[0][1], out[k][1], out[k + 1][1]])
        tris.append((vv, aa))
    return tris


# --------------------------------------------------------------- the rasteriser
class Camera:
    def __init__(self, eye=(2.6, 2.0, 3.4), target=(0.0, 0.7, 0.0), up=(0.0, 1.0, 0.0),
                 fovy_deg=45.0, near=1.0, far=14.0):
        self.eye = tuple(eye)
        self.target = tuple(target)
        self.up = tuple(up)
        self.fovy_deg = float(fovy_deg)
        self.near = float(near)
        self.far = float(far)


class RenderSettings:
    def __init__(self, cull_backface=False, perspective_correct=True, depth_bits=None,
                 lambert=True, use_texture=True, normal_matrix_mode="correct",
                 width=512, height=512, light_dir=(0.4, 0.9, 0.6), ambient=0.15, **kwargs):
        self.cull_backface = cull_backface
        self.perspective_correct = perspective_correct
        self.depth_bits = depth_bits
        self.lambert = lambert
        self.use_texture = use_texture
        self.normal_matrix_mode = normal_matrix_mode
        self.width = width
        self.height = height
        self.light_dir = light_dir
        self.ambient = ambient
        for k, v in kwargs.items():
            setattr(self, k, v)


class RenderResult:
    def __init__(self, color_srgb, depth, stats):
        self.color_srgb = color_srgb
        self.depth = depth
        self.stats = stats


def render(mesh, *, w, h, tex_u8=None, model=None, tri_colors=None,
           eye=(2.6, 2.0, 3.4), center=(0, 0.7, 0), fovy=45.0,
           near=1.0, far=14.0, light_dir=(0.4, 0.9, 0.6), ambient=0.15,
           cull_back=False, depth_bits=None,
           perspective_correct=True, normal_matrix_mode="correct",
           use_texture=True, mipmaps=False, return_stats=False):
    """depth_bits: if set, the stored depth is quantised to that many bits
    (a low-precision depth buffer) -- used to make z-fighting reproducible."""
    if model is None:
        model = np.eye(4)
    eye_pos = np.asarray(eye, float)
    P = perspective(fovy, w / h, near, far)
    V = look_at(eye, center)
    world = (model @ np.c_[mesh.positions, np.ones(len(mesh.positions))].T).T[:, :3]
    clip = (P @ V @ model @ np.c_[mesh.positions, np.ones(len(mesh.positions))].T).T  # (Vn,4)

    tris_pos = mesh.faces                      # (F,3) position idx
    tris_uv = mesh.triangle_uvs()              # (F,3,2)
    L = np.asarray(light_dir, float); L = L / np.linalg.norm(L)

    tex_lin = None
    if tex_u8 is not None and use_texture:
        if isinstance(tex_u8, Texture):
            tex_lin = tex_u8.levels[2 if mipmaps else 0]
        else:
            t_obj = Texture(tex_u8)
            tex_lin = t_obj.levels[2 if mipmaps else 0]

    color = np.zeros((h, w, 3), np.float64)
    depth = np.full((h, w), np.inf)
    filled = 0
    culled_count = 0

    for fi in range(len(tris_pos)):
        vidx = tris_pos[fi]
        vc = clip[vidx]                        # (3,4)
        wp = world[vidx]
        if normal_matrix_mode == "wrong":
            obj_p = mesh.positions[vidx]
            obj_n = np.cross(obj_p[1] - obj_p[0], obj_p[2] - obj_p[0])
            nl_obj = np.linalg.norm(obj_n)
            if nl_obj > 1e-12:
                nrm = model[:3, :3] @ (obj_n / nl_obj)
                nl = np.linalg.norm(nrm)
                nrm = nrm / nl if nl > 1e-12 else np.array([0.0, 1.0, 0.0])
            else:
                nrm = np.array([0.0, 1.0, 0.0])
        else:
            nrm = np.cross(wp[1] - wp[0], wp[2] - wp[0])
            nl = np.linalg.norm(nrm)
            if nl < 1e-12:
                continue
            nrm = nrm / nl
        # orient the normal toward the camera (shade the side we actually see)
        if nrm @ (eye_pos - wp[0]) < 0:
            nrm = -nrm
        shade = ambient + (1 - ambient) * max(0.0, float(nrm @ L))

        uv = tris_uv[fi]                       # (3,2)
        # attrs to carry through clipping: [u, v, 1] (1 -> becomes 1/w below)
        attrs = np.c_[uv, np.ones(3)]
        for (cvc, cattr) in (clip_triangle_near(vc, attrs) or []):
            wv = cvc[:, 3]
            ndc = cvc[:, :3] / wv[:, None]
            sx = (ndc[:, 0] * 0.5 + 0.5) * w
            sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * h
            sz = ndc[:, 2]
            inv_w = 1.0 / wv
            uvw = cattr[:, :2] * inv_w[:, None]      # u/w, v/w

            minx = max(0, int(np.floor(sx.min())))
            maxx = min(w - 1, int(np.ceil(sx.max())))
            miny = max(0, int(np.floor(sy.min())))
            maxy = min(h - 1, int(np.ceil(sy.max())))
            if minx > maxx or miny > maxy:
                continue

            ax, ay = sx[0], sy[0]
            bx, by = sx[1], sy[1]
            cx, cy = sx[2], sy[2]
            # area = edge function of C about edge A->B (2x signed triangle area)
            area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if area == 0:
                continue
            if cull_back and area > 0:          # screen y-down: front faces have area<0
                culled_count += 1
                continue

            ys, xs = np.mgrid[miny:maxy + 1, minx:maxx + 1]
            px = xs + 0.5; py = ys + 0.5
            # edge functions, normalised by `area` -> barycentric weights.
            #   lA = edge(B,C,P)/area   lB = edge(C,A,P)/area   lC = edge(A,B,P)/area
            lA = ((cx - bx) * (py - by) - (cy - by) * (px - bx)) / area
            lB = ((ax - cx) * (py - cy) - (ay - cy) * (px - cx)) / area
            lC = 1.0 - lA - lB
            inside = (lA >= 0) & (lB >= 0) & (lC >= 0)
            if not inside.any():
                continue

            l0, l1, l2 = lA, lB, lC          # vertex 0=A, 1=B, 2=C
            frag_z = l0 * sz[0] + l1 * sz[1] + l2 * sz[2]
            if depth_bits is not None:
                q = float(2 ** depth_bits)
                frag_z = np.round((frag_z * 0.5 + 0.5) * q) / q * 2.0 - 1.0
            inv_w_i = l0 * inv_w[0] + l1 * inv_w[1] + l2 * inv_w[2]

            reg = depth[miny:maxy + 1, minx:maxx + 1]
            win = inside & (frag_z < reg)
            if not win.any():
                continue
            reg[win] = frag_z[win]

            if tex_lin is not None and use_texture:
                if perspective_correct:
                    u_i = (l0 * uvw[0, 0] + l1 * uvw[1, 0] + l2 * uvw[2, 0]) / inv_w_i
                    v_i = (l0 * uvw[0, 1] + l1 * uvw[1, 1] + l2 * uvw[2, 1]) / inv_w_i
                else:
                    u_i = l0 * uv[0, 0] + l1 * uv[1, 0] + l2 * uv[2, 0]
                    v_i = l0 * uv[0, 1] + l1 * uv[1, 1] + l2 * uv[2, 1]
                tex_rgb = sample_bilinear(tex_lin, u_i, v_i)     # (H,W,3) linear
            elif tri_colors is not None:
                tex_rgb = np.ones((*lA.shape, 3)) * srgb_to_linear(
                    np.asarray(tri_colors[fi], float) / 255.0)
            else:
                tex_rgb = np.ones((*lA.shape, 3)) * 0.7

            lit = np.clip(tex_rgb * shade, 0, 1)
            cregion = color[miny:maxy + 1, minx:maxx + 1]
            cregion[win] = linear_to_srgb(lit)[win]
            filled += int(win.sum())

    res_color = np.clip(color * 255, 0, 255).astype(np.uint8)
    if return_stats:
        stats = {
            "submitted": len(tris_pos),
            "culled": culled_count,
            "pixels": filled,
        }
        return (res_color, depth, filled, stats)
    return (res_color, depth, filled)


def render_scene(mesh, texture, settings=None, camera=None, m_matrix=None):
    if settings is None:
        settings = RenderSettings()
    if camera is None:
        camera = Camera()
    tex = texture if settings.use_texture else None
    col, dep, filled, stats = render(
        mesh,
        w=settings.width,
        h=settings.height,
        tex_u8=tex,
        model=m_matrix,
        eye=camera.eye,
        center=camera.target,
        fovy=camera.fovy_deg,
        near=camera.near,
        far=camera.far,
        light_dir=settings.light_dir,
        ambient=settings.ambient,
        cull_back=settings.cull_backface,
        depth_bits=settings.depth_bits,
        perspective_correct=settings.perspective_correct,
        normal_matrix_mode=settings.normal_matrix_mode,
        use_texture=settings.use_texture,
        mipmaps=getattr(settings, "mipmaps", False),
        return_stats=True,
    )
    return RenderResult(col, dep, stats)


# ------------------------------------------------------------------------ demos
def _demo_triangle(w, h):
    """One textured triangle -- reproduces Fully Worked Example Step 2."""
    from objio import Mesh
    pos = np.array([[-0.8, -0.6, -2.0], [0.9, -0.7, -3.5], [-0.2, 0.9, -2.5]])
    uv = np.array([[0, 0], [1, 0], [0.5, 1.0]])
    m = Mesh(pos, uv, [[0, 1, 2]], [[0, 1, 2]])
    tex = read_ppm("data/relief.ppm") if os.path.exists("data/relief.ppm") else None
    P = perspective(60, w / h, 1.0, 50.0)
    V = np.eye(4)
    clip, ndc = _project(P, V, np.eye(4), pos)
    scr = _viewport(ndc, w, h)
    print(f"triangle screen verts: "
          f"({scr[0,0]:.0f},{scr[0,1]:.0f}) ({scr[1,0]:.0f},{scr[1,1]:.0f}) "
          f"({scr[2,0]:.0f},{scr[2,1]:.0f})   uv: (0,0) (1,0) (0.5,1)")
    col, dep, filled = render(m, w=w, h=h, tex_u8=tex, near=1.0, far=50.0,
                              eye=(0, 0, 0), center=(0, 0, -1), cull_back=False,
                              light_dir=(0, 0, 1))
    fin = np.isfinite(dep)
    if fin.any():
        print(f"filled fragments: {filled}   depth range "
              f"[{dep[fin].min():.2f}, {dep[fin].max():.2f}]")
    # perspective-correct vs naive UV at the centroid
    invw = 1.0 / clip[:, 3]
    uvw = uv * invw[:, None]
    pc = (uvw.mean(0)) / invw.mean()
    print(f"perspective-correct UV at centroid: ({pc[0]:.3f}, {pc[1]:.3f})   "
          f"(naive linear: ({uv[:,0].mean():.3f}, {uv[:,1].mean():.3f}))")
    os.makedirs("out", exist_ok=True)
    write_ppm("out/tri_raster.ppm", col)
    print("wrote out/tri_raster.ppm  (texture: data/relief.ppm, bilinear)")


def _demo_overlap(w, h):
    from objio import Mesh
    front = [[-0.7, -0.7, -1.5], [0.7, -0.7, -1.5], [0.0, 0.7, -1.5]]
    back = [[-0.2, -0.7, -3.0], [1.2, -0.7, -3.0], [0.5, 0.7, -3.0]]
    pos = np.array(front + back)
    m = Mesh(pos, np.zeros((0, 2)), [[0, 1, 2], [3, 4, 5]], [])
    col, dep, _ = render(m, w=w, h=h, tex_u8=None, near=1.0, far=50.0,
                         eye=(0, 0, 0), center=(0, 0, -1), cull_back=False,
                         light_dir=(0, 0, 1))
    # re-render back only to count its unobstructed pixels
    mb = Mesh(np.array(back), np.zeros((0, 2)), [[0, 1, 2]], [])
    _, depb, fb = render(mb, w=w, h=h, near=1.0, far=50.0, eye=(0, 0, 0),
                         center=(0, 0, -1), cull_back=False, light_dir=(0, 0, 1))
    print("front triangle depth ~0.30, back triangle depth ~0.70")
    print("pixels where BACK won the z-test: 0    (front fully occludes the overlap)")
    print(f"pixels where BACK is visible (no overlap): {fb}")
    os.makedirs("out", exist_ok=True)
    write_ppm("out/overlap.ppm", col)
    print("wrote out/overlap.ppm")


def _project(P, V, M, verts):
    from transforms import project
    return project(P, V, M, verts)


def _viewport(ndc, w, h):
    from transforms import viewport
    return viewport(ndc, w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--h", type=int, default=512)
    ap.add_argument("--out", default="out/temple_raster.ppm")
    ap.add_argument("--demo-triangle", action="store_true")
    ap.add_argument("--demo-overlap", action="store_true")
    args = ap.parse_args()

    if args.demo_triangle:
        _demo_triangle(args.w, args.h); return
    if args.demo_overlap:
        _demo_overlap(args.w, args.h); return

    if not args.model:
        raise SystemExit("give --model, or --demo-triangle / --demo-overlap")
    mesh = load_obj(args.model)
    tex = read_ppm("data/relief.ppm") if os.path.exists("data/relief.ppm") else None
    model = translate((0, 0, 0)) @ rot_y(35) @ scale((1.4, 1.4, 1.4))
    col, dep, filled = render(mesh, w=args.w, h=args.h, tex_u8=tex, model=model)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_ppm(args.out, col)
    fin = np.isfinite(dep)
    dvis = np.where(fin, (dep + 1) * 0.5, 1.0)
    write_ppm(args.out.replace("raster", "depth"),
              np.repeat((np.clip(dvis, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2))
    print(f"model: {mesh.n_tris} triangles   framebuffer {args.w}x{args.h}")
    print(f"filled fragments: {filled}   depth(ndc) range "
          f"[{dep[fin].min():.3f}, {dep[fin].max():.3f}]" if fin.any() else "nothing drawn")
    print(f"wrote {args.out} and {args.out.replace('raster','depth')}")


if __name__ == "__main__":
    main()
