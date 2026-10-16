"""
broken_scene.py -- the three classic ways a pipeline like this breaks, live.

Phase 7 asks for a "bug showcase": deliberately wrong versions of the
renderer, each behind its own flag, wrapped in a tool that proves what
went wrong and that the one-line fix brings the frame back:

    python broken_scene.py --bug inside-out
    python broken_scene.py --bug zfight
    python broken_scene.py --bug stretched

The three bugs, in the order they usually bite:

  inside-out -- every triangle's winding is reversed, so back-face
      culling removes the faces that should be visible and keeps the
      inside. Silhouettes look identical; the surfaces are all wrong.
  zfight -- the near plane is pushed to 1e-3, so the whole model sits in
      a sliver of the depth range and a 16-bit depth buffer can no
      longer order surfaces that are a world unit apart: pixels flip
      between surfaces depending on rounding. The project's own depth
      buffer is float64 and cannot z-fight by construction, so this demo
      honestly emulates the fixed-point buffer real hardware uses.
  stretched -- project() gets its three matrices in the wrong order
      (V @ P instead of P @ V), so the perspective divide happens before
      the camera transform and the frame shears into garbage.

What the tool prints: the cause, hard evidence (render counters, the
depth histogram, and for zfight a flicker count between two half-LSB
shifted re-renders), how the broken frame scores against the correct one
(PSNR / SSIM / differing pixels), the fix, and proof that restoring the
right code reproduces the correct frame.

What it writes (out/):
  broken_<bug>.ppm        the broken frame (plus a .png preview)
  broken_<bug>_diff.ppm   |correct - broken|, gain 1 (no amplification)

Exit code is always 0: this is a showcase, not a check. The numeric
checks live in tests.py (phase 7).
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import compare
import objio
import ppm_io
import raster3d
from raster3d import Camera, RenderSettings, Texture, render_scene, write_png_preview
from transforms import look_at, perspective, project, viewport


DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "out"

# The three flags, in the order --help lists them.
BUGS = ("inside-out", "zfight", "stretched")

# World point used as the depth probe in the inside-out demo: the centre
# of the "body" box's front wall (body spans x, z in [-1.6, 1.6] and y in
# [2.1, 4.7]). Projecting it finds the pixel where the front wall sits,
# far from any silhouette.
PROBE_POINT = (0.0, 3.4, 1.6)


def _load_scene():
    """Load the temple mesh and texture; abort with a hint when missing.

    Output: (mesh, texture) -- objio.ObjMesh and raster3d.Texture.
    """
    obj_path = DATA_DIR / "temple.obj"
    tex_path = DATA_DIR / "relief.ppm"
    if not (obj_path.is_file() and tex_path.is_file()):
        print("missing data/ assets -- run: python make_model.py")
        raise SystemExit(1)
    mesh = objio.load_obj(obj_path)
    texture = Texture(ppm_io.read_ppm(tex_path)[2])
    return mesh, texture


def _probe_pixel(settings, camera):
    """Pixel where PROBE_POINT lands at this frame size: (row, col).

    Why project the point instead of hard-coding pixels: the probe must
    stay on the body's front wall at any width/height the tests use.
    """
    p_matrix = perspective(
        camera.fovy_deg, settings.width / settings.height, camera.near, camera.far
    )
    v_matrix = look_at(camera.eye, camera.target, camera.up)
    _, ndc = project(p_matrix, v_matrix, np.eye(4), np.array([PROBE_POINT]))
    x_pix, y_pix, _ = viewport(ndc, settings.width, settings.height)[0]
    return int(y_pix), int(x_pix)


def _swap_winding(mesh):
    """Copy of the mesh with every triangle's winding reversed.

    Input:  mesh -- objio.ObjMesh.
    Output: ObjMesh sharing positions and uvs; the corner order of both
            index arrays becomes (0, 2, 1).

    Why both arrays at once: the UVs hang off the corners, so winding and
    UV assignment must turn together or the texture scrambles too.
    """
    return objio.ObjMesh(
        positions=mesh.positions,
        uvs=mesh.uvs,
        face_vertex_indices=mesh.face_vertex_indices[:, [0, 2, 1]],
        face_uv_indices=mesh.face_uv_indices[:, [0, 2, 1]],
    )


def _covered(result):
    """Count the pixels covered by geometry (finite depth in the render)."""
    return int(np.isfinite(result.depth_ndc).sum())


def _depth_codes(result, bits):
    """Distinct depth codes over covered pixels, viewed at `bits` per channel.

    Inputs: result -- RenderResult; bits -- 8 or 16.
    Output: sorted np.ndarray of unique integer codes.
    Why covered pixels only: the background depth is pinned to 1.0 and
    would wash out the histogram on its own.
    """
    covered = np.isfinite(result.depth_ndc)
    codes = np.round(result.depth01[covered] * (2**bits - 1)).astype(np.int64)
    return np.unique(codes)


def _render_fixed16(mesh, texture, settings, camera, dither):
    """Render with the depth interpolator quantized like a 16-bit buffer.

    Inputs: as render_scene, plus dither -- shift of the quantization
            grid, in LSBs (the demos use 0.0 and 0.5).
    Output: RenderResult.

    Why this emulation at all: this project's depth buffer is float64, so
    it cannot z-fight -- exact arithmetic orders nearby surfaces the same
    way every frame. Real hardware stores window-space depth fixed-point,
    rounding the interpolated value to a 1/65535 grid, and that rounding
    is where fighting comes from. raster3d.interp_linear is the single
    place the rasterizer computes an interpolated depth (per pixel,
    inside rasterize_triangle), so swapping it for a rounding version
    during one render emulates a 16-bit depth buffer without touching
    the real code.
    """
    step = 2.0 / 65535.0  # NDC z spans [-1, 1]
    original = raster3d.interp_linear

    def quantized(weights, values):
        # (z + 1) / step is the fixed-point code; rounding to an integer
        # IS the storage limit; dither slides the grid by a fraction of
        # one code.
        z = original(weights, values)
        return np.round((z + 1.0) / step + dither) * step - 1.0

    raster3d.interp_linear = quantized
    try:
        return render_scene(mesh, texture, settings, camera)
    finally:
        raster3d.interp_linear = original


def demo_inside_out(mesh, texture, settings, camera=Camera()):
    """Bug 1: reversed winding -- cull the outside, render the inside.

    Inputs: mesh, texture, settings, camera -- the usual scene.
    Output: dict with the report fields (name, cause, evidence, fix,
            fix_proof, comparison, broken_srgb) plus culled_ref,
            culled_broken, probe_depth_ref, probe_depth_broken and
            fix_identical for the tests.
    """
    reference = render_scene(mesh, texture, settings, camera)
    broken = render_scene(_swap_winding(mesh), texture, settings, camera)

    row, col = _probe_pixel(settings, camera)
    probe_ref = float(reference.depth01[row, col])
    probe_broken = float(broken.depth01[row, col])

    # The fix is the same one-line reorder applied once more: it turns
    # the broken mesh back into the original, so the proof is a byte
    # comparison against the reference frame.
    fixed = render_scene(_swap_winding(_swap_winding(mesh)), texture, settings, camera)

    return {
        "name": "inside-out",
        "cause": (
            "every triangle's winding is reversed, so back-face culling "
            "removes the outward faces and renders the inside of the far "
            "walls instead."
        ),
        "evidence": [
            f"culled triangles: {reference.stats['culled']} in the correct "
            f"frame vs {broken.stats['culled']} in the broken one -- the "
            f"{broken.stats['rasterized']} now rasterized are the old back faces",
            f"depth at pixel (row {row}, col {col}), the front wall of the "
            f"body: {probe_ref:.6f} correct vs {probe_broken:.6f} broken -- "
            f"the ray passes through the culled wall and lands on the far side",
        ],
        "fix": (
            "keep triangles wound CCW seen from outside; for this mesh that "
            "is exactly the reverse of the [0, 2, 1] swap."
        ),
        "fix_proof": (
            "applying the reverse swap to the broken mesh reproduces the "
            "reference frame byte for byte."
        ),
        "comparison": compare.compare_images(
            reference.color_srgb, broken.color_srgb, gain=1.0
        ),
        "broken_srgb": broken.color_srgb,
        "culled_ref": reference.stats["culled"],
        "culled_broken": broken.stats["culled"],
        "probe_depth_ref": probe_ref,
        "probe_depth_broken": probe_broken,
        "fix_identical": bool(
            np.array_equal(fixed.color_srgb, reference.color_srgb)
        ),
    }


def demo_zfight(mesh, texture, settings, camera=Camera()):
    """Bug 2: near plane at 1e-3 -- depth precision collapses to nothing.

    Inputs: as demo_inside_out.
    Output: dict with the report fields plus codes_ref, codes_broken,
            codes16_broken (distinct depth codes) and flicker_broken,
            flicker_fixed, fixed_vs_ref_differing for the tests.

    How the fight is exposed: one LSB of a 16-bit buffer is a fixed step
    in NDC z, but the world-space size of that step grows with the square
    of the distance AND, for near=1e-3, with the compressed depth range
    itself. Near the model one LSB is ~1.5 world units -- bigger than the
    offsets between the roof steps -- so which surface wins a pixel is
    decided by sub-LSB rounding. Rendering twice with the quantization
    grid shifted half an LSB apart and counting informative pixels is
    flicker: the same frame, drawn twice, disagrees with itself.
    """
    reference = render_scene(mesh, texture, settings, camera)
    camera_broken = replace(camera, near=1e-3)

    broken = _render_fixed16(mesh, texture, settings, camera_broken, dither=0.0)
    broken_shift = _render_fixed16(mesh, texture, settings, camera_broken, dither=0.5)

    fixed = _render_fixed16(mesh, texture, settings, camera, dither=0.0)
    fixed_shift = _render_fixed16(mesh, texture, settings, camera, dither=0.5)

    codes_ref = _depth_codes(reference, bits=8)
    codes_broken = _depth_codes(broken, bits=8)
    codes16_broken = _depth_codes(broken, bits=16)

    diff = lambda a, b: int(np.any(a.color_srgb != b.color_srgb, axis=2).sum())
    flicker_broken = diff(broken, broken_shift)
    flicker_fixed = diff(fixed, fixed_shift)
    fixed_vs_ref = diff(fixed, reference)

    return {
        "name": "zfight",
        "cause": (
            "the near plane sits at 1e-3, so the whole model is pushed "
            "against the far end of the depth range; a 16-bit fixed-point "
            "depth buffer then cannot tell most surfaces apart."
        ),
        "evidence": [
            f"8-bit depth view: the correct frame spreads over "
            f"{len(codes_ref)} codes ({int(codes_ref[0])}..{int(codes_ref[-1])}); "
            f"with near=1e-3 everything collapses into "
            f"{len(codes_broken)} code(s) ({int(codes_broken[0])})",
            f"16-bit fixed-point buffer at near=1e-3: only "
            f"{len(codes16_broken)} of 65536 codes are ever stored for the "
            f"model -- one code is worth about a world unit near the walls",
            f"flicker: sliding the quantization grid half an LSB flips "
            f"{flicker_broken} pixels at near=1e-3 vs {flicker_fixed} at "
            f"near=1 -- those are pixels whose winner is decided by sub-LSB "
            f"rounding, i.e. by noise",
        ],
        "fix": (
            "keep the near plane as far as the framing allows (near=1 "
            "here): the depth LSB shrinks back to ~1e-3 world units and "
            "the surfaces can be ordered again."
        ),
        "fix_proof": (
            f"with near=1 the same 16-bit emulation differs from the "
            f"float64 reference in {fixed_vs_ref} pixel(s) "
            f"({'byte-identical' if fixed_vs_ref == 0 else 'exact seam edges only'}) "
            f"and the flicker count drops to {flicker_fixed}."
        ),
        "comparison": compare.compare_images(
            reference.color_srgb, broken.color_srgb, gain=1.0
        ),
        "broken_srgb": broken.color_srgb,
        "codes_ref": int(len(codes_ref)),
        "codes_broken": int(len(codes_broken)),
        "codes16_broken": int(len(codes16_broken)),
        "flicker_broken": flicker_broken,
        "flicker_fixed": flicker_fixed,
        "fixed_vs_ref_differing": fixed_vs_ref,
    }


def demo_stretched(mesh, texture, settings, camera=Camera()):
    """Bug 3: project() called with (P, V, M) in the wrong order.

    Inputs: as demo_inside_out.
    Output: dict with the report fields plus covered_ref, covered_broken
            and fix_identical for the tests.

    The pinned chain is p' = P @ V @ M @ p (model first, then view, then
    projection). The broken version swaps the outer two: V @ P. The
    perspective divide then happens before the camera transform, and the
    frame shears into garbage. The demo swaps the function itself for one
    render, so nothing in the real renderer changes.
    """
    reference = render_scene(mesh, texture, settings, camera)
    original_project = raster3d.project

    def swapped_project(p_matrix, v_matrix, m_matrix, verts):
        # The bug: hand the matrices to the real project() in the order
        # (M, V, P) -- i.e. compose V @ P @ M instead of P @ V @ M.
        return original_project(m_matrix, v_matrix, p_matrix, verts)

    raster3d.project = swapped_project
    try:
        broken = render_scene(mesh, texture, settings, camera)
    finally:
        raster3d.project = original_project

    fixed = render_scene(mesh, texture, settings, camera)

    p_matrix = perspective(
        camera.fovy_deg, settings.width / settings.height, camera.near, camera.far
    )
    v_matrix = look_at(camera.eye, camera.target, camera.up)

    covered_ref = _covered(reference)
    covered_broken = _covered(broken)

    return {
        "name": "stretched",
        "cause": (
            "project() composes V @ P instead of P @ V: the perspective "
            "divide runs before the camera transform, so screen "
            "coordinates are built from the wrong space."
        ),
        "evidence": [
            "correct chain P @ V (M is identity) =\n"
            + np.array2string(p_matrix @ v_matrix, precision=3, suppress_small=True),
            "broken chain V @ P =\n"
            + np.array2string(v_matrix @ p_matrix, precision=3, suppress_small=True),
            f"covered pixels: {covered_ref} correct vs {covered_broken} broken "
            f"-- the silhouette geometry no longer matches the model",
        ],
        "fix": (
            "call project(P, V, M, verts); the chain is P @ V @ M @ p, and "
            "the rightmost matrix (the model) acts first."
        ),
        "fix_proof": (
            "restoring the argument order reproduces the reference frame "
            "byte for byte."
        ),
        "comparison": compare.compare_images(
            reference.color_srgb, broken.color_srgb, gain=1.0
        ),
        "broken_srgb": broken.color_srgb,
        "covered_ref": covered_ref,
        "covered_broken": covered_broken,
        "fix_identical": bool(np.array_equal(fixed.color_srgb, reference.color_srgb)),
    }


# The --bug flag values and the demo each one runs.
DEMOS = {
    "inside-out": demo_inside_out,
    "zfight": demo_zfight,
    "stretched": demo_stretched,
}


def _print_block(title, entries):
    """Print a titled block, indenting every (possibly multi-line) entry."""
    print(f"{title}:")
    for entry in entries:
        for line in str(entry).splitlines() or [""]:
            print(f"  {line}")


def _print_report(result, ppm_path, diff_path, settings):
    """Print cause, evidence, score against the reference, fix and outputs."""
    comparison = result["comparison"]
    psnr_text = "inf" if np.isinf(comparison["psnr"]) else f"{comparison['psnr']:.2f}"
    print(f"broken scene: {result['name']}")
    _print_block("cause", [result["cause"]])
    _print_block("evidence", result["evidence"])
    _print_block(
        "against the correct frame",
        [
            f"PSNR {psnr_text} dB, SSIM {comparison['ssim']:.6f}, "
            f"{comparison['differing_pixels']} of "
            f"{settings.width * settings.height} pixels differ "
            f"({100.0 * comparison['differing_fraction']:.2f}%)"
        ],
    )
    _print_block("fix", [result["fix"]])
    _print_block("fix proof", [result["fix_proof"]])
    _print_block(
        "outputs", [f"{ppm_path} (+ .png preview)", f"{diff_path} (+ .png preview)"]
    )


def main(argv=None):
    """CLI: render one deliberately broken scene next to the correct one."""
    parser = argparse.ArgumentParser(
        description="Bug showcase: render a broken temple, prove the fix (phase 7)."
    )
    parser.add_argument(
        "--bug", choices=BUGS, required=True, help="which deliberate bug to show"
    )
    args = parser.parse_args(argv)

    mesh, texture = _load_scene()
    settings = RenderSettings()  # defaults = the final-stage look
    camera = Camera()

    result = DEMOS[args.bug](mesh, texture, settings, camera)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ppm_path = OUT_DIR / f"broken_{args.bug}.ppm"
    diff_path = OUT_DIR / f"broken_{args.bug}_diff.ppm"
    ppm_io.write_ppm(ppm_path, result["broken_srgb"])
    write_png_preview(ppm_path)
    ppm_io.write_ppm(diff_path, result["comparison"]["diff"])
    write_png_preview(diff_path)

    _print_report(result, ppm_path, diff_path, settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())


