"""
e4_demo.py -- demonstrates perspective-correct UV interpolation vs naive linear interpolation (Contoh E4).

Part 1:
  Prints step-by-step mathematical calculation of u at screen midpoint
  for an edge with wL = 1 (u=0) and wR = 9 (u=1):
  Naive linear (0.5) vs perspective-correct (0.1).

Part 2:
  Renders a receding floor quad (2 triangles) textured with a procedural NumPy
  checkerboard twice:
  (a) Naive linear UV interpolation (settings.perspective_correct = False)
  (b) Perspective-correct UV interpolation (settings.perspective_correct = True)
  Saves:
    out/e4_naive.ppm / out/e4_naive.png
    out/e4_correct.ppm / out/e4_correct.png
    out/e4_diff.png (amplified difference image)

Run: python e4_demo.py
"""

from pathlib import Path
import numpy as np
from PIL import Image

import metrics
import objio
import ppm_io
import raster3d
from raster3d import Camera

OUT_DIR = Path(__file__).resolve().parent / "out"


def print_step_by_step_derivation():
    print("=" * 65)
    print("BAGIAN 1: PERHITUNGAN LANGKAH-DEMI-LANGKAH (TEPI wL=1, wR=9)")
    print("=" * 65)
    print("Parameter tepi:")
    print("  Ujung kiri  (L): w_L = 1.0, u_L = 0.0")
    print("  Ujung kanan (R): w_R = 9.0, u_R = 1.0")
    print("  Titik tengah layar (midpoint): bobot layar t = 0.5")
    print("")
    print("1. Interpolasi Naif (Linear di Screen Space):")
    print("   u_naive = (1 - t) * u_L + t * u_R")
    print("           = 0.5 * 0.0 + 0.5 * 1.0")
    print("           = 0.500000")
    print("")
    print("2. Interpolasi Perspective-Correct:")
    print("   a. Hitung atribut dibagi w pada kedua ujung:")
    print("      u_L / w_L = 0.0 / 1.0 = 0.000000")
    print("      u_R / w_R = 1.0 / 9.0 = 0.111111 (1/9)")
    print("      1.0 / w_L = 1.0 / 1.0 = 1.000000")
    print("      1.0 / w_R = 1.0 / 9.0 = 0.111111 (1/9)")
    print("")
    print("   b. Interpolasikan u/w dan 1/w secara linear di screen space:")
    print("      (u/w)_mid = 0.5 * (u_L / w_L) + 0.5 * (u_R / w_R)")
    print("                = 0.5 * 0.0 + 0.5 * (1/9) = 1/18 = 0.055556")
    print("      (1/w)_mid = 0.5 * (1.0 / w_L) + 0.5 * (1.0 / w_R)")
    print("                = 0.5 * 1.0 + 0.5 * (1/9) = 9/18 + 1/18 = 10/18 = 5/9 = 0.555556")
    print("")
    print("   c. Pulihkan u sesungguhnya dengan membagi kedua nilai:")
    print("      u_correct = (u/w)_mid / (1/w)_mid")
    print("                = (1/18) / (10/18) = 1/10")
    print("                = 0.100000")
    print("")
    print("Selisih: |u_naive - u_correct| = |0.5 - 0.1| = 0.400000 (kesalahan 40% dari rentang texture)")
    print("=" * 65)
    print("")


def create_checkerboard_texture(size=256, tiles=8):
    """Create a procedural checkerboard texture using pure NumPy."""
    tile_size = size // tiles
    x = np.arange(size) // tile_size
    y = np.arange(size) // tile_size
    grid = (x[None, :] + y[:, None]) % 2
    # Black [30, 30, 30] and white/cream [240, 240, 240]
    img = np.where(grid[:, :, None] == 0, [30, 30, 30], [240, 240, 240]).astype(np.uint8)
    return img


def build_floor_mesh():
    """Build a receding floor quad (2 triangles) extending in -z."""
    # Quad corners at y = -1.0, x in [-4, 4], z from +1.0 (near) to -15.0 (far)
    v0 = np.array([-4.0, -1.0, 1.0])
    v1 = np.array([4.0, -1.0, 1.0])
    v2 = np.array([4.0, -1.0, -15.0])
    v3 = np.array([-4.0, -1.0, -15.0])

    # Planar UVs covering 4 repeats
    vt0 = np.array([0.0, 0.0])
    vt1 = np.array([4.0, 0.0])
    vt2 = np.array([4.0, 4.0])
    vt3 = np.array([0.0, 4.0])

    positions = np.array([v0, v1, v2, v3], dtype=np.float64)
    uvs = np.array([vt0, vt1, vt2, vt3], dtype=np.float64)

    # CCW winding viewed from above (+y looking down):
    # tri1: v0 -> v1 -> v2
    # tri2: v0 -> v2 -> v3
    face_vertex_indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    face_uv_indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    return objio.ObjMesh(
        positions=positions,
        uvs=uvs,
        face_vertex_indices=face_vertex_indices,
        face_uv_indices=face_uv_indices,
    )


def render_floor_comparison():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh = build_floor_mesh()
    tex_data = create_checkerboard_texture(256, tiles=8)
    texture = raster3d.Texture(tex_data)

    # Camera looking at the floor
    cam = raster3d.Camera(
        eye=(0.0, 2.5, 3.5),
        target=(0.0, -1.0, -6.0),
        up=(0.0, 1.0, 0.0),
        fovy_deg=60.0,
        near=0.5,
        far=50.0,
    )

    # 1. Render Naive (perspective_correct=False)
    cfg_naive = raster3d.RenderSettings(
        width=600,
        height=450,
        use_texture=True,
        bilinear=True,
        mipmaps=False,
        lambert=False,
        perspective_correct=False,
    )
    res_naive = raster3d.render_scene(mesh, texture, cfg_naive, cam)

    # 2. Render Perspective-Correct (perspective_correct=True)
    cfg_correct = raster3d.RenderSettings(
        width=600,
        height=450,
        use_texture=True,
        bilinear=True,
        mipmaps=False,
        lambert=False,
        perspective_correct=True,
    )
    res_correct = raster3d.render_scene(mesh, texture, cfg_correct, cam)

    # Save outputs
    naive_ppm = OUT_DIR / "e4_naive.ppm"
    naive_png = OUT_DIR / "e4_naive.png"
    correct_ppm = OUT_DIR / "e4_correct.ppm"
    correct_png = OUT_DIR / "e4_correct.png"
    diff_png = OUT_DIR / "e4_diff.png"

    ppm_io.write_ppm(naive_ppm, res_naive.color_srgb)
    Image.fromarray(res_naive.color_srgb).save(naive_png)

    ppm_io.write_ppm(correct_ppm, res_correct.color_srgb)
    Image.fromarray(res_correct.color_srgb).save(correct_png)

    diff_img = metrics.diff_image(res_correct.color_srgb, res_naive.color_srgb, gain=2.0)
    Image.fromarray(diff_img).save(diff_png)

    psnr = metrics.psnr(res_correct.color_srgb, res_naive.color_srgb)
    ssim = metrics.ssim(res_correct.color_srgb, res_naive.color_srgb)
    mean_diff = metrics.mean_abs_diff(res_correct.color_srgb, res_naive.color_srgb)
    max_diff = metrics.max_abs_diff(res_correct.color_srgb, res_naive.color_srgb)

    print("BAGIAN 2: HASIL RENDER QUAD LANTAI (600x450)")
    print(f"  PSNR (correct vs naive) : {psnr:.2f} dB")
    print(f"  SSIM (correct vs naive) : {ssim:.6f}")
    print(f"  Mean |diff|             : {mean_diff:.4f} / 255")
    print(f"  Max  |diff|             : {max_diff:.4f} / 255")
    print(f"  Saved naive render      : {naive_ppm}, {naive_png}")
    print(f"  Saved correct render    : {correct_ppm}, {correct_png}")
    print(f"  Saved diff preview      : {diff_png}")


def main():
    print_step_by_step_derivation()
    render_floor_comparison()


if __name__ == "__main__":
    main()
