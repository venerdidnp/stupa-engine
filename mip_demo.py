"""
mip_demo.py -- demonstrates and measures the effect of mipmapping (Lab 2 no. 4).

1. Renders temple from a distant camera (texture minification) with and without mipmaps.
2. Measures flicker (mean |diff|) between two slightly shifted frames (0.01 camera shift)
   for both modes.
3. Measures high-frequency energy (mean |Laplacian|) for both images.
4. Saves images, difference images, and reports metrics to out/mip_report.txt.

Run: python mip_demo.py
"""

from pathlib import Path
import numpy as np
from PIL import Image

import metrics
import objio
import ppm_io
import raster3d
from raster3d import Camera, RenderSettings

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "out"


def compute_laplacian_energy(img_rgb):
    """Compute mean |Laplacian| energy of an RGB image using pure NumPy."""
    # Convert to grayscale float [0, 255]
    img = img_rgb.astype(np.float64)
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    # Standard 4-neighbor discrete Laplacian: L(x, y) = up + down + left + right - 4 * center
    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    return float(np.mean(np.abs(lap)))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    obj_path = DATA_DIR / "temple.obj"
    tex_path = DATA_DIR / "relief.ppm"
    if not (obj_path.is_file() and tex_path.is_file()):
        import make_model
        make_model.build()

    mesh = objio.load_obj(obj_path)
    texture = raster3d.Texture(ppm_io.read_ppm(tex_path)[2])

    # Far camera: scale eye distance by 3x away from target (0, 4.6, 0)
    # Default: eye=(11.0, 8.0, 11.0), target=(0.0, 4.6, 0.0)
    # eye - target = (11.0, 3.4, 11.0)
    # far eye = target + 3.0 * (eye - target) = (33.0, 14.8, 33.0)
    target = np.array([0.0, 4.6, 0.0])
    base_eye = np.array([11.0, 8.0, 11.0])
    far_eye1 = target + 3.0 * (base_eye - target)
    far_eye2 = far_eye1 + np.array([0.01, 0.0, 0.0])

    cam_far1 = Camera(
        eye=tuple(far_eye1),
        target=tuple(target),
        up=(0.0, 1.0, 0.0),
        fovy_deg=50.0,
        near=1.0,
        far=200.0,
    )
    cam_far2 = Camera(
        eye=tuple(far_eye2),
        target=tuple(target),
        up=(0.0, 1.0, 0.0),
        fovy_deg=50.0,
        near=1.0,
        far=200.0,
    )

    # Common settings: 800x600, bilinear sampling enabled
    cfg_nomip = RenderSettings(width=800, height=600, use_texture=True, bilinear=True, mipmaps=False)
    cfg_mip = RenderSettings(width=800, height=600, use_texture=True, bilinear=True, mipmaps=True)

    # Render without mipmaps: frame 1 and shifted frame 2
    res_nomip1 = raster3d.render_scene(mesh, texture, cfg_nomip, cam_far1)
    res_nomip2 = raster3d.render_scene(mesh, texture, cfg_nomip, cam_far2)

    # Render with mipmaps: frame 1 and shifted frame 2
    res_mip1 = raster3d.render_scene(mesh, texture, cfg_mip, cam_far1)
    res_mip2 = raster3d.render_scene(mesh, texture, cfg_mip, cam_far2)

    # Save images
    ppm_io.write_ppm(OUT_DIR / "mip_without.ppm", res_nomip1.color_srgb)
    Image.fromarray(res_nomip1.color_srgb).save(OUT_DIR / "mip_without.png")

    ppm_io.write_ppm(OUT_DIR / "mip_with.ppm", res_mip1.color_srgb)
    Image.fromarray(res_mip1.color_srgb).save(OUT_DIR / "mip_with.png")

    # Difference images
    diff_modes = metrics.diff_image(res_mip1.color_srgb, res_nomip1.color_srgb, gain=4.0)
    Image.fromarray(diff_modes).save(OUT_DIR / "mip_diff_modes.png")

    diff_flicker_nomip = metrics.diff_image(res_nomip1.color_srgb, res_nomip2.color_srgb, gain=10.0)
    Image.fromarray(diff_flicker_nomip).save(OUT_DIR / "mip_diff_flicker_without.png")

    diff_flicker_mip = metrics.diff_image(res_mip1.color_srgb, res_mip2.color_srgb, gain=10.0)
    Image.fromarray(diff_flicker_mip).save(OUT_DIR / "mip_diff_flicker_with.png")

    # Calculations
    # 1. Flicker (mean |diff| between shifted camera positions)
    flicker_nomip = metrics.mean_abs_diff(res_nomip1.color_srgb, res_nomip2.color_srgb)
    flicker_mip = metrics.mean_abs_diff(res_mip1.color_srgb, res_mip2.color_srgb)
    flicker_reduction_pct = ((flicker_nomip - flicker_mip) / flicker_nomip) * 100.0 if flicker_nomip > 0 else 0.0

    # 2. High frequency energy (mean |Laplacian|)
    energy_nomip = compute_laplacian_energy(res_nomip1.color_srgb)
    energy_mip = compute_laplacian_energy(res_mip1.color_srgb)
    energy_reduction_pct = ((energy_nomip - energy_mip) / energy_nomip) * 100.0 if energy_nomip > 0 else 0.0

    # 3. Direct comparison between with and without mipmap
    psnr_modes = metrics.psnr(res_mip1.color_srgb, res_nomip1.color_srgb)
    ssim_modes = metrics.ssim(res_mip1.color_srgb, res_nomip1.color_srgb)
    mean_diff_modes = metrics.mean_abs_diff(res_mip1.color_srgb, res_nomip1.color_srgb)

    lines = []
    lines.append("=== LAPORAN PENGUKURAN BUKTI MIPMAP (Lab 2 no. 4) ===")
    lines.append("")
    lines.append("1. SETUP KAMERA & MINIFIKASI")
    lines.append(f"   Kamera target                   : {list(target)}")
    lines.append(f"   Kamera eye frame 1 (jarak jauh) : [{far_eye1[0]:.2f}, {far_eye1[1]:.2f}, {far_eye1[2]:.2f}]")
    lines.append(f"   Kamera eye frame 2 (geser 0.01) : [{far_eye2[0]:.2f}, {far_eye2[1]:.2f}, {far_eye2[2]:.2f}]")
    lines.append(f"   Ukuran frame                    : 800 x 600")
    lines.append("")
    lines.append("2. PENGUKURAN KELAP-KELIP (FLICKER PADA PERGESERAN KAMERA 0.01 UNIT)")
    lines.append(f"   Mean |diff| tanpa mipmap        : {flicker_nomip:.6f} / 255")
    lines.append(f"   Mean |diff| dengan mipmap       : {flicker_mip:.6f} / 255")
    lines.append(f"   Penurunan kelap-kelip           : {flicker_reduction_pct:.2f}%")
    lines.append("")
    lines.append("3. ENERGI FREKUENSI TINGGI (MEAN |LAPLACIAN|)")
    lines.append(f"   Energi tanpa mipmap             : {energy_nomip:.6f}")
    lines.append(f"   Energi dengan mipmap            : {energy_mip:.6f}")
    lines.append(f"   Penurunan frekuensi tinggi      : {energy_reduction_pct:.2f}%")
    lines.append("")
    lines.append("4. PERBANDINGAN LANGSUNG (DENGAN MIPMAP vs TANPA MIPMAP)")
    lines.append(f"   PSNR                            : {psnr_modes:.2f} dB")
    lines.append(f"   SSIM                            : {ssim_modes:.6f}")
    lines.append(f"   Mean |diff|                     : {mean_diff_modes:.4f} / 255")
    lines.append("")
    lines.append("5. FILE BUKTI")
    lines.append("   - out/mip_without.ppm / out/mip_without.png")
    lines.append("   - out/mip_with.ppm / out/mip_with.png")
    lines.append("   - out/mip_diff_modes.png (gain 4x)")
    lines.append("   - out/mip_diff_flicker_without.png (gain 10x)")
    lines.append("   - out/mip_diff_flicker_with.png (gain 10x)")

    report_text = "\n".join(lines) + "\n"
    report_file = OUT_DIR / "mip_report.txt"
    report_file.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"Laporan disimpan di: {report_file.resolve()}")


if __name__ == "__main__":
    main()
