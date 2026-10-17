"""
pitfall_demo.py -- demonstrates and measures two common rendering pitfalls (Lab 2 no. 5):
1. Back-face culling (cull_backface=False vs True):
   - Measures fragments passing z-test (written pixels), triangles culled,
     percent change, and median render time over 5 runs.
   - Verifies whether the resulting images of the closed model are identical.
2. Normal matrix inverse-transpose under non-uniform scaling:
   - Compares Lambert shading with direct M transformation vs inverse-transpose of M.
   - Saves out/normal_wrong.ppm/.png, out/normal_correct.ppm/.png, and out/normal_diff.ppm/.png.
   - Reports PSNR, SSIM, and mean |diff|.
Writes full report to out/pitfall_report.txt.
"""

from pathlib import Path
import time
import numpy as np
from PIL import Image

import compare
import metrics
import objio
import ppm_io
import raster3d
from transforms import scale

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "out"


def measure_culling(mesh, texture, camera):
    cfg_cull = raster3d.RenderSettings(cull_backface=True)
    cfg_nocull = raster3d.RenderSettings(cull_backface=False)

    times_cull = []
    times_nocull = []
    N_RUNS = 5

    res_cull = None
    res_nocull = None

    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        res_cull = raster3d.render_scene(mesh, texture, cfg_cull, camera)
        times_cull.append(time.perf_counter() - t0)

    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        res_nocull = raster3d.render_scene(mesh, texture, cfg_nocull, camera)
        times_nocull.append(time.perf_counter() - t0)

    med_time_cull = float(np.median(times_cull))
    med_time_nocull = float(np.median(times_nocull))

    frag_cull = res_cull.stats["pixels"]
    frag_nocull = res_nocull.stats["pixels"]
    culled_tri = res_cull.stats["culled"]
    submitted_tri = res_cull.stats["submitted"]

    frag_diff = frag_nocull - frag_cull
    pct_frag_reduction = (frag_diff / frag_nocull) * 100.0 if frag_nocull > 0 else 0.0
    time_diff = med_time_nocull - med_time_cull
    pct_time_reduction = (time_diff / med_time_nocull) * 100.0 if med_time_nocull > 0 else 0.0

    comp = compare.compare_images(res_cull.color_srgb, res_nocull.color_srgb)

    return {
        "runs": N_RUNS,
        "submitted_tri": submitted_tri,
        "culled_tri_cull": culled_tri,
        "culled_tri_nocull": res_nocull.stats["culled"],
        "frag_cull": frag_cull,
        "frag_nocull": frag_nocull,
        "pct_frag_reduction": pct_frag_reduction,
        "time_cull_median_s": med_time_cull,
        "time_nocull_median_s": med_time_nocull,
        "pct_time_reduction": pct_time_reduction,
        "image_comparison": comp,
    }


def measure_normal_matrix(mesh, texture, camera):
    M_aligned = scale(1.0, 3.0, 1.0)
    from transforms import rot_z
    M_tilted = scale(1.0, 3.0, 1.0) @ rot_z(30.0)

    # Use flat Lambert shading with flat color to highlight normal differences
    cfg_wrong = raster3d.RenderSettings(
        lambert=True,
        use_texture=False,
        normal_matrix_mode="wrong",
    )
    cfg_correct = raster3d.RenderSettings(
        lambert=True,
        use_texture=False,
        normal_matrix_mode="correct",
    )

    # 1. Pure scale(1,3,1) on axis-aligned model
    res_wrong_aligned = raster3d.render_scene(mesh, texture, cfg_wrong, camera, m_matrix=M_aligned)
    res_corr_aligned = raster3d.render_scene(mesh, texture, cfg_correct, camera, m_matrix=M_aligned)
    psnr_aligned = metrics.psnr(res_corr_aligned.color_srgb, res_wrong_aligned.color_srgb)
    ssim_aligned = metrics.ssim(res_corr_aligned.color_srgb, res_wrong_aligned.color_srgb)
    mean_diff_aligned = metrics.mean_abs_diff(res_corr_aligned.color_srgb, res_wrong_aligned.color_srgb)
    max_diff_aligned = metrics.max_abs_diff(res_corr_aligned.color_srgb, res_wrong_aligned.color_srgb)

    # 2. Scale(1,3,1) with tilted orientation (rot_z 30) so normals are not axis-aligned
    res_wrong = raster3d.render_scene(mesh, texture, cfg_wrong, camera, m_matrix=M_tilted)
    res_correct = raster3d.render_scene(mesh, texture, cfg_correct, camera, m_matrix=M_tilted)

    wrong_ppm = OUT_DIR / "normal_wrong.ppm"
    wrong_png = OUT_DIR / "normal_wrong.png"
    correct_ppm = OUT_DIR / "normal_correct.ppm"
    correct_png = OUT_DIR / "normal_correct.png"
    diff_ppm = OUT_DIR / "normal_diff.ppm"
    diff_png = OUT_DIR / "normal_diff.png"

    ppm_io.write_ppm(wrong_ppm, res_wrong.color_srgb)
    Image.fromarray(res_wrong.color_srgb).save(wrong_png)

    ppm_io.write_ppm(correct_ppm, res_correct.color_srgb)
    Image.fromarray(res_correct.color_srgb).save(correct_png)

    diff_img = metrics.diff_image(res_correct.color_srgb, res_wrong.color_srgb, gain=4.0)
    ppm_io.write_ppm(diff_ppm, diff_img)
    Image.fromarray(diff_img).save(diff_png)

    psnr = metrics.psnr(res_correct.color_srgb, res_wrong.color_srgb)
    ssim = metrics.ssim(res_correct.color_srgb, res_wrong.color_srgb)
    mean_diff = metrics.mean_abs_diff(res_correct.color_srgb, res_wrong.color_srgb)
    max_diff = metrics.max_abs_diff(res_correct.color_srgb, res_wrong.color_srgb)

    return {
        "aligned": {
            "scale": [1.0, 3.0, 1.0],
            "psnr": psnr_aligned,
            "ssim": ssim_aligned,
            "mean_diff": mean_diff_aligned,
            "max_diff": max_diff_aligned,
        },
        "tilted": {
            "transform": "scale(1, 3, 1) @ rot_z(30)",
            "psnr": psnr,
            "ssim": ssim,
            "mean_diff": mean_diff,
            "max_diff": max_diff,
            "wrong_files": (str(wrong_ppm), str(wrong_png)),
            "correct_files": (str(correct_ppm), str(correct_png)),
            "diff_files": (str(diff_ppm), str(diff_png)),
        },
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    obj_path = DATA_DIR / "temple.obj"
    tex_path = DATA_DIR / "relief.ppm"
    if not (obj_path.is_file() and tex_path.is_file()):
        import make_model
        make_model.build()

    mesh = objio.load_obj(obj_path)
    texture = raster3d.Texture(ppm_io.read_ppm(tex_path)[2])
    camera = raster3d.Camera()

    cull_results = measure_culling(mesh, texture, camera)
    normal_results = measure_normal_matrix(mesh, texture, camera)

    lines = []
    lines.append("=== LAPORAN PENGUKURAN PITFALL GRAFIKA (Lab 2 no. 5) ===")
    lines.append("")
    lines.append("1. PENGUKURAN BACK-FACE CULLING (Kamera Default, 800x600)")
    lines.append(f"   Jumlah segitiga disubmit          : {cull_results['submitted_tri']}")
    lines.append(f"   Segitiga di-cull (cull=True)     : {cull_results['culled_tri_cull']}")
    lines.append(f"   Segitiga di-cull (cull=False)    : {cull_results['culled_tri_nocull']}")
    lines.append(f"   Fragment lolos z-test (cull=True) : {cull_results['frag_cull']}")
    lines.append(f"   Fragment lolos z-test (cull=False): {cull_results['frag_nocull']}")
    lines.append(f"   Penurunan jumlah fragment         : {cull_results['pct_frag_reduction']:.2f}%")
    lines.append(f"   Waktu render median (cull=True)   : {cull_results['time_cull_median_s']:.4f} s ({cull_results['runs']} runs)")
    lines.append(f"   Waktu render median (cull=False)  : {cull_results['time_nocull_median_s']:.4f} s ({cull_results['runs']} runs)")
    lines.append(f"   Penurunan waktu render            : {cull_results['pct_time_reduction']:.2f}%")
    lines.append("   Perbandingan citra (cull=True vs cull=False):")
    lines.append(f"     PSNR                           : {cull_results['image_comparison']['psnr']:.2f} dB")
    lines.append(f"     SSIM                           : {cull_results['image_comparison']['ssim']:.6f}")
    lines.append(f"     Pixel berbeda                  : {cull_results['image_comparison']['differing_pixels']} / {800*600}")
    lines.append("")
    lines.append("2. PENGUKURAN NORMAL MATRIX INVERSE-TRANSPOSE")
    lines.append("   (a) Kasus Axis-Aligned: scale(1, 3, 1)")
    lines.append(f"       PSNR                         : {normal_results['aligned']['psnr']:.2f} dB")
    lines.append(f"       SSIM                         : {normal_results['aligned']['ssim']:.6f}")
    lines.append(f"       Mean |diff|                  : {normal_results['aligned']['mean_diff']:.4f} / 255")
    lines.append(f"       Max  |diff|                  : {normal_results['aligned']['max_diff']:.4f} / 255")
    lines.append("       Catatan: Model candi tersusun dari kotak sejajar sumbu (axis-aligned).")
    lines.append("       Normal permukaan yang sejajar sumbu kartesius tidak berubah arah ternormalisasi")
    lines.append("       ketika diskalakan tepat sepanjang sumbu tersebut.")
    lines.append("   (b) Kasus Permukaan Tilted: scale(1, 3, 1) @ rot_z(30)")
    lines.append(f"       PSNR (correct vs wrong)      : {normal_results['tilted']['psnr']:.2f} dB")
    lines.append(f"       SSIM (correct vs wrong)      : {normal_results['tilted']['ssim']:.6f}")
    lines.append(f"       Mean |diff|                  : {normal_results['tilted']['mean_diff']:.4f} / 255")
    lines.append(f"       Max  |diff|                  : {normal_results['tilted']['max_diff']:.4f} / 255")
    lines.append(f"       Output normal_wrong          : out/normal_wrong.ppm, out/normal_wrong.png")
    lines.append(f"       Output normal_correct        : out/normal_correct.ppm, out/normal_correct.png")
    lines.append(f"       Output normal_diff (gain 4x) : out/normal_diff.ppm, out/normal_diff.png")

    report_text = "\n".join(lines) + "\n"
    report_path = OUT_DIR / "pitfall_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"Laporan disimpan di: {report_path.resolve()}")


if __name__ == "__main__":
    main()
