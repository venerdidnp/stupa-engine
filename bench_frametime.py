"""
bench_frametime.py -- benchmark frame time: raster3d vs raytrace (Lab 3 no. 3).

Measures frame rendering time for raster3d and raytrace at:
  - 256 x 256
  - 512 x 512
Performs at least 5 runs for each combination, reporting:
  - Median, Min, Max frame time
  - Ratio (raster : raytrace)
Prints system information:
  - CPU model / processor
  - Triangle count of the loaded model
  - Python and NumPy versions

Flags:
  --shadow   Tests hard shadow if supported.
             (Note: hard shadow is not implemented in the baseline single-pass pipeline).
  --runs N   Number of benchmark runs (default 5).

DO NOT RUN before filling in PREDIKSI.md!
"""

import argparse
from pathlib import Path
import platform
import sys
import time
import numpy as np

import objio
import ppm_io
import raster3d
import raytrace

DATA_DIR = Path(__file__).resolve().parent / "data"


def benchmark_renderer(render_fn, runs=5):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        render_fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return {
        "median": float(np.median(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "runs": times,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark raster3d vs raytrace frame time.")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs (default 5)")
    parser.add_argument("--shadow", action="store_true", help="Enable directional hard shadow")
    args = parser.parse_args()

    obj_path = DATA_DIR / "temple.obj"
    tex_path = DATA_DIR / "relief.ppm"
    if not (obj_path.is_file() and tex_path.is_file()):
        import make_model
        make_model.build()

    mesh = objio.load_obj(obj_path)
    texture = raster3d.Texture(ppm_io.read_ppm(tex_path)[2])
    camera = raster3d.Camera()

    cpu_info = platform.processor() or platform.machine() or "Unknown CPU"
    os_info = f"{platform.system()} {platform.release()} ({platform.version()})"
    py_ver = sys.version.split()[0]
    np_ver = np.__version__
    tri_count = len(mesh.face_vertex_indices)

    print("=========================================================================")
    print("BENCHMARK FRAME TIME: RASTERISATION (raster3d) vs RAY TRACING (raytrace)")
    print("=========================================================================")
    print(f"CPU               : {cpu_info}")
    print(f"OS                : {os_info}")
    print(f"Python            : {py_ver}")
    print(f"NumPy             : {np_ver}")
    print(f"Jumlah Segitiga   : {tri_count}")
    print(f"Jumlah Iterasi    : {args.runs} runs per kombinasi")
    if args.shadow:
        print("Flag --shadow     : shadow tidak diimplementasi pada pipeline dasar")
    else:
        print("Flag --shadow     : off")
    print("=========================================================================\n")

    resolutions = [(256, 256), (512, 512)]
    results = {}

    for w, h in resolutions:
        print(f"Mengukur resolusi {w}x{h} ({args.runs} kali)...")
        cfg = raster3d.RenderSettings(width=w, height=h)

        # 1. Rasterizer
        raster_fn = lambda: raster3d.render_scene(mesh, texture, cfg, camera)
        raster_stats = benchmark_renderer(raster_fn, runs=args.runs)

        # 2. Ray tracer
        ray_fn = lambda: raytrace.ray_trace(mesh, texture, cfg, camera)
        ray_stats = benchmark_renderer(ray_fn, runs=args.runs)

        ratio = ray_stats["median"] / raster_stats["median"] if raster_stats["median"] > 0 else 0.0

        results[(w, h)] = {
            "raster": raster_stats,
            "ray": ray_stats,
            "ratio": ratio,
        }

    print("\n" + "=" * 73)
    print("HASIL PENGUKURAN (Waktu dalam detik)")
    print("=" * 73)
    print(f"{'Resolusi':<12} | {'Renderer':<10} | {'Median (s)':<12} | {'Min (s)':<10} | {'Max (s)':<10} | {'Rasio (Ray/Raster)':<18}")
    print("-" * 73)

    for (w, h), data in results.items():
        res_str = f"{w}x{h}"
        rst = data["raster"]
        ray = data["ray"]
        ratio_str = f"{data['ratio']:.2f}x"

        print(f"{res_str:<12} | {'raster3d':<10} | {rst['median']:<12.4f} | {rst['min']:<10.4f} | {rst['max']:<10.4f} | {'1.00x':<18}")
        print(f"{'':<12} | {'raytrace':<10} | {ray['median']:<12.4f} | {ray['min']:<10.4f} | {ray['max']:<10.4f} | {ratio_str:<18}")
        print("-" * 73)

    print("=========================================================================\n")


if __name__ == "__main__":
    main()
