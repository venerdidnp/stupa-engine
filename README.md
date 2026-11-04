# stupa-engine (Module 3 &mdash; 3D Transforms, Projection, Texture & Real-Time)

Course: *Grafika Komputer / Computer Graphics* (`25MP7716G030`), S1 Informatika.

A from-scratch 3D renderer (Python + NumPy) and a Three.js viewer for a
textured low-poly temple (candi), aligned with the official reference codebase.

**Rule:** The transform stack, the z-buffer rasteriser, and the ray tracer are
**NumPy only**. Pillow is optional (for display conversion / references only).
The real-time renderer (`web/`) is Three.js and serves as the GitHub Pages deliverable.

**Conventions:** Pinned at the top of `transforms.py`:
- column vectors, pre-multiply (`p' = M @ p`), chains right-to-left
- right-handed; camera looks down `-z`
- NDC `[-1, 1]^3`, near at `z = -1`, far at `z = +1`
- viewport flips `y` (screen y points down)

---

## Quick Start (Official Module 3 Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate deterministic models & texture
python make_model.py
# Writes:
#   data/pusaka_model.obj  (180 tris, baseline reference model)
#   data/mymodel.obj       (312 tris, Lab 1 custom candi with corner stupas & stairs)
#   data/relief.ppm        (256x256 procedural sandstone texture)
# Copies assets to web/data/ for standalone web hosting.

# 3. Run all numeric checks
python tests.py
# Expected: ALL CHECKS PASSED (8 ok)

# 4. Step-by-step vertex space trace (Worked Example Step 1)
python worked_example.py

# 5. Rasterize the temple (from scratch)
python raster3d.py --model data/pusaka_model.obj --w 512 --h 512
python raster3d.py --demo-triangle --w 64 --h 64
python raster3d.py --demo-overlap --w 64 --h 64

# 6. Ray trace the temple (independent pinhole + Moller-Trumbore)
python raytrace.py --model data/pusaka_model.obj --w 256 --h 256

# 7. Compare rasterizer vs ray tracer
python compare.py out/temple_raster.ppm out/temple_rt.ppm
# Expected: ~77 dB PSNR (byte-identical except ~10 edge pixels)

# 8. Reproduce and fix the 3 classic bugs (Lab 2)
python broken_scene.py --all
# (or individually: --bug inside-out, --bug zfight, --bug stretched)

# 9. Serve the interactive Three.js real-time viewer
npx serve .
# Open http://localhost:3000/web/ to orbit the temple, toggle wireframe, or export PPM.
```

---

## Unit Table

| File | Job / Description | Reproduces / Maps To |
|---|---|---|
| `transforms.py` | mat4 T/S/R, `perspective`, `look_at`, `project`, `viewport`, `perspective_divide_2d` | E1, E2, E3 |
| `objio.py` | Minimal OBJ loader (`v`/`vt`/`f`, fan triangulation, `v` flip) + `save_obj` | Case Study input |
| `ppm_io.py` | From-scratch P5 / P6 Netpbm reader/writer | Base I/O |
| `srgb.py` | Real sRGB curve and gamma-2 transfer functions | E4, E5 (Modul 1/3) |
| `make_model.py` | Generates `pusaka_model.obj` (180 tris) and `mymodel.obj` (312 tris) + `relief.ppm` | Case Study & Lab 1 |
| `raster3d.py` | MVP &rarr; near-clip &rarr; w-divide &rarr; z-buffer barycentric raster &rarr; Lambert | Worked Example Steps 2-3 |
| `raytrace.py` | Pinhole rays + M&ouml;ller-Trumbore, nearest hit, identical shading | Worked Example Step 4 |
| `worked_example.py` | Running point (2,3,4) through Object &rarr; World &rarr; View &rarr; Clip &rarr; NDC &rarr; Screen | Worked Example Step 1 |
| `broken_scene.py` | Diagnoses and fixes `inside-out`, `zfight`, `stretched` with hard evidence | Lab 2 diagnose-and-fix |
| `metrics.py` / `compare.py` | PSNR, global SSIM, mean/max diff, amplified difference image | Step 4, Case Study |
| `tests.py` | Invariants E1-E5, projection pipeline, 1-triangle render, reference frame match | MASTER Definition of Done |
| `web/index.html`, `viewer.js` | Real-time Three.js viewer with model selector, OrbitControls, and PPM export | Lab 3 deliverable |

---

## Lab 1, 2, and 3 Tasks & Extras

- **Lab 1: Custom Model & Coordinate Trace**
  - Custom model `data/mymodel.obj`: 26 axis-aligned boxes (312 triangles, 208 vertices &ge; 200 triangles) featuring 3-tiered base, sanctuary body, cornice, 7-step roof, top spire, 8 corner stupas, and 3 entrance steps.
  - `lab1_trace.py`: Traces vertex 0 through all coordinate spaces and computes column readings of matrices $M$, $V$, and $V \cdot M$ (output saved in `KUNCI_LAB1.txt`).
  - `uv_orientation_check.py`: Verifies UV orientation and handedness with an asymmetric letter 'F' texture.

- **Lab 2: Pitfalls, Perspective UV & Mipmapping**
  - `e4_demo.py`: Demonstrates perspective-correct UV ($u=0.1$) vs naive linear ($u=0.5$) on a receding textured floor quad, saving diff images.
  - `mip_demo.py`: Evaluates a 4-level linear box-filter mip pyramid from a distant camera, demonstrating a 93% reduction in camera-motion flicker and 55% reduction in high-frequency aliasing energy.
  - `pitfall_demo.py`: Quantifies back-face culling efficiency (~37% time reduction, 18% fragment reduction) and demonstrates normal distortion under non-uniform scaling without the inverse-transpose normal matrix.

- **Lab 3: Real-Time WebGL & Benchmark**
  - `bench_frametime.py`: Benchmarks frame time of `raster3d` vs `raytrace` at 256&times;256 and 512&times;512 across multiple runs (results predicted in `PREDIKSI.md`).
  - `web/`: Real-time viewer with model selection (`pusaka_model.obj` vs `mymodel.obj`), toggle wireframe, and 512&times;512 PPM export.
  - `verify_clean.ps1`: Automated PowerShell script to verify a clean clone, virtualenv creation, tests, and web server response.

---

## Verified Status

| Item | Status | Date | Environment |
|---|---|---|---|
| `python make_model.py` | **180 tris & 312 tris written** | 2026-11-04 | Windows 11, Python 3.11.9, NumPy 2.3.5 |
| `python tests.py` | **ALL CHECKS PASSED (8 ok)** | 2026-11-04 | Windows 11, Python 3.11.9, NumPy 2.3.5 |
| `raster3d.py` vs `raytrace.py` | **PSNR 76.5 dB, SSIM 1.000** | 2026-11-04 | 256&times;256 frame |
| `broken_scene.py --all` | **inside-out (SSIM 0.790), zfight (2852 vs 184 px), stretched (SSIM 0.594)** | 2026-11-04 | Verified with PPM output |
| `lab1_trace.py`, `e4_demo.py`, `mip_demo.py`, `pitfall_demo.py` | **All run green** | 2026-11-04 | Verified |
| Clean clone verification | **Automated via verify_clean.ps1** | 2026-11-04 | PowerShell 5.1/7+ |
