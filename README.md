# stupa-engine

A from-scratch 3D renderer (Python + NumPy) and a Three.js viewer for a
low-poly textured temple (candi).

The pipeline uses no graphics libraries: matrices, clipping, the z-buffer,
texture sampling and the shading model are all written by hand. Every
stage is checked against hand-computed reference numbers in `tests.py`,
and the two renderers (raster and ray) are required to agree.

## Quick start

```
pip install -r requirements.txt
python make_model.py        # writes data/ (and web/data/ for the viewer)
python tests.py             # all numeric checks, exit code 0 on success
```

Then render and compare:

```
python raster3d.py          # out/temple_raster.ppm (+ depth map, PNG)
python raster3d.py --steps  # one frame per build stage, out/stepN_*.ppm
python raytrace.py          # out/temple_ray.ppm (independent ray tracer)
python compare.py out/temple_raster.ppm out/temple_ray.ppm --min-psnr 60
python worked_example.py    # one vertex through every space, printed
```

## The bug showcase

```
python broken_scene.py --bug inside-out   # reversed winding: see the inside
python broken_scene.py --bug zfight       # near = 1e-3: a 16-bit depth buffer
                                          # cannot order the scene any more
python broken_scene.py --bug stretched    # V @ P instead of P @ V
```

Each run renders the broken frame next to the correct one, prints hard
evidence (counters, depth histograms, matrix printouts, flicker counts)
and writes `out/broken_<bug>.ppm` plus a diff image.

## Live viewer

```
python make_model.py
npx serve web                      # or: python -m http.server --directory web
```

Open `http://localhost:3000/` (or `http://localhost:8000/`), orbit with
the mouse. The viewer is configured to match the NumPy renderer exactly
(same camera, light, UV convention via `flipY = false`, sRGB output, no
tone mapping). "Export PPM" downloads the current frame as `webgl.ppm`.

`python webshot.py` does the same capture headlessly (an installed
Chrome/Edge is required) and scores it against a fresh NumPy render;
`python compare.py` can diff the exported file against
`out/webgl_ref.ppm` the same way.

## Deploying

`.github/workflows/pages.yml` rebuilds the assets on every push to
`main`, runs the checks, and publishes `web/` to GitHub Pages. One-time
setup: repository Settings -> Pages -> Source: **GitHub Actions**.

## What is in here

| file | what it does |
| --- | --- |
| `transforms.py` | 4x4 matrices, `perspective`, `look_at`, `viewport`, `project` |
| `raster3d.py` | the rasterizer, built in 9 stages; also RGB/depth writers |
| `raytrace.py` | pinhole rays + Moller-Trumbore against every triangle |
| `worked_example.py` | traces one vertex through every coordinate space |
| `broken_scene.py` | the three classic bugs, with evidence and fixes |
| `metrics.py`, `compare.py` | PSNR, global SSIM, diff images, comparison CLI |
| `make_model.py` | generates the stepped temple .obj + texture, deterministically |
| `objio.py`, `ppm_io.py` | minimal OBJ loader and P6 PPM reader/writer |
| `web/` | Three.js viewer (index.html, style.css, viewer.js) |
| `webshot.py` | headless browser capture scored against the NumPy frame |
| `tests.py` | all numeric checks, one section per build phase |

## Status

- `python tests.py`: all checks pass (see the count printed at the end)
- raster vs ray tracer: byte-identical frames (PSNR = inf), depth to 3e-16
- browser (WebGL) vs NumPy: ~40 dB PSNR, differences only at silhouette
  pixels (MSAA) -- the interior matches exactly
- three broken scenes reproduce, each with a verified fix

## Tested on

- Windows 11 Home Single Language (10.0.26200), Python 3.11.9,
  numpy 2.3.5, Pillow 12.2.0
- headless capture (`python webshot.py`) with Chrome 155.0.8059.79
