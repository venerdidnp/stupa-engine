"""
webshot.py -- capture the Three.js viewer headlessly and score it vs NumPy.

The plan's phase 8 expects the WebGL frame to land at "about 30 dB" PSNR
against the NumPy render, with differences on silhouettes (MSAA) rather
than in the interior. Checking that needs a browser; this script provides
it without any npm packages:

  1. renders the reference frame with raster3d at the same size and the
     pinned camera (out/webgl_ref.ppm, plus a PNG preview);
  2. serves the repo over http://127.0.0.1:<port> (the viewer fetches
     data/ and the pinned CDN modules; file:// would be blocked);
  3. launches an installed Chrome/Edge in headless mode with a fixed
     512x512 window and --screenshot (WebGL runs on SwiftShader when no
     GPU is available, hence --enable-unsafe-swiftshader);
  4. converts the screenshot PNG to out/webgl_shot.ppm (Pillow is used
     for this conversion only, as everywhere in this project);
  5. compares the two frames with compare.py and writes the amplified
     difference image out/webgl_diff.ppm.

Exit codes: 0 = PSNR at or above --min-psnr, 1 = below, 2 = setup
problem (missing assets, no browser, no screenshot produced).

Usage:  python webshot.py [--browser PATH] [--size 512] [--min-psnr 25]
"""

import argparse
import subprocess
import sys
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import compare
import objio
import ppm_io
import raster3d
from raster3d import RenderSettings, Texture, render_scene, write_png_preview


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"

# Candidate browsers, Windows paths first (this repo is developed there),
# then the usual Linux and macOS locations.
BROWSER_CANDIDATES = (
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/chromium"),
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
)


class _QuietHandler(SimpleHTTPRequestHandler):
    """Serve the repo without printing a log line per request."""

    def log_message(self, *args):
        pass


def find_browser(explicit):
    """Return the browser executable to use, or None when none exists.

    Input: explicit -- user-provided path or None.
    Output: Path or None.
    """
    if explicit is not None:
        path = Path(explicit)
        return path if path.is_file() else None
    for path in BROWSER_CANDIDATES:
        if path.is_file():
            return path
    return None


def render_reference(size):
    """Render the NumPy frame the viewer must match. Output: PPM path."""
    mesh = objio.load_obj(DATA_DIR / "temple.obj")
    texture = Texture(ppm_io.read_ppm(DATA_DIR / "relief.ppm")[2])
    settings = RenderSettings(width=size, height=size)
    result = render_scene(mesh, texture, settings)
    path = OUT_DIR / "webgl_ref.ppm"
    ppm_io.write_ppm(path, result.color_srgb)
    write_png_preview(path)
    return path


def capture(browser, size, timeout):
    """Screenshot the viewer headlessly. Output: PNG path on disk.

    Inputs: browser -- executable path; size -- square window in pixels;
            timeout -- seconds to wait for the browser to finish.
    Output: Path of the written PNG (may not exist when the browser
            failed; main() checks).
    """
    handler = partial(_QuietHandler, directory=str(ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    png_path = OUT_DIR / "webgl_shot.png"
    if png_path.exists():
        png_path.unlink()

    try:
        with tempfile.TemporaryDirectory() as profile:
            # A throwaway profile avoids touching the user's browser state
            # (and avoids "profile in use" lock errors).
            args = [
                str(browser),
                "--headless=new",
                "--disable-gpu",
                "--enable-unsafe-swiftshader",  # software WebGL, no GPU needed
                "--hide-scrollbars",
                "--force-device-scale-factor=1",
                "--no-first-run",
                "--no-default-browser-check",
                f"--user-data-dir={profile}",
                f"--window-size={size},{size}",
                # Virtual time fast-forwards timers/rAF so the scene is
                # fully loaded and rendered before the screenshot.
                "--virtual-time-budget=20000",
                f"--screenshot={png_path}",
                f"http://127.0.0.1:{port}/web/?auto=1",
            ]
            subprocess.run(args, timeout=timeout, check=False, capture_output=True)
    finally:
        server.shutdown()
    return png_path


def png_to_ppm(png_path, size):
    """Convert the screenshot PNG to a PPM the comparator understands.

    Inputs: png_path -- Path of the PNG; size -- expected square size.
    Output: Path of the written PPM.
    Why RGB: browser screenshots often carry an (opaque) alpha channel;
    the pipeline works with 3-channel images only.
    """
    from PIL import Image  # conversion only, see the module docstring

    img = Image.open(png_path).convert("RGB")
    if img.size != (size, size):
        raise SystemExit(f"webshot: expected a {size}x{size} shot, got {img.size}")
    path = OUT_DIR / "webgl_shot.ppm"
    ppm_io.write_ppm(path, np.asarray(img))
    return path


def main(argv=None):
    """CLI: reference render, headless capture, compare, diff image."""
    parser = argparse.ArgumentParser(
        description="Headless WebGL capture scored against NumPy (phase 8)."
    )
    parser.add_argument("--browser", default=None, help="path to Chrome/Edge")
    parser.add_argument("--size", type=int, default=512, help="square size (default 512)")
    parser.add_argument(
        "--min-psnr", type=float, default=25.0, help="exit 1 below this (default 25)"
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="browser timeout")
    args = parser.parse_args(argv)

    required = [
        DATA_DIR / "temple.obj",
        DATA_DIR / "relief.ppm",
        DATA_DIR / "relief.png",
        ROOT / "web" / "data" / "temple.obj",
        ROOT / "web" / "data" / "relief.png",
        ROOT / "web" / "index.html",
        ROOT / "web" / "viewer.js",
    ]
    for path in required:
        if not path.is_file():
            hint = " -- run python make_model.py" if path.parent.name == "data" else ""
            print(f"webshot: missing {path}{hint}")
            return 2

    browser = find_browser(args.browser)
    if browser is None:
        print("webshot: no Chrome/Edge found (pass --browser PATH)")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref_path = render_reference(args.size)

    print(f"webshot: capturing {args.size}x{args.size} with {browser.name} ...")
    try:
        png_path = capture(browser, args.size, args.timeout)
    except subprocess.TimeoutExpired:
        print(f"webshot: the browser did not finish within {args.timeout:g} s")
        return 2
    if not png_path.is_file():
        print("webshot: no screenshot was produced (is WebGL available?)")
        return 2
    shot_path = png_to_ppm(png_path, args.size)

    _, _, ref_img = ppm_io.read_ppm(ref_path)
    _, _, shot_img = ppm_io.read_ppm(shot_path)
    result = compare.compare_images(ref_img, shot_img)

    diff_path = OUT_DIR / "webgl_diff.ppm"
    ppm_io.write_ppm(diff_path, result["diff"])
    write_png_preview(diff_path)

    row, col, chan = result["worst"]
    psnr_text = "inf" if np.isinf(result["psnr"]) else f"{result['psnr']:.2f}"
    print(f"webshot: {shot_path.name} vs {ref_path.name} ({args.size}x{args.size})")
    print(f"  PSNR             : {psnr_text} dB")
    print(f"  SSIM (global)    : {result['ssim']:.6f}")
    print(
        f"  max  |diff|      : {result['max_abs']:.6f} "
        f"at row {row}, col {col}, channel {chan}"
    )
    print(
        f"  different pixels : {result['differing_pixels']} of {args.size * args.size} "
        f"({100.0 * result['differing_fraction']:.2f}%)"
    )
    print(f"  diff image       : {diff_path} (+ .png preview)")

    if result["psnr"] >= args.min_psnr:
        print(f"  PASS: PSNR {psnr_text} dB >= {args.min_psnr:g} dB")
        return 0
    print(f"  FAIL: PSNR {result['psnr']:.2f} dB < {args.min_psnr:g} dB")
    return 1


if __name__ == "__main__":
    sys.exit(main())
