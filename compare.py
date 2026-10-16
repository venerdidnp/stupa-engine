"""
compare.py -- compare two rendered frames and write a difference image.

This is the tool the build plan's definition of done calls for in phase 4:

    python compare.py out/temple_raster.ppm out/temple_ray.ppm --min-psnr 60

What it prints: PSNR, the global SSIM, the mean and max absolute
difference, how many pixels differ at all, and where the worst difference
sits. With --min-psnr the exit code is 1 when the measured PSNR is below
the bound (0 = passed, or no bound given).

What it writes: an amplified difference image, PPM plus a PNG preview.
Black means identical; --gain (default 4) scales the differences before
they are clipped at white.

The default diff path puts "diff_<A>_vs_<B>.ppm" next to the first input;
--diff overrides it.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import metrics
import ppm_io
from raster3d import write_png_preview  # shared PPM -> PNG converter (Pillow)


def compare_images(a, b, gain=4.0):
    """Measure two RGB images and build their amplified diff image.

    Inputs: a, b -- (H, W, 3) images, uint8 or float; gain -- diff gain.
    Output: dict with keys
      psnr, ssim, mean_abs, max_abs -- the metrics.metrics of the pair;
      differing_pixels   -- count of pixels where any channel differs;
      differing_fraction -- that count over all pixels;
      worst              -- (row, col, channel) of the max |difference|;
      diff               -- (H, W, 3) uint8 amplified difference image.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"compare: shape mismatch {a.shape} vs {b.shape}")

    abs_diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    row, col, chan = np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape)
    differing = np.any(a != b, axis=2)
    return {
        "psnr": metrics.psnr(a, b),
        "ssim": metrics.ssim(a, b),
        "mean_abs": metrics.mean_abs_diff(a, b),
        "max_abs": metrics.max_abs_diff(a, b),
        "differing_pixels": int(differing.sum()),
        "differing_fraction": float(differing.mean()),
        "worst": (int(row), int(col), int(chan)),
        "diff": metrics.diff_image(a, b, gain=gain),
    }


def _print_report(a_path, b_path, size, result, gain, diff_path):
    """Print the metric table for one comparison."""
    width, height = size
    psnr_text = "inf" if np.isinf(result["psnr"]) else f"{result['psnr']:.2f}"
    row, col, chan = result["worst"]
    print(f"compare: {a_path.name} vs {b_path.name} ({width}x{height})")
    print(f"  PSNR             : {psnr_text} dB")
    print(f"  SSIM (global)    : {result['ssim']:.6f}")
    print(f"  mean |diff|      : {result['mean_abs']:.6f} / 255")
    print(
        f"  max  |diff|      : {result['max_abs']:.6f} "
        f"at row {row}, col {col}, channel {chan}"
    )
    print(
        f"  different pixels : {result['differing_pixels']} of {width * height} "
        f"({100.0 * result['differing_fraction']:.6f}%)"
    )
    print(f"  diff image       : {diff_path} (.ppm + .png preview, gain {gain:g}x)")


def main(argv=None):
    """CLI: compare two PPM frames, print metrics, write the diff image."""
    parser = argparse.ArgumentParser(
        description="Compare two rendered PPM frames (phase 5)."
    )
    parser.add_argument("first", type=Path, help="first PPM (e.g. the rasterizer)")
    parser.add_argument("second", type=Path, help="second PPM (e.g. the ray tracer)")
    parser.add_argument(
        "--diff",
        type=Path,
        default=None,
        help="diff image path (default: next to the first input)",
    )
    parser.add_argument(
        "--gain", type=float, default=4.0, help="diff amplification (default 4)"
    )
    parser.add_argument(
        "--min-psnr",
        type=float,
        default=None,
        help="exit 1 when the measured PSNR is below this bound",
    )
    args = parser.parse_args(argv)

    for path in (args.first, args.second):
        if not path.is_file():
            print(f"compare: no such file: {path}")
            return 1

    w_a, h_a, img_a = ppm_io.read_ppm(args.first)
    w_b, h_b, img_b = ppm_io.read_ppm(args.second)
    if (w_a, h_a) != (w_b, h_b):
        print(f"compare: size mismatch {w_a}x{h_a} vs {w_b}x{h_b}")
        return 1

    result = compare_images(img_a, img_b, gain=args.gain)

    diff_path = (
        args.diff
        if args.diff is not None
        else args.first.parent / f"diff_{args.first.stem}_vs_{args.second.stem}.ppm"
    )
    ppm_io.write_ppm(diff_path, result["diff"])
    write_png_preview(diff_path)

    _print_report(args.first, args.second, (w_a, h_a), result, args.gain, diff_path)

    if args.min_psnr is not None:
        if result["psnr"] >= args.min_psnr:
            psnr_text = "inf" if np.isinf(result["psnr"]) else f"{result['psnr']:.2f}"
            print(f"  PASS: PSNR {psnr_text} dB >= {args.min_psnr:g} dB")
        else:
            print(f"  FAIL: PSNR {result['psnr']:.2f} dB < {args.min_psnr:g} dB")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
