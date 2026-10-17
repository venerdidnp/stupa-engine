"""
uv_orientation_check.py -- render temple with an asymmetric texture.

Creates a clearly asymmetric texture (the letter 'F' with an arrow)
using NumPy, renders the temple mesh with it, and saves
out/uv_orientation.png.

This allows visual inspection that texture orientation and UV axes (u to the right,
v pointing up) are not flipped horizontally or vertically on any face.

Run: python uv_orientation_check.py
"""

from pathlib import Path
import numpy as np
from PIL import Image

import objio
import ppm_io
import raster3d

OUT_DIR = Path(__file__).resolve().parent / "out"
DATA_DIR = Path(__file__).resolve().parent / "data"


def generate_asymmetric_texture(size=256):
    """Generate an asymmetric 256x256 RGB texture containing the letter 'F'."""
    # Background: pale yellow-orange parchment
    img = np.full((size, size, 3), [240, 230, 205], dtype=np.uint8)

    # Grid / border so tile edges are visible
    img[0:4, :, :] = [180, 50, 50]       # top border: red
    img[-4:, :, :] = [50, 180, 50]       # bottom border: green
    img[:, 0:4, :] = [50, 50, 180]       # left border: blue
    img[:, -4:, :] = [180, 180, 50]      # right border: yellow

    # Draw large letter 'F' in dark navy [20, 25, 60]
    # Vertical spine: x in [45, 80], y in [40, 215]
    img[40:215, 45:80] = [20, 25, 60]
    # Top bar: x in [80, 195], y in [40, 75]
    img[40:75, 80:195] = [20, 25, 60]
    # Middle bar: x in [80, 155], y in [110, 145]
    img[110:145, 80:155] = [20, 25, 60]

    # Red dot indicator at top-right quadrant (x=210..230, y=50..70)
    img[50:70, 210:230] = [220, 30, 30]

    return img


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    obj_path = DATA_DIR / "temple.obj"
    if not obj_path.is_file():
        import make_model
        make_model.build()

    mesh = objio.load_obj(obj_path)
    tex_data = generate_asymmetric_texture(256)
    
    # Save the asymmetric texture separately
    tex_path = OUT_DIR / "asymmetric_marker.ppm"
    ppm_io.write_ppm(tex_path, tex_data)
    
    texture = raster3d.Texture(tex_data)
    settings = raster3d.RenderSettings(width=800, height=600)
    camera = raster3d.Camera()
    
    result = raster3d.render_scene(mesh, texture, settings, camera)
    
    ppm_out = OUT_DIR / "uv_orientation.ppm"
    ppm_io.write_ppm(ppm_out, result.color_srgb)
    
    png_out = OUT_DIR / "uv_orientation.png"
    Image.fromarray(result.color_srgb).save(png_out)
    
    print(f"Wrote asymmetric texture: {tex_path}")
    print(f"Rendered UV orientation check: {ppm_out}")
    print(f"Saved PNG preview: {png_out}")


if __name__ == "__main__":
    main()
