"""
lab1_trace.py -- traces one vertex of the temple model through the coordinate pipeline.

Arguments:
    --vertex INDEX   (default 0)
    --width WIDTH    (default 800)
    --height HEIGHT  (default 600)
    --out PATH       (default KUNCI_LAB1.txt)

Header required:
"KUNCI JAWABAN - hitung dulu dengan tangan, baru bandingkan"
"""

import argparse
from pathlib import Path
import numpy as np

import objio
from raster3d import Camera
from transforms import look_at, perspective, project, viewport


def run_trace(vertex_idx=0, width=800, height=600, m_matrix=None):
    obj_path = Path(__file__).resolve().parent / "data" / "temple.obj"
    mesh = objio.load_obj(obj_path)

    if vertex_idx < 0 or vertex_idx >= len(mesh.positions):
        raise ValueError(f"Vertex index {vertex_idx} out of range [0, {len(mesh.positions)-1}]")

    p_obj = np.array([*mesh.positions[vertex_idx], 1.0], dtype=np.float64)

    # Default model matrix: identity
    if m_matrix is None:
        M = np.eye(4, dtype=np.float64)
    else:
        M = np.asarray(m_matrix, dtype=np.float64)

    # Camera & View matrix
    cam = Camera()
    V = look_at(cam.eye, cam.target, cam.up)

    # Projection matrix
    aspect = width / height
    P = perspective(cam.fovy_deg, aspect, cam.near, cam.far)

    # Step-by-step coordinate transformations
    p_world = M @ p_obj
    p_view = V @ p_world
    p_clip = P @ p_view
    p_ndc = p_clip[:3] / p_clip[3]

    scr = viewport(p_ndc[None, :], width, height)[0]
    p_screen = np.array([scr[0], scr[1], scr[2]], dtype=np.float64)

    # Baca kolom:
    # Untuk M:
    m_origin = M @ np.array([0.0, 0.0, 0.0, 1.0])
    m_x = M @ np.array([1.0, 0.0, 0.0, 0.0])
    m_y = M @ np.array([0.0, 1.0, 0.0, 0.0])
    m_z = M @ np.array([0.0, 0.0, 1.0, 0.0])

    # Untuk V terhadap origin dan basis world:
    v_origin = V @ np.array([0.0, 0.0, 0.0, 1.0])
    v_x = V @ np.array([1.0, 0.0, 0.0, 0.0])
    v_y = V @ np.array([0.0, 1.0, 0.0, 0.0])
    v_z = V @ np.array([0.0, 0.0, 1.0, 0.0])

    # Untuk V @ M (model origin dan basis setelah M lalu V):
    VM = V @ M
    vm_origin = VM @ np.array([0.0, 0.0, 0.0, 1.0])
    vm_x = VM @ np.array([1.0, 0.0, 0.0, 0.0])
    vm_y = VM @ np.array([0.0, 1.0, 0.0, 0.0])
    vm_z = VM @ np.array([0.0, 0.0, 1.0, 0.0])

    lines = []
    lines.append("KUNCI JAWABAN - hitung dulu dengan tangan, baru bandingkan")
    lines.append("")
    lines.append(f"Vertex Index: {vertex_idx} (dari {len(mesh.positions)} vertex)")
    lines.append(f"Viewport: {width} x {height}")
    lines.append(f"Camera Eye: {list(cam.eye)}")
    lines.append(f"Camera Target: {list(cam.target)}")
    lines.append(f"Camera Up: {list(cam.up)}")
    lines.append(f"Camera FOV (deg): {cam.fovy_deg}, Near: {cam.near}, Far: {cam.far}")
    lines.append("")

    lines.append("Matriks M (Model Matrix):")
    for row in M:
        lines.append("  [ " + "  ".join(f"{v:10.6f}" for v in row) + " ]")
    lines.append("")

    lines.append("Baca Kolom M:")
    lines.append(f"  Origin model (0,0,0,1) -> [{m_origin[0]:.6f}, {m_origin[1]:.6f}, {m_origin[2]:.6f}, {m_origin[3]:.6f}]")
    lines.append(f"  Arah +x model (1,0,0,0) -> [{m_x[0]:.6f}, {m_x[1]:.6f}, {m_x[2]:.6f}, {m_x[3]:.6f}]")
    lines.append(f"  Arah +y model (0,1,0,0) -> [{m_y[0]:.6f}, {m_y[1]:.6f}, {m_y[2]:.6f}, {m_y[3]:.6f}]")
    lines.append(f"  Arah +z model (0,0,1,0) -> [{m_z[0]:.6f}, {m_z[1]:.6f}, {m_z[2]:.6f}, {m_z[3]:.6f}]")
    lines.append("")

    lines.append("Matriks V (View Matrix):")
    for row in V:
        lines.append("  [ " + "  ".join(f"{v:10.6f}" for v in row) + " ]")
    lines.append("")

    lines.append("Baca Kolom V (terhadap basis world):")
    lines.append(f"  Origin world (0,0,0,1) -> [{v_origin[0]:.6f}, {v_origin[1]:.6f}, {v_origin[2]:.6f}, {v_origin[3]:.6f}]")
    lines.append(f"  Arah +x world (1,0,0,0) -> [{v_x[0]:.6f}, {v_x[1]:.6f}, {v_x[2]:.6f}, {v_x[3]:.6f}]")
    lines.append(f"  Arah +y world (0,1,0,0) -> [{v_y[0]:.6f}, {v_y[1]:.6f}, {v_y[2]:.6f}, {v_y[3]:.6f}]")
    lines.append(f"  Arah +z world (0,0,1,0) -> [{v_z[0]:.6f}, {v_z[1]:.6f}, {v_z[2]:.6f}, {v_z[3]:.6f}]")
    lines.append("")

    lines.append("Baca Kolom V @ M (ke mana origin dan sumbu model pergi setelah M lalu V):")
    lines.append(f"  Origin model setelah M lalu V -> [{vm_origin[0]:.6f}, {vm_origin[1]:.6f}, {vm_origin[2]:.6f}, {vm_origin[3]:.6f}]")
    lines.append(f"  Arah +x model setelah M lalu V -> [{vm_x[0]:.6f}, {vm_x[1]:.6f}, {vm_x[2]:.6f}, {vm_x[3]:.6f}]")
    lines.append(f"  Arah +y model setelah M lalu V -> [{vm_y[0]:.6f}, {vm_y[1]:.6f}, {vm_y[2]:.6f}, {vm_y[3]:.6f}]")
    lines.append(f"  Arah +z model setelah M lalu V -> [{vm_z[0]:.6f}, {vm_z[1]:.6f}, {vm_z[2]:.6f}, {vm_z[3]:.6f}]")
    lines.append("")

    lines.append("Matriks P (Projection Matrix):")
    for row in P:
        lines.append("  [ " + "  ".join(f"{v:10.6f}" for v in row) + " ]")
    lines.append("")

    lines.append("Telusuran Koordinat Vertex:")
    lines.append(f"  1. Object Space : [{p_obj[0]:.6f}, {p_obj[1]:.6f}, {p_obj[2]:.6f}, {p_obj[3]:.6f}]")
    lines.append(f"     (3 desimal)  : [{p_obj[0]:.3f}, {p_obj[1]:.3f}, {p_obj[2]:.3f}, {p_obj[3]:.3f}]")
    lines.append(f"  2. World Space  : [{p_world[0]:.6f}, {p_world[1]:.6f}, {p_world[2]:.6f}, {p_world[3]:.6f}]")
    lines.append(f"     (3 desimal)  : [{p_world[0]:.3f}, {p_world[1]:.3f}, {p_world[2]:.3f}, {p_world[3]:.3f}]")
    lines.append(f"  3. View Space   : [{p_view[0]:.6f}, {p_view[1]:.6f}, {p_view[2]:.6f}, {p_view[3]:.6f}]")
    lines.append(f"     (3 desimal)  : [{p_view[0]:.3f}, {p_view[1]:.3f}, {p_view[2]:.3f}, {p_view[3]:.3f}]")
    lines.append(f"  4. Clip Space   : [{p_clip[0]:.6f}, {p_clip[1]:.6f}, {p_clip[2]:.6f}, {p_clip[3]:.6f}]")
    lines.append(f"     (3 desimal)  : [{p_clip[0]:.3f}, {p_clip[1]:.3f}, {p_clip[2]:.3f}, {p_clip[3]:.3f}]")
    lines.append(f"  5. NDC Space    : [{p_ndc[0]:.6f}, {p_ndc[1]:.6f}, {p_ndc[2]:.6f}]")
    lines.append(f"     (3 desimal)  : [{p_ndc[0]:.3f}, {p_ndc[1]:.3f}, {p_ndc[2]:.3f}]")
    lines.append(f"  6. Screen Space : pixel x = {p_screen[0]:.6f}, y = {p_screen[1]:.6f}, depth01 = {p_screen[2]:.6f}")
    lines.append(f"     (3 desimal)  : pixel x = {p_screen[0]:.3f}, y = {p_screen[1]:.3f}, depth01 = {p_screen[2]:.3f}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Trace a vertex through coordinates pipeline.")
    parser.add_argument("--vertex", type=int, default=0, help="Vertex index (default 0)")
    parser.add_argument("--width", type=int, default=800, help="Screen width (default 800)")
    parser.add_argument("--height", type=int, default=600, help="Screen height (default 600)")
    parser.add_argument("--out", type=str, default="KUNCI_LAB1.txt", help="Output file path")
    args = parser.parse_args()

    content = run_trace(vertex_idx=args.vertex, width=args.width, height=args.height)
    out_path = Path(args.out)
    out_path.write_text(content, encoding="utf-8")
    print(content)
    print(f"Output saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
