"""
Procedural generator for the Germany banner plaque OBJ.

This script creates a rounded plaque with embossed "Deutschland" text and
exports it as an OBJ with vertex normals. It is intentionally self-contained
and uses only common Python packages already available in the workspace.

Usage:
    python gen_deutschland_banner.py
    python gen_deutschland_banner.py --output deutschland_banner.obj

Dependencies:
    numpy
    pillow
    opencv-python
    shapely
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import triangulate, unary_union


def add_vertex(vertices, index_map, x, y, z):
    """Add a vertex once and reuse the index if it already exists."""
    key = (round(float(x), 6), round(float(y), 6), round(float(z), 6))
    idx = index_map.get(key)
    if idx is None:
        idx = len(vertices)
        index_map[key] = idx
        vertices.append(key)
    return idx


def add_face(faces, a, b, c):
    """Append a triangle if it is not degenerate."""
    if a != b and b != c and a != c:
        faces.append((a, b, c))


def extrude_polygon(poly, z0, z1, vertices, index_map, faces):
    """Extrude a Shapely polygon into a watertight solid."""
    poly = orient(poly, sign=1.0)
    rings = [poly.exterior] + list(poly.interiors)

    # Top and bottom caps.
    for tri in triangulate(poly):
        if not poly.contains(tri.representative_point()):
            continue

        coords = list(tri.exterior.coords)[:-1]
        pts = coords[:]

        area2 = 0.0
        for i in range(3):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % 3]
            area2 += x0 * y1 - x1 * y0
        if area2 < 0.0:
            pts = [pts[0], pts[2], pts[1]]

        top = [add_vertex(vertices, index_map, x, y, z1) for x, y in pts]
        bot = [add_vertex(vertices, index_map, x, y, z0) for x, y in pts]
        add_face(faces, top[0], top[1], top[2])
        add_face(faces, bot[0], bot[2], bot[1])

    # Side walls for exterior and holes.
    for ring in rings:
        pts = list(ring.coords)[:-1]
        if len(pts) < 2:
            continue

        top_ids = [add_vertex(vertices, index_map, x, y, z1) for x, y in pts]
        bot_ids = [add_vertex(vertices, index_map, x, y, z0) for x, y in pts]
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            b0, b1 = bot_ids[i], bot_ids[j]
            t0, t1 = top_ids[i], top_ids[j]
            add_face(faces, b0, b1, t1)
            add_face(faces, b0, t1, t0)


def compute_normals(vertices, faces):
    """Compute area-weighted vertex normals."""
    acc = np.zeros((len(vertices), 3), dtype=np.float64)
    verts_np = np.array(vertices, dtype=np.float64)

    for a, b, c in faces:
        va = verts_np[a]
        vb = verts_np[b]
        vc = verts_np[c]
        n = np.cross(vb - va, vc - va)
        acc[a] += n
        acc[b] += n
        acc[c] += n

    normals = []
    for n in acc:
        length = np.linalg.norm(n)
        if length < 1e-12:
            normals.append((0.0, 0.0, 1.0))
        else:
            normals.append(tuple((n / length).tolist()))
    return normals


def write_obj(path, vertices, normals, faces):
    """Write the mesh as OBJ with indexed vertex normals."""
    with path.open("w", encoding="utf-8") as f:
        f.write("# Curved plaque with raised Germany text\n")
        f.write("# Text: Deutschland\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n\n")

        for x, y, z in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

        f.write("\n")
        for nx, ny, nz in normals:
            f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")

        f.write("\n")
        f.write("g plaque_and_text\n")
        for a, b, c in faces:
            ia, ib, ic = a + 1, b + 1, c + 1
            f.write(f"f {ia}//{ia} {ib}//{ib} {ic}//{ic}\n")


def build_banner(text="Deutschland"):
    """Build the banner geometry in object space."""
    font_path = r"C:\Windows\Fonts\arialbd.ttf"
    plaque_w = 6.4
    plaque_h = 2.05
    plaque_r = 0.26
    plaque_thickness = 0.22
    text_depth = 0.11
    margin_x = 0.42
    margin_y = 0.22

    # Rasterize the word so we can extract closed contours and extrude them.
    img_w, img_h = 2400, 760
    img = Image.new("L", (img_w, img_h), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 340)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=2)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = (img_w - text_w) / 2 - bbox[0]
    text_y = (img_h - text_h) / 2 - bbox[1] - 12
    draw.text((text_x, text_y), text, fill=0, font=font, stroke_width=2, stroke_fill=0)

    mask = (np.array(img) < 192).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        raise RuntimeError("Could not detect text contours")
    hierarchy = hierarchy[0]

    letter_polys = []
    for idx, cnt in enumerate(contours):
        if hierarchy[idx][3] != -1:
            continue
        if cv2.contourArea(cnt) < 500.0:
            continue

        epsilon = max(1.0, 0.004 * cv2.arcLength(cnt, True))
        outer = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
        if len(outer) < 3:
            continue

        holes = []
        child = hierarchy[idx][2]
        while child != -1:
            hcnt = contours[child]
            if cv2.contourArea(hcnt) >= 120.0:
                heps = max(1.0, 0.004 * cv2.arcLength(hcnt, True))
                hole = cv2.approxPolyDP(hcnt, heps, True).reshape(-1, 2)
                if len(hole) >= 3:
                    holes.append(hole)
            child = hierarchy[child][0]

        poly = Polygon(outer, holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue

        if poly.geom_type == "Polygon":
            letter_polys.append(poly)
        else:
            letter_polys.extend(list(poly.geoms))

    if not letter_polys:
        raise RuntimeError("No letter polygons were generated")

    union = unary_union(letter_polys)
    minx, miny, maxx, maxy = union.bounds
    text_box_w = maxx - minx
    text_box_h = maxy - miny
    scale = min((plaque_w - 2 * margin_x) / text_box_w, (plaque_h - 2 * margin_y) / text_box_h)
    scale *= 0.98
    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0

    vertices = []
    index_map = {}
    faces = []

    # Base plaque.
    plaque_poly = box(
        -plaque_w / 2 + plaque_r,
        -plaque_h / 2 + plaque_r,
        plaque_w / 2 - plaque_r,
        plaque_h / 2 - plaque_r,
    ).buffer(plaque_r, resolution=24, join_style=1)
    extrude_polygon(plaque_poly, -plaque_thickness / 2, plaque_thickness / 2, vertices, index_map, faces)

    # Raised word.
    for poly in letter_polys:
        p = orient(poly, sign=1.0)
        ext = [((x - center_x) * scale, (center_y - y) * scale + 0.02) for x, y in p.exterior.coords[:-1]]
        holes = [
            [((x - center_x) * scale, (center_y - y) * scale + 0.02) for x, y in ring.coords[:-1]]
            for ring in p.interiors
        ]
        obj_poly = Polygon(ext, holes)
        if not obj_poly.is_valid:
            obj_poly = obj_poly.buffer(0)
        if obj_poly.is_empty:
            continue
        if obj_poly.geom_type == "Polygon":
            extrude_polygon(obj_poly, plaque_thickness / 2, plaque_thickness / 2 + text_depth, vertices, index_map, faces)
        else:
            for part in obj_poly.geoms:
                extrude_polygon(part, plaque_thickness / 2, plaque_thickness / 2 + text_depth, vertices, index_map, faces)

    normals = compute_normals(vertices, faces)
    return vertices, normals, faces


def main():
    parser = argparse.ArgumentParser(description="Generate the Deutschland banner OBJ.")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("deutschland_banner.obj")),
        help="Output OBJ path",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    vertices, normals, faces = build_banner("Deutschland")
    write_obj(out_path, vertices, normals, faces)
    print(f"Wrote {out_path}")
    print(f"Vertices={len(vertices)} Faces={len(faces)}")


if __name__ == "__main__":
    main()
