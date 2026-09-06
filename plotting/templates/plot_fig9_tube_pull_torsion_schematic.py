#!/usr/bin/env python3
"""Draw a schematic for the finite hollow tube pull-torsion benchmark.

This figure is generated from the analytic boundary-condition map, not from an
image model.  The deformed body therefore remains a nearly cylindrical hollow
tube with uniform wall thickness: 50% axial stretch, 45 deg torsion, and the
corresponding Poisson contraction.  The axial coordinate is compressed only in
the rendered schematic so that the large deformation remains readable.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pyvista as pv


SCRIPT_DIR = Path(__file__).resolve().parent

RO = 3.0  # cm
RI = 1.5  # cm
L = 30.0  # cm
AXIAL_STRAIN = 0.50
TWIST_ANGLE = math.radians(45.0)
POISSON = 0.20
RADIAL_SCALE = 1.0 - POISSON * AXIAL_STRAIN
L_DEFORMED = (1.0 + AXIAL_STRAIN) * L
AXIAL_DISPLAY_SCALE = 0.70
L_DEFORMED_DISPLAY = AXIAL_DISPLAY_SCALE * L_DEFORMED

DEFORMED_COLOR = (0.82, 0.83, 0.80)
INNER_COLOR = (0.64, 0.66, 0.63)
CLAMP_COLOR = (0.56, 0.58, 0.57)
EDGE_COLOR = (0.16, 0.16, 0.15)
REFERENCE_BLUE = (0.00, 0.18, 0.92)
REFERENCE_HALO = (0.86, 0.94, 1.00)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.linewidth": 0.65,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def deform_points(radius: np.ndarray, theta: np.ndarray, z: np.ndarray) -> np.ndarray:
    theta_def = theta + TWIST_ANGLE * z / L
    radius_def = RADIAL_SCALE * radius
    return np.column_stack(
        (
            (AXIAL_DISPLAY_SCALE * (1.0 + AXIAL_STRAIN) * z).ravel(),
            (radius_def * np.cos(theta_def)).ravel(),
            (radius_def * np.sin(theta_def)).ravel(),
        )
    )


def reference_points(radius: np.ndarray, theta: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            (AXIAL_DISPLAY_SCALE * z).ravel(),
            (radius * np.cos(theta)).ravel(),
            (radius * np.sin(theta)).ravel(),
        )
    )


def append_quad_faces(faces: list[int], offset: int, n0: int, n1: int) -> None:
    for i in range(n0 - 1):
        row = offset + i * n1
        nxt = offset + (i + 1) * n1
        for j in range(n1 - 1):
            faces.extend((4, row + j, row + j + 1, nxt + j + 1, nxt + j))


def add_grid(
    points: list[np.ndarray], faces: list[int], grid_points: np.ndarray, n0: int, n1: int
) -> None:
    offset = sum(p.shape[0] for p in points)
    points.append(grid_points)
    append_quad_faces(faces, offset, n0, n1)


def make_cutaway_mesh(
    *, deformed: bool, theta_points: int = 180, z_points: int = 260, radial_points: int = 36
) -> pv.PolyData:
    transform = deform_points if deformed else reference_points
    theta0, theta1 = cutaway_angles()

    theta = np.linspace(theta0, theta1, theta_points)
    z = np.linspace(0.0, L, z_points)
    radius = np.linspace(RI, RO, radial_points)
    points: list[np.ndarray] = []
    faces: list[int] = []

    tt, zz = np.meshgrid(theta, z, indexing="ij")
    for r in (RO, RI):
        rr = np.full_like(tt, r)
        add_grid(points, faces, transform(rr, tt, zz), theta_points, z_points)

    tt_cap, rr_cap = np.meshgrid(theta, radius, indexing="ij")
    for zcap in (0.0, L):
        zz_cap = np.full_like(tt_cap, zcap)
        add_grid(points, faces, transform(rr_cap, tt_cap, zz_cap), theta_points, radial_points)

    rr_cut, zz_cut = np.meshgrid(radius, z, indexing="ij")
    for th in (theta0, theta1):
        tt_cut = np.full_like(rr_cut, th)
        add_grid(points, faces, transform(rr_cut, tt_cut, zz_cut), radial_points, z_points)

    mesh = pv.PolyData(np.vstack(points), np.asarray(faces, dtype=np.int64))
    return mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)


def make_surface_mesh(
    *,
    radius_value: float,
    deformed: bool,
    theta_points: int = 180,
    z_points: int = 260,
) -> pv.PolyData:
    transform = deform_points if deformed else reference_points
    theta0, theta1 = cutaway_angles()
    theta = np.linspace(theta0, theta1, theta_points)
    z = np.linspace(0.0, L, z_points)
    tt, zz = np.meshgrid(theta, z, indexing="ij")
    rr = np.full_like(tt, radius_value)
    points = transform(rr, tt, zz)
    faces: list[int] = []
    append_quad_faces(faces, 0, theta_points, z_points)
    mesh = pv.PolyData(points, np.asarray(faces, dtype=np.int64))
    return mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)


def cutaway_angles() -> tuple[float, float]:
    cutaway = math.radians(78.0)
    center = math.radians(-42.0)
    theta0 = center + 0.5 * cutaway
    theta1 = center + 2.0 * math.pi - 0.5 * cutaway
    return theta0, theta1


def dashed_line_polydata(
    curves: list[np.ndarray], *, dash: float = 0.62, gap: float = 0.38
) -> pv.PolyData:
    """Convert smooth curves to many 2-point line cells with dash/gap spacing."""
    points: list[np.ndarray] = []
    lines: list[int] = []
    period = dash + gap
    next_point = 0
    for curve in curves:
        curve = np.asarray(curve, dtype=np.float64)
        segment_lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        distance0 = np.concatenate(([0.0], np.cumsum(segment_lengths[:-1])))
        for i, length in enumerate(segment_lengths):
            if length <= 1.0e-12:
                continue
            midpoint_distance = distance0[i] + 0.5 * length
            if midpoint_distance % period > dash:
                continue
            points.extend((curve[i], curve[i + 1]))
            lines.extend((2, next_point, next_point + 1))
            next_point += 2
    if not points:
        return pv.PolyData()
    mesh = pv.PolyData(np.asarray(points))
    mesh.lines = np.asarray(lines, dtype=np.int64)
    return mesh


def make_reference_dashed_lines() -> pv.PolyData:
    theta0, theta1 = cutaway_angles()
    curves: list[np.ndarray] = []
    theta = np.linspace(theta0, theta1, 520)
    z_line = np.linspace(0.0, L, 520)

    # End and mid-span circular outlines of the undeformed tube.
    for zcap in (0.0, 0.50 * L, L):
        for radius in (RI, RO):
            curves.append(
                reference_points(np.full_like(theta, radius), theta, np.full_like(theta, zcap))
            )

    # Axial generator lines on both the outer and inner cylindrical surfaces.
    for th in np.linspace(theta0, theta1, 7):
        for radius in (RI, RO):
            curves.append(
                reference_points(np.full_like(z_line, radius), np.full_like(z_line, th), z_line)
            )

    # Cut-face radial lines at both ends clarify that the reference is hollow.
    radius_line = np.linspace(RI, RO, 120)
    for zcap in (0.0, L):
        for th in (theta0, theta1):
            curves.append(
                reference_points(
                    radius_line, np.full_like(radius_line, th), np.full_like(radius_line, zcap)
                )
            )

    return dashed_line_polydata(curves, dash=0.82, gap=0.34)


def crop_white(image: np.ndarray, *, pad: int = 70) -> np.ndarray:
    rgb = image[..., :3]
    mask = np.any(rgb < 248, axis=2)
    if not np.any(mask):
        return rgb
    rows, cols = np.where(mask)
    y0 = max(int(rows.min()) - pad, 0)
    y1 = min(int(rows.max()) + pad + 1, rgb.shape[0])
    x0 = max(int(cols.min()) - pad, 0)
    x1 = min(int(cols.max()) + pad + 1, rgb.shape[1])
    return rgb[y0:y1, x0:x1]


def render_tube_image() -> np.ndarray:
    pv.OFF_SCREEN = True
    deformed = make_cutaway_mesh(deformed=True)
    outer_surface = make_surface_mesh(radius_value=RO, deformed=True)
    inner_surface = make_surface_mesh(radius_value=RI, deformed=True)
    reference_lines = make_reference_dashed_lines()
    reference_halo_tubes = reference_lines.tube(radius=0.080, n_sides=10)
    reference_core_tubes = reference_lines.tube(radius=0.040, n_sides=10)

    plotter = pv.Plotter(off_screen=True, window_size=(2400, 980))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    try:
        plotter.enable_eye_dome_lighting()
    except Exception:
        pass

    light_focus = (0.50 * L_DEFORMED_DISPLAY, 0.0, 0.0)
    plotter.add_light(
        pv.Light(position=(7.0, -12.0, 12.0), focal_point=light_focus, intensity=0.85)
    )
    plotter.add_light(pv.Light(position=(-6.0, 10.0, 8.0), focal_point=light_focus, intensity=0.35))
    clamp_points = np.array(
        [
            [-0.82, -4.55, -4.45],
            [-0.82, 4.55, -4.45],
            [-0.82, 4.55, 4.45],
            [-0.82, -4.55, 4.45],
        ],
        dtype=np.float64,
    )
    clamp = pv.PolyData(clamp_points, np.asarray([4, 0, 1, 2, 3], dtype=np.int64))
    plotter.add_mesh(
        clamp,
        color=CLAMP_COLOR,
        smooth_shading=True,
        show_edges=True,
        edge_color=EDGE_COLOR,
        line_width=1.1,
        ambient=0.35,
        diffuse=0.60,
    )
    plotter.add_mesh(
        deformed,
        color=INNER_COLOR,
        smooth_shading=True,
        show_edges=False,
        ambient=0.30,
        diffuse=0.68,
        specular=0.22,
        specular_power=20.0,
        roughness=0.55,
    )
    plotter.add_mesh(
        outer_surface,
        color=DEFORMED_COLOR,
        smooth_shading=True,
        show_edges=False,
        ambient=0.34,
        diffuse=0.66,
        specular=0.32,
        specular_power=24.0,
        roughness=0.48,
    )
    plotter.add_mesh(
        inner_surface,
        color=INNER_COLOR,
        smooth_shading=True,
        show_edges=False,
        ambient=0.32,
        diffuse=0.64,
        specular=0.20,
        specular_power=16.0,
        roughness=0.58,
    )
    edges = deformed.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    plotter.add_mesh(edges, color=EDGE_COLOR, line_width=2.0)
    plotter.add_mesh(
        reference_halo_tubes,
        color=REFERENCE_HALO,
        opacity=0.88,
        smooth_shading=True,
        ambient=0.50,
        diffuse=0.50,
    )
    plotter.add_mesh(
        reference_core_tubes,
        color=REFERENCE_BLUE,
        opacity=1.00,
        smooth_shading=True,
        ambient=0.42,
        diffuse=0.58,
    )

    center = (0.50 * L_DEFORMED_DISPLAY, 0.0, 0.0)
    view_angle = math.radians(-18.0)
    camera_radius = 38.0
    plotter.camera.view_angle = 28.0
    plotter.camera_position = [
        (
            center[0] - 2.2,
            camera_radius * math.cos(view_angle),
            camera_radius * math.sin(view_angle),
        ),
        center,
        (0.0, math.sin(view_angle), -math.cos(view_angle)),
    ]
    plotter.camera.zoom(0.95)
    plotter.show(auto_close=False)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return crop_white(image, pad=46)


def make_figure() -> None:
    configure_style()
    image = render_tube_image()

    fig = plt.figure(figsize=(9.2, 3.70), constrained_layout=False)
    ax = fig.add_axes([0.0, 0.20, 1.0, 0.80])
    ax.imshow(image)
    ax.set_axis_off()
    legend = fig.add_axes([0.31, 0.020, 0.52, 0.165])
    legend.set_axis_off()
    legend.add_patch(
        Rectangle(
            (0.000, 0.58),
            0.090,
            0.245,
            transform=legend.transAxes,
            facecolor=DEFORMED_COLOR,
            edgecolor="black",
            lw=0.6,
        )
    )
    legend.text(
        0.120,
        0.610,
        "Deformed configuration (stretched and twisted)",
        transform=legend.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.4,
    )
    legend.add_patch(
        Rectangle(
            (0.000, 0.125),
            0.090,
            0.245,
            transform=legend.transAxes,
            facecolor=(REFERENCE_BLUE[0], REFERENCE_BLUE[1], REFERENCE_BLUE[2], 0.16),
            edgecolor=REFERENCE_BLUE,
            lw=1.5,
            linestyle=(0, (5, 3)),
        )
    )
    legend.text(
        0.120,
        0.155,
        "Undeformed reference configuration",
        transform=legend.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.4,
    )

    for ext in ("png", "pdf"):
        fig.savefig(
            SCRIPT_DIR / f"fig9_tube_pull_torsion_schematic.{ext}",
            bbox_inches="tight",
            pad_inches=0.015,
        )
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
