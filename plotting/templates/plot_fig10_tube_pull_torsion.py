#!/usr/bin/env python3
"""Draw Fig. 10 for the 3-D tube pull-torsion benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR
STATE_NAME = "tube_pull_torsion_r30_z300_final.npz"
SUMMARY_NAME = "tube_pull_torsion_summary.json"
LENGTH_SCALE_CM = 10.0
STRESS_SCALE_MPA = 10.0


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.65,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def compact_tick(value: float, _position: int | None = None) -> str:
    if not np.isfinite(value) or abs(value) < 1.0e-14:
        return "0"
    if 1.0e-3 <= abs(value) < 1.0e4:
        return f"{value:.3g}"
    text = f"{value:.1e}"
    mantissa, exponent = text.split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent):+d}"


def read_geometry(summary_path: Path) -> dict[str, float]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    load = summary["row"]["load_resultants"]
    return {
        "cx": float(load["center_x"]),
        "cy": float(load["center_y"]),
        "inner_radius": float(load["inner_radius"]),
        "outer_radius": float(load["outer_radius"]),
        "length_z": float(load["length_z"]),
    }


def append_quad_faces(faces: list[int], offset: int, n0: int, n1: int) -> None:
    for i in range(n0 - 1):
        row = offset + i * n1
        nxt = offset + (i + 1) * n1
        for j in range(n1 - 1):
            faces.extend((4, row + j, row + j + 1, nxt + j + 1, nxt + j))


def add_grid(
    points: list[np.ndarray], faces: list[int], x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> None:
    offset = sum(p.shape[0] for p in points)
    points.append(np.column_stack((x.ravel(), y.ravel(), z.ravel())))
    append_quad_faces(faces, offset, x.shape[0], x.shape[1])


def make_cutaway_tube_mesh(
    *,
    cx: float,
    cy: float,
    inner_radius: float,
    outer_radius: float,
    length_z: float,
    theta_points: int,
    z_points: int,
    radial_points: int,
    cutaway_degrees: float,
    cutaway_center_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    cut = math.radians(cutaway_degrees)
    center = math.radians(cutaway_center_degrees)
    theta0 = center + 0.5 * cut
    theta1 = center + 2.0 * math.pi - 0.5 * cut
    theta = np.linspace(theta0, theta1, theta_points)
    z = np.linspace(0.0, length_z, z_points)
    radius = np.linspace(inner_radius, outer_radius, radial_points)
    points: list[np.ndarray] = []
    faces: list[int] = []

    tt, zz = np.meshgrid(theta, z, indexing="ij")
    for r in (outer_radius, inner_radius):
        add_grid(points, faces, cx + r * np.cos(tt), cy + r * np.sin(tt), zz)

    tt, rr = np.meshgrid(theta, radius, indexing="ij")
    for zcap in (0.0, length_z):
        add_grid(points, faces, cx + rr * np.cos(tt), cy + rr * np.sin(tt), np.full_like(rr, zcap))

    rr, zz = np.meshgrid(radius, z, indexing="ij")
    for th in (theta0, theta1):
        add_grid(points, faces, cx + rr * math.cos(th), cy + rr * math.sin(th), zz)

    return np.vstack(points), np.asarray(faces, dtype=np.int64)


def von_mises_stress(sigma: np.ndarray) -> np.ndarray:
    tr = (sigma[0, 0] + sigma[1, 1] + sigma[2, 2]) / 3.0
    dev = sigma.copy()
    dev[0, 0] -= tr
    dev[1, 1] -= tr
    dev[2, 2] -= tr
    return np.sqrt(np.maximum(1.5 * np.sum(dev * dev, axis=(0, 1)), 0.0))


def axial_hoop_shear_stress(
    sigma: np.ndarray, active_points: np.ndarray, *, cx: float, cy: float
) -> np.ndarray:
    dx = active_points[:, 0] - cx
    dy = active_points[:, 1] - cy
    radius = np.maximum(np.sqrt(dx * dx + dy * dy), 1.0e-30)
    etheta_x = -dy / radius
    etheta_y = dx / radius
    return sigma[2, 0] * etheta_x + sigma[2, 1] * etheta_y


def interpolate_active_fields(
    active_points: np.ndarray,
    query_points: np.ndarray,
    values: np.ndarray,
    *,
    k: int,
    dx: float,
    chunk_size: int = 120_000,
) -> np.ndarray:
    tree = cKDTree(active_points)
    out = np.empty((query_points.shape[0], values.shape[1]), dtype=np.float64)
    softening = max(0.25 * dx, 1.0e-12)
    for start in range(0, query_points.shape[0], chunk_size):
        stop = min(start + chunk_size, query_points.shape[0])
        dist, idx = tree.query(query_points[start:stop], k=k, workers=-1)
        if k == 1:
            out[start:stop] = values[idx]
            continue
        exact = dist[:, 0] < 1.0e-14
        weights = 1.0 / np.maximum(dist, softening) ** 2
        weighted = np.einsum("mk,mkc->mc", weights, values[idx])
        out[start:stop] = weighted / np.sum(weights, axis=1)[:, None]
        if np.any(exact):
            out[start:stop][exact] = values[idx[exact, 0]]
    return out


def build_surface_state(
    args: argparse.Namespace, geom: dict[str, float], state: np.lib.npyio.NpzFile
) -> tuple[pv.PolyData, dict[str, tuple[float, float]]]:
    active_points = np.column_stack((state["x_active"], state["y_active"], state["z_active"]))
    u = np.asarray(state["u"], dtype=np.float64).T
    sigma = np.asarray(state["sigma"], dtype=np.float64)
    u_mag_cm = LENGTH_SCALE_CM * np.sqrt(np.sum(u * u, axis=1))
    sigma_vm_mpa = STRESS_SCALE_MPA * von_mises_stress(sigma)
    sigma_ztheta_mpa = STRESS_SCALE_MPA * axial_hoop_shear_stress(
        sigma,
        active_points,
        cx=geom["cx"],
        cy=geom["cy"],
    )
    field_values = np.column_stack((u, u_mag_cm, sigma_vm_mpa, sigma_ztheta_mpa))

    surface_points, faces = make_cutaway_tube_mesh(
        cx=geom["cx"],
        cy=geom["cy"],
        inner_radius=geom["inner_radius"],
        outer_radius=geom["outer_radius"],
        length_z=geom["length_z"],
        theta_points=int(args.theta_points),
        z_points=int(args.z_points),
        radial_points=int(args.radial_points),
        cutaway_degrees=float(args.cutaway_degrees),
        cutaway_center_degrees=float(args.cutaway_center_degrees),
    )
    interp = interpolate_active_fields(
        active_points,
        surface_points,
        field_values,
        k=int(args.interp_neighbors),
        dx=float(state["dx"]),
    )
    deformed = surface_points + float(args.warp_scale) * interp[:, :3]
    horizontal = np.column_stack(
        (
            deformed[:, 2],
            deformed[:, 0] - geom["cx"],
            deformed[:, 1] - geom["cy"],
        )
    )
    mesh = pv.PolyData(horizontal, faces)
    mesh.point_data["u_mag_cm"] = interp[:, 3]
    mesh.point_data["sigma_vm_mpa"] = interp[:, 4]
    mesh.point_data["sigma_ztheta_mpa"] = interp[:, 5]
    mesh = mesh.compute_normals(
        point_normals=True,
        cell_normals=False,
        auto_orient_normals=True,
        split_vertices=False,
    )
    clim = {
        "u_mag_cm": (0.0, float(np.nanmax(mesh.point_data["u_mag_cm"]))),
        "sigma_vm_mpa": (
            float(np.nanmin(mesh.point_data["sigma_vm_mpa"])),
            float(np.nanmax(mesh.point_data["sigma_vm_mpa"])),
        ),
        "sigma_ztheta_mpa": (
            float(np.nanmin(mesh.point_data["sigma_ztheta_mpa"])),
            float(np.nanmax(mesh.point_data["sigma_ztheta_mpa"])),
        ),
    }
    return mesh, clim


def render_mesh_image(
    mesh: pv.PolyData,
    *,
    scalars: str,
    clim: tuple[float, float],
    cmap: str,
    width: int,
    height: int,
    camera_zoom: float,
    view_angle_degrees: float,
    camera_radius: float,
    camera_axis_offset: float,
    perspective_angle: float,
    crop_pad: int,
) -> np.ndarray:
    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(int(width), int(height)))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    plotter.camera.view_angle = float(perspective_angle)
    try:
        plotter.enable_eye_dome_lighting()
    except Exception:
        pass
    plotter.add_light(
        pv.Light(position=(2.0, -3.0, 5.0), focal_point=(1.6, 0.0, 0.0), intensity=0.86)
    )
    plotter.add_light(
        pv.Light(position=(-2.0, 2.0, 4.0), focal_point=(1.6, 0.0, 0.0), intensity=0.36)
    )
    plotter.add_mesh(
        mesh,
        scalars=scalars,
        cmap=cmap,
        smooth_shading=True,
        show_edges=False,
        ambient=0.22,
        diffuse=0.72,
        specular=0.25,
        specular_power=18.0,
        roughness=0.52,
        clim=clim,
        show_scalar_bar=False,
    )
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    plotter.add_mesh(edges, color=(0.16, 0.16, 0.16), line_width=2.0)

    bounds = mesh.bounds
    center = (
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    )
    view_angle = math.radians(float(view_angle_degrees))
    plotter.camera_position = [
        (
            center[0] + float(camera_axis_offset),
            float(camera_radius) * math.cos(view_angle),
            float(camera_radius) * math.sin(view_angle),
        ),
        center,
        (0.0, math.sin(view_angle), -math.cos(view_angle)),
    ]
    plotter.camera.zoom(float(camera_zoom))
    plotter.show(auto_close=False)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return crop_white(image, pad=int(crop_pad))


def crop_white(image: np.ndarray, *, pad: int = 135) -> np.ndarray:
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


def draw_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    cax: plt.Axes,
    image: np.ndarray,
    *,
    letter: str,
    cmap: str,
    clim: tuple[float, float],
    cbar_label: str,
) -> None:
    ax.imshow(image)
    ax.set_axis_off()
    ax.text(
        -0.006,
        0.965,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.62, "pad": 0.8},
        clip_on=False,
    )
    mappable = ScalarMappable(norm=Normalize(*clim), cmap=plt.get_cmap(cmap))
    mappable.set_array([])
    cbar = fig.colorbar(mappable, cax=cax, format=FuncFormatter(compact_tick))
    cbar.set_ticks(np.linspace(clim[0], clim[1], 4))
    cbar.set_label(cbar_label, fontsize=7.8, labelpad=0.6)
    cbar.ax.tick_params(direction="in", length=2.5, width=0.55, labelsize=7.2, pad=0.9)
    cbar.outline.set_linewidth(0.65)


def make_figure(data_dir: Path, output_dir: Path, basename: str, args: argparse.Namespace) -> None:
    configure_style()
    geom = read_geometry(data_dir / SUMMARY_NAME)
    state = np.load(data_dir / STATE_NAME)
    mesh, clim = build_surface_state(args, geom, state)
    cmap = str(args.cmap)
    u_image = render_mesh_image(
        mesh,
        scalars="u_mag_cm",
        clim=clim["u_mag_cm"],
        cmap=cmap,
        width=int(args.render_width),
        height=int(args.render_height),
        camera_zoom=float(args.camera_zoom),
        view_angle_degrees=float(args.view_angle_degrees),
        camera_radius=float(args.camera_radius),
        camera_axis_offset=float(args.camera_axis_offset),
        perspective_angle=float(args.perspective_angle),
        crop_pad=int(args.crop_pad),
    )
    sigma_image = render_mesh_image(
        mesh,
        scalars="sigma_vm_mpa",
        clim=clim["sigma_vm_mpa"],
        cmap=cmap,
        width=int(args.render_width),
        height=int(args.render_height),
        camera_zoom=float(args.camera_zoom),
        view_angle_degrees=float(args.view_angle_degrees),
        camera_radius=float(args.camera_radius),
        camera_axis_offset=float(args.camera_axis_offset),
        perspective_angle=float(args.perspective_angle),
        crop_pad=int(args.crop_pad),
    )
    shear_image = render_mesh_image(
        mesh,
        scalars="sigma_ztheta_mpa",
        clim=clim["sigma_ztheta_mpa"],
        cmap=cmap,
        width=int(args.render_width),
        height=int(args.render_height),
        camera_zoom=float(args.camera_zoom),
        view_angle_degrees=float(args.view_angle_degrees),
        camera_radius=float(args.camera_radius),
        camera_axis_offset=float(args.camera_axis_offset),
        perspective_angle=float(args.perspective_angle),
        crop_pad=int(args.crop_pad),
    )

    fig = plt.figure(figsize=(7.25, 4.25), constrained_layout=False)
    ax_u = fig.add_axes([0.005, 0.648, 0.850, 0.225])
    ax_s = fig.add_axes([0.005, 0.390, 0.850, 0.225])
    ax_t = fig.add_axes([0.005, 0.128, 0.850, 0.225])
    cax_u = fig.add_axes([0.874, 0.670, 0.018, 0.181])
    cax_s = fig.add_axes([0.874, 0.412, 0.018, 0.181])
    cax_t = fig.add_axes([0.874, 0.150, 0.018, 0.181])

    draw_panel(
        fig,
        ax_u,
        cax_u,
        u_image,
        letter="(a)",
        cmap=cmap,
        clim=clim["u_mag_cm"],
        cbar_label=r"$|\mathbf{u}|$ [cm]",
    )
    draw_panel(
        fig,
        ax_s,
        cax_s,
        sigma_image,
        letter="(b)",
        cmap=cmap,
        clim=clim["sigma_vm_mpa"],
        cbar_label=r"$\sigma_{\mathrm{vm}}$ [MPa]",
    )
    draw_panel(
        fig,
        ax_t,
        cax_t,
        shear_image,
        letter="(c)",
        cmap=cmap,
        clim=clim["sigma_ztheta_mpa"],
        cbar_label=r"$\sigma_{z\theta}$ [MPa]",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{basename}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--basename", default="fig10_tube_pull_torsion")
    parser.add_argument("--cmap", default="turbo")
    parser.add_argument("--warp-scale", type=float, default=1.0)
    parser.add_argument("--theta-points", type=int, default=240)
    parser.add_argument("--z-points", type=int, default=320)
    parser.add_argument("--radial-points", type=int, default=36)
    parser.add_argument("--cutaway-degrees", type=float, default=70.0)
    parser.add_argument("--cutaway-center-degrees", type=float, default=-45.0)
    parser.add_argument("--interp-neighbors", type=int, default=16)
    parser.add_argument("--render-width", type=int, default=1900)
    parser.add_argument("--render-height", type=int, default=760)
    parser.add_argument("--view-angle-degrees", type=float, default=-20.0)
    parser.add_argument("--camera-radius", type=float, default=4.4)
    parser.add_argument("--camera-axis-offset", type=float, default=-0.34)
    parser.add_argument("--camera-zoom", type=float, default=1.08)
    parser.add_argument("--perspective-angle", type=float, default=30.0)
    parser.add_argument("--crop-pad", type=int, default=16)
    args = parser.parse_args()
    make_figure(Path(args.data_dir), Path(args.output_dir), str(args.basename), args)


if __name__ == "__main__":
    main()
