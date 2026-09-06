#!/usr/bin/env python3
"""Draw Fig. 7 for the 3-D spherical-shell radial BVP benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR
LENGTH_SCALE_CM = 10.0
STRESS_SCALE_MPA = 10.0

CENTER = np.array([0.5, 0.5, 0.5], dtype=np.float64)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.65,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def compact_sci(value: float, _position: int | None = None) -> str:
    if not np.isfinite(value) or abs(value) < 1.0e-14:
        return "0"
    if 1.0e-3 <= abs(value) < 1.0e4:
        return f"{value:.3g}"
    text = f"{value:.1e}"
    mantissa, exponent = text.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent):+d}"


def panel_label(ax, text: str, *, is_3d: bool = False) -> None:
    kwargs = {
        "ha": "left",
        "va": "top",
        "fontsize": 10.5,
        "fontweight": "bold",
        "bbox": {"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    }
    if is_3d:
        ax.text2D(0.02, 0.98, text, transform=ax.transAxes, **kwargs)
    else:
        ax.text(0.02, 0.98, text, transform=ax.transAxes, **kwargs)


def horizontal_cbar_ticks(cbar) -> None:
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.ax.tick_params(axis="x", direction="in", length=2.5, width=0.55, labelsize=7.0, pad=1.0)
    cbar.outline.set_linewidth(0.65)


def clean_3d_axes(ax) -> None:
    ax.grid(False)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        except AttributeError:
            pass
        try:
            axis._axinfo["grid"]["linewidth"] = 0.0
            axis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.0)
        except (AttributeError, KeyError):
            pass


def projected_zlabel(ax, label: str, *, dx_px: float = 120.0) -> None:
    xmin, xmax = ax.get_xlim3d()
    ymin, ymax = ax.get_ylim3d()
    zmin, zmax = ax.get_zlim3d()
    zmid = 0.5 * (zmin + zmax)

    projected = []
    for x, y in ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)):
        xp, yp, _ = proj3d.proj_transform(x, y, zmid, ax.get_proj())
        xd, yd = ax.transData.transform((xp, yp))
        projected.append((xd, yd))

    xd, yd = max(projected, key=lambda xy: xy[0])
    xa, ya = ax.transAxes.inverted().transform((xd + dx_px, yd))
    ax.text2D(
        xa,
        ya,
        label,
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        clip_on=False,
    )


def set_equal_3d(ax, points: np.ndarray, *, pad: float = 0.06, zoom: float = 1.14) -> None:
    points_cm = LENGTH_SCALE_CM * points
    lo = np.min(points_cm, axis=1)
    hi = np.max(points_cm, axis=1)
    span = np.maximum(hi - lo, 1.0e-12)
    lo = lo - pad * span
    hi = hi + pad * span
    span = hi - lo
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_box_aspect(tuple(span / np.max(span)), zoom=zoom)
    except TypeError:
        ax.set_box_aspect(tuple(span / np.max(span)))
    except AttributeError:
        pass


def set_3d_cm_ticks(ax) -> None:
    ax.set_xticks([2.5, 5.0, 7.5])
    ax.set_yticks([2.5, 5.0, 7.5])
    ax.set_zticks([2.5, 5.0, 7.5])


def sphere_surface(radius: float, *, ntheta: int = 64, nphi: int = 116) -> np.ndarray:
    theta = np.linspace(0.0, np.pi, ntheta)
    phi = np.linspace(0.55 * np.pi, 1.75 * np.pi, nphi)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    return np.stack(
        [
            CENTER[0] + radius * np.sin(TH) * np.cos(PH),
            CENTER[1] + radius * np.sin(TH) * np.sin(PH),
            CENTER[2] + radius * np.cos(TH),
        ],
        axis=0,
    )


def von_mises_3d(sigma: np.ndarray) -> np.ndarray:
    tr = (sigma[0, 0] + sigma[1, 1] + sigma[2, 2]) / 3.0
    dev = sigma.copy()
    dev[0, 0] -= tr
    dev[1, 1] -= tr
    dev[2, 2] -= tr
    return np.sqrt(np.maximum(1.5 * np.sum(dev * dev, axis=(0, 1)), 0.0))


def field_limits(values: np.ndarray, *, zero_min: bool = True) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(vals, [0.5, 99.5])
    if zero_min:
        lo = 0.0
    if not hi > lo:
        hi = lo + max(abs(lo), 1.0) * 1.0e-6
    return float(lo), float(hi)


def interpolate_active(
    tree: cKDTree, values: np.ndarray, query: np.ndarray, *, k: int = 10
) -> np.ndarray:
    dist, idx = tree.query(query, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    weights = 1.0 / np.maximum(dist, 1.0e-12) ** 2
    weights /= np.sum(weights, axis=1, keepdims=True)
    flat = values.reshape((-1, values.shape[-1]))
    out = np.sum(flat[:, idx] * weights[None, :, :], axis=2)
    return out.reshape(values.shape[:-1] + query.shape[:1])


def build_surface_fields(data_dir: Path, summary: dict) -> dict[str, list[dict[str, np.ndarray]]]:
    rows = sorted(summary["rows"], key=lambda row: int(row["n"]))
    n = int(rows[-1]["n"])
    num = np.load(data_dir / f"spherical_shell_radial_bvp_n{n}_final.npz")
    ref = np.load(data_dir / f"spherical_shell_radial_bvp_reference_n{n}.npz")

    coords = np.column_stack([num["x_active"], num["y_active"], num["z_active"]])
    tree = cKDTree(coords)
    u_num = np.asarray(num["u"], dtype=np.float64)
    u_ex = np.asarray(ref["u"], dtype=np.float64)
    sigma_num = STRESS_SCALE_MPA * np.asarray(num["sigma"], dtype=np.float64)
    if "sigma" in ref.files:
        sigma_ex = STRESS_SCALE_MPA * np.asarray(ref["sigma"], dtype=np.float64)
    else:
        raise RuntimeError(
            "reference file does not contain exact sigma; rerun the Fig. 7 case script"
        )

    vm_num = von_mises_3d(sigma_num)[None, :]
    vm_ex = von_mises_3d(sigma_ex)[None, :]
    radii = [
        float(summary["geometry"]["outer_radius"]),
        float(summary["geometry"]["inner_radius"]),
    ]

    surfaces: list[dict[str, np.ndarray]] = []
    for radius in radii:
        X0 = sphere_surface(radius)
        query = X0.reshape(3, -1).T
        us = interpolate_active(tree, u_num, query).reshape((3,) + X0.shape[1:])
        ues = interpolate_active(tree, u_ex, query).reshape((3,) + X0.shape[1:])
        vms = interpolate_active(tree, vm_num, query).reshape(X0.shape[1:])
        vm_es = interpolate_active(tree, vm_ex, query).reshape(X0.shape[1:])
        Xd = X0 + us
        fields = {
            "X0": X0,
            "Xd": Xd,
            "u_num": LENGTH_SCALE_CM * np.sqrt(np.sum(us * us, axis=0)),
            "u_err": LENGTH_SCALE_CM * np.sqrt(np.sum((us - ues) * (us - ues), axis=0)),
            "sigma_vm": vms,
            "sigma_vm_err": np.abs(vms - vm_es),
        }
        for key in ("u_num", "u_err", "sigma_vm", "sigma_vm_err"):
            fields[key] = gaussian_filter(fields[key], sigma=0.55, mode=("nearest", "wrap"))
        surfaces.append(fields)

    return {
        "n": n,
        "surfaces": surfaces,
    }


def draw_shell_panel(
    fig: plt.Figure,
    ax,
    cax,
    surfaces: list[dict[str, np.ndarray]],
    field_key: str,
    label: str,
    limits: tuple[float, float],
    letter: str,
) -> None:
    norm = Normalize(*limits)
    for index, surf in enumerate(surfaces):
        values = np.clip(surf[field_key], limits[0], limits[1])
        colors = plt.cm.rainbow(norm(values))
        alpha = 0.58 if index == 0 else 0.98
        ax.plot_surface(
            *(LENGTH_SCALE_CM * surf["Xd"]),
            facecolors=colors,
            rstride=1,
            cstride=1,
            linewidth=0.0,
            antialiased=True,
            shade=False,
            alpha=alpha,
            rasterized=True,
        )
        ax.plot_wireframe(
            *(LENGTH_SCALE_CM * surf["X0"]),
            rstride=8,
            cstride=12,
            color="0.25",
            linewidth=0.18,
            alpha=0.18,
        )

    all_points = np.concatenate([surf["Xd"].reshape(3, -1) for surf in surfaces], axis=1)
    set_equal_3d(ax, all_points, pad=0.025, zoom=1.10)
    set_3d_cm_ticks(ax)
    ax.set_xlabel(r"$X_1$ [cm]", labelpad=-9.0)
    ax.set_ylabel(r"$X_2$ [cm]", labelpad=-9.0)
    ax.set_zlabel("")
    ax.tick_params(pad=-3.5)
    clean_3d_axes(ax)
    ax.view_init(elev=18.0, azim=-18.0)
    projected_zlabel(ax, r"$X_3$ [cm]", dx_px=120.0)
    panel_label(ax, letter, is_3d=True)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap="rainbow")
    mappable.set_array([])
    cbar = fig.colorbar(
        mappable, cax=cax, orientation="horizontal", format=FuncFormatter(compact_sci)
    )
    horizontal_cbar_ticks(cbar)
    cbar.set_ticks(np.linspace(limits[0], limits[1], 3))
    cbar.set_label(label, fontsize=7.2, labelpad=1.0)


def radial_stress(sigma: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    radial = np.stack([x - CENTER[0], y - CENTER[1], z - CENTER[2]], axis=0)
    radius = np.maximum(np.sqrt(np.sum(radial * radial, axis=0)), 1.0e-30)
    er = radial / radius
    return np.einsum("in,ijn,jn->n", er, sigma, er)


def masked_finite_values(*arrays: np.ndarray) -> np.ndarray:
    pieces = []
    for array in arrays:
        compressed = np.ma.compressed(np.ma.masked_invalid(array))
        if compressed.size:
            pieces.append(np.asarray(compressed, dtype=np.float64))
    if not pieces:
        return np.array([], dtype=np.float64)
    return np.concatenate(pieces)


def slice_grid(
    summary: dict, *, resolution: int = 500
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometry = summary["geometry"]
    inner_radius = float(geometry["inner_radius"])
    outer_radius = float(geometry["outer_radius"])
    pad = 0.012
    axis = np.linspace(CENTER[0] - outer_radius - pad, CENTER[0] + outer_radius + pad, resolution)
    A, B = np.meshgrid(axis, axis, indexing="xy")
    radius = np.sqrt((A - CENTER[0]) ** 2 + (B - CENTER[1]) ** 2)
    mask = (radius < inner_radius) | (radius > outer_radius)
    return LENGTH_SCALE_CM * A, LENGTH_SCALE_CM * B, mask


def interpolate_slice(
    coords: np.ndarray,
    values: np.ndarray,
    summary: dict,
    *,
    normal_axis: int,
    resolution: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    rows = sorted(summary["rows"], key=lambda row: int(row["n"]))
    dx = float(rows[-1]["dx"])
    plane_axes = tuple(axis for axis in range(3) if axis != normal_axis)
    distance = np.abs(coords[normal_axis] - CENTER[normal_axis])

    band = np.zeros_like(distance, dtype=bool)
    for width in (0.55, 1.05, 1.55, 2.55):
        band = distance <= (np.min(distance) + width * dx)
        if np.count_nonzero(band) >= 2500:
            break

    GX, GY, outside_mask = slice_grid(summary, resolution=resolution)
    grid_a = GX / LENGTH_SCALE_CM
    grid_b = GY / LENGTH_SCALE_CM
    points = np.column_stack([coords[plane_axes[0], band], coords[plane_axes[1], band]])
    sampled = np.asarray(values, dtype=np.float64)[band]

    field = griddata(points, sampled, (grid_a, grid_b), method="linear")
    missing = np.isnan(field) & ~outside_mask
    if np.any(missing):
        nearest = griddata(points, sampled, (grid_a, grid_b), method="nearest")
        field[missing] = nearest[missing]

    return GX, GY, np.ma.array(field, mask=outside_mask | ~np.isfinite(field))


def build_slice_fields(data_dir: Path, summary: dict) -> dict[str, dict[str, object]]:
    rows = sorted(summary["rows"], key=lambda row: int(row["n"]))
    n = int(rows[-1]["n"])
    num = np.load(data_dir / f"spherical_shell_radial_bvp_n{n}_final.npz")
    ref = np.load(data_dir / f"spherical_shell_radial_bvp_reference_n{n}.npz")

    coords = np.vstack([num["x_active"], num["y_active"], num["z_active"]]).astype(np.float64)
    u_num = np.asarray(num["u"], dtype=np.float64)
    u_ex = np.asarray(ref["u"], dtype=np.float64)
    sigma_num = STRESS_SCALE_MPA * np.asarray(num["sigma"], dtype=np.float64)
    if "sigma" not in ref.files:
        raise RuntimeError(
            "reference file does not contain exact sigma; rerun the Fig. 7 case script"
        )
    sigma_ex = STRESS_SCALE_MPA * np.asarray(ref["sigma"], dtype=np.float64)

    scalar_fields = {
        "u": {
            "normal_axis": 0,
            "title": r"$|\mathbf{u}|$  ($X_1$ projection)",
            "unit": r"[cm]",
            "num": LENGTH_SCALE_CM * np.sqrt(np.sum(u_num * u_num, axis=0)),
            "exact": LENGTH_SCALE_CM * np.sqrt(np.sum(u_ex * u_ex, axis=0)),
            "error": LENGTH_SCALE_CM * np.sqrt(np.sum((u_num - u_ex) * (u_num - u_ex), axis=0)),
            "signed": False,
        },
        "sigma_rr": {
            "normal_axis": 1,
            "title": r"$\sigma_{rr}$  ($X_2$ projection)",
            "unit": r"[MPa]",
            "num": radial_stress(sigma_num, coords[0], coords[1], coords[2]),
            "exact": radial_stress(sigma_ex, coords[0], coords[1], coords[2]),
            "error": np.abs(
                radial_stress(sigma_num, coords[0], coords[1], coords[2])
                - radial_stress(sigma_ex, coords[0], coords[1], coords[2])
            ),
            "signed": True,
        },
        "sigma_vm": {
            "normal_axis": 2,
            "title": r"$\sigma_{\mathrm{vm}}$  ($X_3$ projection)",
            "unit": r"[MPa]",
            "num": von_mises_3d(sigma_num),
            "exact": von_mises_3d(sigma_ex),
            "error": np.abs(von_mises_3d(sigma_num) - von_mises_3d(sigma_ex)),
            "signed": False,
        },
    }

    packed: dict[str, dict[str, object]] = {"n": {"value": n}}
    for key, info in scalar_fields.items():
        normal_axis = int(info["normal_axis"])
        num_grid = interpolate_slice(coords, info["num"], summary, normal_axis=normal_axis)
        exact_grid = interpolate_slice(coords, info["exact"], summary, normal_axis=normal_axis)
        error_grid = interpolate_slice(coords, info["error"], summary, normal_axis=normal_axis)
        packed[key] = {
            "normal_axis": normal_axis,
            "title": info["title"],
            "unit": info["unit"],
            "signed": bool(info["signed"]),
            "num": num_grid,
            "exact": exact_grid,
            "error": error_grid,
        }
    return packed


def axis_labels_for_projection(normal_axis: int) -> tuple[str, str]:
    if normal_axis == 0:
        return r"$X_2$ [cm]", r"$X_3$ [cm]"
    if normal_axis == 1:
        return r"$X_1$ [cm]", r"$X_3$ [cm]"
    return r"$X_1$ [cm]", r"$X_2$ [cm]"


def draw_slice_panel(
    fig: plt.Figure,
    ax,
    cax,
    grid: tuple[np.ndarray, np.ndarray, np.ma.MaskedArray],
    *,
    normal_axis: int,
    radii_cm: tuple[float, float],
    label: str,
    limits: tuple[float, float],
    letter: str,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    GX, GY, field = grid
    cmap = plt.cm.rainbow.copy()
    cmap.set_bad("white")
    mesh = ax.pcolormesh(
        GX,
        GY,
        np.clip(field, limits[0], limits[1]),
        cmap=cmap,
        shading="auto",
        vmin=limits[0],
        vmax=limits[1],
        rasterized=True,
    )

    center_cm = LENGTH_SCALE_CM * CENTER[0]
    for radius in radii_cm:
        ax.add_patch(
            Circle((center_cm, center_cm), radius, fill=False, edgecolor="black", linewidth=0.75)
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(np.min(GX), np.max(GX))
    ax.set_ylim(np.min(GY), np.max(GY))
    ax.set_xticks([2.5, 5.0, 7.5])
    ax.set_yticks([2.5, 5.0, 7.5])
    ax.tick_params(top=True, right=True, labelbottom=show_xlabel, labelleft=show_ylabel)
    xlabel, ylabel = axis_labels_for_projection(normal_axis)
    ax.set_xlabel(xlabel if show_xlabel else "")
    ax.set_ylabel(ylabel if show_ylabel else "")
    panel_label(ax, letter)

    cbar = fig.colorbar(mesh, cax=cax, format=FuncFormatter(compact_sci))
    cbar.ax.tick_params(direction="in", length=2.4, width=0.55, labelsize=6.8, pad=1.0)
    cbar.outline.set_linewidth(0.65)
    cbar.set_ticks(np.linspace(limits[0], limits[1], 3))
    cbar.set_label(label, fontsize=7.2, labelpad=1.0)


def make_figure(data_dir: Path, output_dir: Path, basename: str, *, comparison_column=None) -> None:
    configure_style()
    summary = json.loads(
        (data_dir / "spherical_shell_radial_bvp_summary.json").read_text(encoding="utf-8")
    )
    packed = build_slice_fields(data_dir, summary)
    radii_cm = (
        LENGTH_SCALE_CM * float(summary["geometry"]["inner_radius"]),
        LENGTH_SCALE_CM * float(summary["geometry"]["outer_radius"]),
    )

    field_rows = ("u", "sigma_rr", "sigma_vm")
    columns = (
        ("num", "num"),
        ("exact", "exact"),
        ("error", "error"),
    )
    limits: dict[tuple[str, str], tuple[float, float]] = {}
    for field_key in field_rows:
        info = packed[field_key]
        num_field = info["num"][2]
        exact_field = info["exact"][2]
        error_field = info["error"][2]
        shared_limits = field_limits(
            masked_finite_values(num_field, exact_field), zero_min=not bool(info["signed"])
        )
        if bool(info["signed"]):
            shared_limits = (min(shared_limits[0], 0.0), max(shared_limits[1], 0.0))
        limits[(field_key, "num")] = shared_limits
        limits[(field_key, "exact")] = shared_limits
        limits[(field_key, "error")] = field_limits(
            masked_finite_values(error_field), zero_min=True
        )

    fig_width, fig_height = 7.25, 6.65
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    main_w = 0.212
    main_h = main_w * fig_width / fig_height
    cbar_w = 0.009
    cbar_pad = 0.005
    xs = (0.065, 0.382, 0.699)
    ys = (0.670, 0.385, 0.100)

    for row_index, field_key in enumerate(field_rows):
        info = packed[field_key]

        for col_index, (column_key, column_label) in enumerate(columns):
            x0 = xs[col_index]
            y0 = ys[row_index]
            ax = fig.add_axes([x0, y0, main_w, main_h])
            cax = fig.add_axes([x0 + main_w + cbar_pad, y0, cbar_w, main_h])
            draw_slice_panel(
                fig,
                ax,
                cax,
                info[column_key],
                normal_axis=int(info["normal_axis"]),
                radii_cm=radii_cm,
                label=str(info["unit"]),
                limits=limits[(field_key, column_key)],
                letter=f"({chr(ord('a') + row_index * len(columns) + col_index)})",
                show_xlabel=row_index == 2,
                show_ylabel=col_index == 0,
            )
            if row_index == 0:
                ax.set_title(column_label, pad=7.0, fontsize=8.8)
            if col_index == 0:
                ax.text(
                    -0.34,
                    0.5,
                    str(info["title"]),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=8.8,
                    clip_on=False,
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_bbox = "tight" if comparison_column is None else comparison_column(fig)
    fig.savefig(output_dir / f"{basename}.pdf", bbox_inches=save_bbox)
    fig.savefig(output_dir / f"{basename}.png", bbox_inches=save_bbox)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--basename", default="fig7_spherical_shell_radial_bvp")
    args = parser.parse_args()
    make_figure(Path(args.data_dir), Path(args.output_dir), str(args.basename))


if __name__ == "__main__":
    main()
