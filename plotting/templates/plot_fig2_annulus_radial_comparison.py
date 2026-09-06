#!/usr/bin/env python3
"""Draw CMAME-style Fig. 2 for the annulus radial traction comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.interpolate import griddata


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR
LENGTH_SCALE_CM = 10.0


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.65,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def field_limits(
    values: np.ndarray, *, symmetric: bool = False, zero_min: bool = False
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric:
        vmax = max(float(np.max(np.abs(finite))), 1.0e-14)
        return -vmax, vmax
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if zero_min:
        lo = 0.0
    if np.isclose(lo, hi):
        pad = max(abs(float(lo)), 1.0) * 0.02
        lo -= pad
        hi += pad
    return float(lo), float(hi)


def compact_sci(value: float, _position: int | None = None) -> str:
    if not np.isfinite(value) or abs(value) < 1.0e-14:
        return "0"
    if 1.0e-3 <= abs(value) < 1.0e4:
        return f"{value:.3g}"
    text = f"{value:.1e}"
    mantissa, exponent = text.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent):+d}"


def annulus_grid(
    cx: float, cy: float, ri: float, ro: float, res: int = 680
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = np.linspace(cx - ro, cx + ro, res)
    gy = np.linspace(cy - ro, cy + ro, res)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    radius = np.sqrt((GX - cx) ** 2 + (GY - cy) ** 2)
    inside = (radius >= ri) & (radius <= ro)
    return GX, GY, inside


def interpolate_annulus(
    X: np.ndarray,
    Y: np.ndarray,
    mask: np.ndarray,
    values: np.ndarray,
    *,
    cx: float,
    cy: float,
    ri: float,
    ro: float,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    GX, GY, inside = annulus_grid(cx, cy, ri, ro)
    points = np.column_stack([X[mask].ravel(), Y[mask].ravel()])
    vals = values[mask].ravel()
    Z = griddata(points, vals, (GX, GY), method="cubic")
    if np.isnan(Z[inside]).any():
        Z_linear = griddata(points, vals, (GX, GY), method="linear")
        Z = np.where(np.isnan(Z), Z_linear, Z)
    if np.isnan(Z[inside]).any():
        Z_nearest = griddata(points, vals, (GX, GY), method="nearest")
        Z = np.where(np.isnan(Z), Z_nearest, Z)
    return GX.T, GY.T, np.ma.array(Z.T, mask=(~inside).T)


def draw_outline(ax: plt.Axes, *, cx: float, cy: float, ri: float, ro: float) -> None:
    cx_cm = LENGTH_SCALE_CM * cx
    cy_cm = LENGTH_SCALE_CM * cy
    ri_cm = LENGTH_SCALE_CM * ri
    ro_cm = LENGTH_SCALE_CM * ro
    ax.add_patch(Circle((cx_cm, cy_cm), ro_cm, fill=False, ec="black", lw=0.7))
    ax.add_patch(Circle((cx_cm, cy_cm), ri_cm, fill=False, ec="black", lw=0.7))
    pad = LENGTH_SCALE_CM * 0.018
    ax.set_xlim(cx_cm - ro_cm - pad, cx_cm + ro_cm + pad)
    ax.set_ylim(cy_cm - ro_cm - pad, cy_cm + ro_cm + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$X_1$ [cm]")
    ax.set_ylabel(r"$X_2$ [cm]")
    ax.tick_params(top=True, right=True)


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )


def draw_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    cax: plt.Axes,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ma.MaskedArray,
    *,
    cx: float,
    cy: float,
    ri: float,
    ro: float,
    label: str,
    limits: tuple[float, float],
    cbar_format: str | None = None,
) -> None:
    cmap = plt.get_cmap("rainbow").copy()
    cmap.set_bad(alpha=0.0)
    im = ax.pcolormesh(
        X, Y, Z, cmap=cmap, shading="auto", vmin=limits[0], vmax=limits[1], rasterized=True
    )
    draw_outline(ax, cx=cx, cy=cy, ri=ri, ro=ro)
    cbar = fig.colorbar(im, cax=cax, format=cbar_format)
    cbar.set_ticks(np.linspace(limits[0], limits[1], 5))
    cbar.update_ticks()
    cbar.set_label(label, labelpad=1.0)
    cbar.ax.tick_params(direction="in", length=2.5, width=0.55, labelsize=7.2, pad=1.5)


def make_figure(data_dir: Path, output_dir: Path, basename: str, *, comparison_column=None) -> None:
    configure_style()
    data = np.load(data_dir / "annulus_traction_comparison_n196.npz")
    X = data["x"]
    Y = data["y"]
    mask = data["mask"].astype(bool)
    u_lbm = data["u_lbm"]
    u_fem = data["u_fem"]
    F_lbm = data["F_lbm"]
    F_fem = data["F_fem"]
    outer = data["outer"]
    inner = data["inner"]
    cx, cy, ro = float(outer[0]), float(outer[1]), float(outer[2])
    ri = float(inner[2])

    u_fem_mag = LENGTH_SCALE_CM * np.sqrt(np.sum(u_fem * u_fem, axis=0))
    u_lbm_mag = LENGTH_SCALE_CM * np.sqrt(np.sum(u_lbm * u_lbm, axis=0))
    u_err = LENGTH_SCALE_CM * np.sqrt(np.sum((u_lbm - u_fem) ** 2, axis=0))
    identity = np.eye(2, dtype=np.float64)[:, :, None, None]
    F_fem_inc = np.sqrt(np.sum((F_fem - identity) ** 2, axis=(0, 1)))
    F_lbm_inc = np.sqrt(np.sum((F_lbm - identity) ** 2, axis=(0, 1)))
    F_err = np.sqrt(np.sum((F_lbm - F_fem) ** 2, axis=(0, 1)))

    shared_u_limits = field_limits(
        np.concatenate([u_fem_mag[mask], u_lbm_mag[mask]]), zero_min=True
    )
    shared_F_limits = field_limits(
        np.concatenate([F_fem_inc[mask], F_lbm_inc[mask]]), zero_min=True
    )
    u_err_limits = field_limits(u_err[mask], zero_min=True)
    F_err_limits = field_limits(F_err[mask], zero_min=True)

    fields = [
        (
            u_fem_mag,
            r"$|\mathbf{u}_{\mathrm{FEM}}|$ [cm]",
            shared_u_limits,
            FuncFormatter(compact_sci),
        ),
        (
            u_lbm_mag,
            r"$|\mathbf{u}_{\mathrm{LBM}}|$ [cm]",
            shared_u_limits,
            FuncFormatter(compact_sci),
        ),
        (
            u_err,
            r"$\|\mathbf{u}_{\mathrm{LBM}}-\mathbf{u}_{\mathrm{FEM}}\|$ [cm]",
            u_err_limits,
            FuncFormatter(compact_sci),
        ),
        (
            F_fem_inc,
            r"$\|\mathbf{F}_{\mathrm{FEM}}-\mathbf{I}\|_F$",
            shared_F_limits,
            FuncFormatter(compact_sci),
        ),
        (
            F_lbm_inc,
            r"$\|\mathbf{F}_{\mathrm{LBM}}-\mathbf{I}\|_F$",
            shared_F_limits,
            FuncFormatter(compact_sci),
        ),
        (
            F_err,
            r"$\|\mathbf{F}_{\mathrm{LBM}}-\mathbf{F}_{\mathrm{FEM}}\|_F$",
            F_err_limits,
            FuncFormatter(compact_sci),
        ),
    ]

    fig_width, fig_height = 7.25, 4.25
    fig = plt.figure(figsize=(fig_width, fig_height))
    main_w = 0.200
    main_h = main_w * fig_width / fig_height
    cbar_w = 0.009
    cbar_pad = 0.006
    xs = (0.055, 0.382, 0.710)
    ys = (0.560, 0.105)
    axs = []
    caxs = []
    for y0 in ys:
        for x0 in xs:
            axs.append(fig.add_axes([x0, y0, main_w, main_h]))
            caxs.append(fig.add_axes([x0 + main_w + cbar_pad, y0, cbar_w, main_h]))

    for index, (ax, (field, label, limits, fmt), letter) in enumerate(
        zip(axs, fields, ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)"))
    ):
        GX, GY, Z = interpolate_annulus(X, Y, mask, field, cx=cx, cy=cy, ri=ri, ro=ro)
        GX *= LENGTH_SCALE_CM
        GY *= LENGTH_SCALE_CM
        draw_panel(
            fig,
            ax,
            caxs[index],
            GX,
            GY,
            Z,
            cx=cx,
            cy=cy,
            ri=ri,
            ro=ro,
            label=label,
            limits=limits,
            cbar_format=fmt,
        )
        if index % 3:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        panel_label(ax, letter)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_bbox = "tight" if comparison_column is None else comparison_column(fig)
    fig.savefig(output_dir / f"{basename}.pdf", bbox_inches=save_bbox)
    fig.savefig(output_dir / f"{basename}.png", bbox_inches=save_bbox)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--basename", default="fig2_annulus_radial_comparison")
    args = parser.parse_args()
    make_figure(Path(args.data_dir), Path(args.output_dir), str(args.basename))


if __name__ == "__main__":
    main()
