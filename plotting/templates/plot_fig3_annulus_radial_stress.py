#!/usr/bin/env python3
"""Draw CMAME-style Fig. 3 for the annulus radial traction stress fields."""

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
from mpl_toolkits.axes_grid1 import make_axes_locatable


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR
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


def cauchy_stress(F: np.ndarray, mu: float, poisson: float) -> np.ndarray:
    lam = 2.0 * mu * poisson / (1.0 - 2.0 * poisson)
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]

    with np.errstate(divide="ignore", invalid="ignore"):
        FinvT = np.empty_like(F)
        FinvT[0, 0] = F[1, 1] / J
        FinvT[0, 1] = -F[1, 0] / J
        FinvT[1, 0] = -F[0, 1] / J
        FinvT[1, 1] = F[0, 0] / J

        P = mu * (F - FinvT) + 0.5 * lam * (J * J - 1.0) * FinvT
        sigma = np.einsum("ia...,ja...->ij...", P, F) / J
    return sigma


def von_mises_2d(sigma: np.ndarray) -> np.ndarray:
    s11 = sigma[0, 0]
    s22 = sigma[1, 1]
    s12 = 0.5 * (sigma[0, 1] + sigma[1, 0])
    vm2 = s11 * s11 - s11 * s22 + s22 * s22 + 3.0 * s12 * s12
    return np.sqrt(np.maximum(vm2, 0.0))


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
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.0%", pad=0.028)
    cbar = fig.colorbar(im, cax=cax, format=cbar_format)
    cbar.set_ticks(np.linspace(limits[0], limits[1], 5))
    cbar.update_ticks()
    cbar.set_label(label, labelpad=1.5)
    cbar.ax.tick_params(direction="in", length=2.5, width=0.55, labelsize=7.2, pad=1.5)


def make_figure(data_dir: Path, output_dir: Path, basename: str, *, comparison_column=None) -> None:
    configure_style()
    data = np.load(data_dir / "annulus_traction_comparison_n196.npz")
    X = data["x"]
    Y = data["y"]
    mask = data["mask"].astype(bool)
    F_lbm = data["F_lbm"]
    F_fem = data["F_fem"]
    outer = data["outer"]
    inner = data["inner"]
    mu, poisson = map(float, data["material"])
    cx, cy, ro = float(outer[0]), float(outer[1]), float(outer[2])
    ri = float(inner[2])

    sigma_lbm = STRESS_SCALE_MPA * cauchy_stress(F_lbm, mu, poisson)
    sigma_fem = STRESS_SCALE_MPA * cauchy_stress(F_fem, mu, poisson)
    vm_lbm = von_mises_2d(sigma_lbm)
    vm_fem = von_mises_2d(sigma_fem)
    vm_err = np.abs(vm_lbm - vm_fem)
    s22_lbm = sigma_lbm[1, 1]
    s22_fem = sigma_fem[1, 1]
    s22_err = np.abs(s22_lbm - s22_fem)

    shared_vm_limits = field_limits(np.concatenate([vm_fem[mask], vm_lbm[mask]]), zero_min=True)
    shared_s22_limits = field_limits(np.concatenate([s22_fem[mask], s22_lbm[mask]]), symmetric=True)
    vm_err_limits = field_limits(vm_err[mask], zero_min=True)
    s22_err_limits = field_limits(s22_err[mask], zero_min=True)

    fields = [
        (
            vm_fem,
            r"$\sigma_{\mathrm{vm},\mathrm{FEM}}$ [MPa]",
            shared_vm_limits,
            FuncFormatter(compact_sci),
        ),
        (
            vm_lbm,
            r"$\sigma_{\mathrm{vm},\mathrm{LBM}}$ [MPa]",
            shared_vm_limits,
            FuncFormatter(compact_sci),
        ),
        (
            vm_err,
            r"$|\sigma_{\mathrm{vm},\mathrm{LBM}}-\sigma_{\mathrm{vm},\mathrm{FEM}}|$ [MPa]",
            vm_err_limits,
            FuncFormatter(compact_sci),
        ),
        (
            s22_fem,
            r"$\sigma_{22,\mathrm{FEM}}$ [MPa]",
            shared_s22_limits,
            FuncFormatter(compact_sci),
        ),
        (
            s22_lbm,
            r"$\sigma_{22,\mathrm{LBM}}$ [MPa]",
            shared_s22_limits,
            FuncFormatter(compact_sci),
        ),
        (
            s22_err,
            r"$|\sigma_{22,\mathrm{LBM}}-\sigma_{22,\mathrm{FEM}}|$ [MPa]",
            s22_err_limits,
            FuncFormatter(compact_sci),
        ),
    ]

    fig, axs = plt.subplots(2, 3, figsize=(7.25, 4.25), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.105, top=0.985, wspace=0.08, hspace=0.30)
    for index, (ax, (field, label, limits, fmt), letter) in enumerate(
        zip(axs.ravel(), fields, ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)"))
    ):
        GX, GY, Z = interpolate_annulus(X, Y, mask, field, cx=cx, cy=cy, ri=ri, ro=ro)
        GX *= LENGTH_SCALE_CM
        GY *= LENGTH_SCALE_CM
        draw_panel(
            fig,
            ax,
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
    parser.add_argument("--basename", default="fig3_annulus_radial_stress")
    args = parser.parse_args()
    make_figure(Path(args.data_dir), Path(args.output_dir), str(args.basename))


if __name__ == "__main__":
    main()
