#!/usr/bin/env python3
"""Draw CMAME-style Fig. 1 for the analytic Dirichlet annulus benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import FixedLocator, NullFormatter
import numpy as np
from scipy.interpolate import griddata


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR

CENTER = (0.5, 0.5)
RI = 0.18
RO = 0.42
T_END = 0.20
A_FINAL = np.array([[0.35, 0.18], [-0.10, 0.24]], dtype=np.float64)
LENGTH_SCALE_CM = 10.0


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "legend.fontsize": 7.0,
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


def exact_displacement(x: np.ndarray, y: np.ndarray, time: float) -> tuple[np.ndarray, np.ndarray]:
    tau = float(time) / T_END
    dx = x - CENTER[0]
    dy = y - CENTER[1]
    ux = tau * (A_FINAL[0, 0] * dx + A_FINAL[0, 1] * dy)
    uy = tau * (A_FINAL[1, 0] * dx + A_FINAL[1, 1] * dy)
    return ux, uy


def boundary_points(samples: int = 720) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    xo = CENTER[0] + RO * np.cos(theta)
    yo = CENTER[1] + RO * np.sin(theta)
    xi = CENTER[0] + RI * np.cos(theta)
    yi = CENTER[1] + RI * np.sin(theta)
    return xo, yo, xi, yi


def interpolate_annulus(
    X: np.ndarray,
    Y: np.ndarray,
    active: np.ndarray,
    values: np.ndarray,
    boundary_values: tuple[np.ndarray, np.ndarray],
    *,
    res: int = 700,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    xo, yo, xi, yi = boundary_points()
    points = np.column_stack([X[active].ravel(), Y[active].ravel()])
    vals = values[active].ravel()
    points = np.vstack([points, np.column_stack([xo, yo]), np.column_stack([xi, yi])])
    vals = np.concatenate([vals, boundary_values[0].ravel(), boundary_values[1].ravel()])

    gx = np.linspace(CENTER[0] - RO, CENTER[0] + RO, res)
    gy = np.linspace(CENTER[1] - RO, CENTER[1] + RO, res)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    radius = np.sqrt((GX - CENTER[0]) ** 2 + (GY - CENTER[1]) ** 2)
    inside = (radius >= RI) & (radius <= RO)

    Z = griddata(points, vals, (GX, GY), method="cubic")
    if np.isnan(Z[inside]).any():
        Z_linear = griddata(points, vals, (GX, GY), method="linear")
        Z = np.where(np.isnan(Z), Z_linear, Z)
    if np.isnan(Z[inside]).any():
        Z_nearest = griddata(points, vals, (GX, GY), method="nearest")
        Z = np.where(np.isnan(Z), Z_nearest, Z)
    return GX.T, GY.T, np.ma.array(Z.T, mask=(~inside).T)


def von_mises_2d(sigma: np.ndarray) -> np.ndarray:
    sxx = sigma[0]
    syy = sigma[3]
    sxy = 0.5 * (sigma[1] + sigma[2])
    return np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))


def exact_cauchy_stress(time: float, *, model: str, lam: float, mu: float) -> np.ndarray:
    tau = float(time) / T_END
    F = np.eye(2, dtype=np.float64) + tau * A_FINAL
    if model != "neo_hooke":
        raise ValueError(f"exact stress in this figure is implemented for neo_hooke, got {model!r}")
    j = float(np.linalg.det(F))
    H = np.array([[F[1, 1], -F[1, 0]], [-F[0, 1], F[0, 0]]], dtype=np.float64) / j
    P = float(mu) * (F - H) + 0.5 * float(lam) * (j * j - 1.0) * H
    sigma = P @ F.T / j
    return np.array([sigma[0, 0], sigma[0, 1], sigma[1, 0], sigma[1, 1]], dtype=np.float64)


def levels_for(z: np.ma.MaskedArray, *, n: int = 96) -> np.ndarray:
    vals = z.compressed()
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [0.5, 99.5])
    if math.isclose(float(lo), float(hi), rel_tol=1.0e-10, abs_tol=1.0e-14):
        pad = max(abs(float(lo)), 1.0) * 0.01
        lo -= pad
        hi += pad
    return np.linspace(float(lo), float(hi), n)


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


def draw_annulus_outline(ax: plt.Axes) -> None:
    center_cm = (LENGTH_SCALE_CM * CENTER[0], LENGTH_SCALE_CM * CENTER[1])
    ri_cm = LENGTH_SCALE_CM * RI
    ro_cm = LENGTH_SCALE_CM * RO
    ax.add_patch(Circle(center_cm, ro_cm, fill=False, ec="black", lw=0.7))
    ax.add_patch(Circle(center_cm, ri_cm, fill=False, ec="black", lw=0.7))
    ax.set_aspect("equal", adjustable="box")
    pad = LENGTH_SCALE_CM * 0.018
    ax.set_xlim(center_cm[0] - ro_cm - pad, center_cm[0] + ro_cm + pad)
    ax.set_ylim(center_cm[1] - ro_cm - pad, center_cm[1] + ro_cm + pad)
    ax.set_xlabel(r"$X_1$ [cm]")
    ax.set_ylabel(r"$X_2$ [cm]")
    ax.tick_params(top=True, right=True)


def draw_field_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ma.MaskedArray,
    *,
    cbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_format: str | None = None,
) -> None:
    levels = levels_for(Z)
    lo = float(levels[0]) if vmin is None else float(vmin)
    hi = float(levels[-1]) if vmax is None else float(vmax)
    cmap = plt.get_cmap("rainbow").copy()
    cmap.set_bad(alpha=0.0)
    cf = ax.pcolormesh(
        X,
        Y,
        Z,
        cmap=cmap,
        shading="auto",
        vmin=lo,
        vmax=hi,
        rasterized=True,
    )
    draw_annulus_outline(ax)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.048, pad=0.025, format=cbar_format)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(direction="in", length=2.5, width=0.55)


def draw_convergence(ax: plt.Axes, summary: dict) -> None:
    rows = summary["rows"]
    h = LENGTH_SCALE_CM * np.array([float(row["dx"]) for row in rows])
    colors = plt.cm.rainbow(np.linspace(0.08, 0.82, 3))
    metrics = [
        ("rel_l2_u", r"$\mathbf{u}$", "o", colors[0]),
        ("rel_l2_sigma", r"$\boldsymbol{\sigma}$", "s", colors[1]),
        ("rel_l2_F_minus_I", r"$\mathbf{F}-\mathbf{I}$", "^", colors[2]),
    ]
    for key, label, marker, color in metrics:
        err = np.array([float(row[key]) for row in rows])
        ax.loglog(h, err, marker=marker, ms=3.5, lw=1.05, color=color, label=label)
    ref = 0.55 * np.array([float(row["rel_l2_u"]) for row in rows])[-1] * (h / h[-1]) ** 2
    ax.loglog(h, ref, "--", color="0.15", lw=0.85, label=r"$O(\Delta X^2)$")
    ax.invert_xaxis()
    ax.set_xlim(float(np.max(h)) * 1.12, float(np.min(h)) * 0.72)
    ax.set_xlabel(r"$\Delta X$ [cm]", fontsize=11.5)
    ax.set_ylabel(r"$E_{L2}$", fontsize=12.5)
    ax.grid(True, which="both", color="0.82", lw=0.35, ls=":")
    ax.tick_params(top=True, right=True)
    ax.xaxis.set_major_locator(
        FixedLocator(
            [
                LENGTH_SCALE_CM / 64.0,
                LENGTH_SCALE_CM / 128.0,
                LENGTH_SCALE_CM / 256.0,
                LENGTH_SCALE_CM / 384.0,
            ]
        )
    )
    ax.set_xticklabels([r"$0.156$", r"$0.0781$", r"$0.0391$", r"$0.0260$"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(frameon=False, loc="upper right", handlelength=1.5, fontsize=8.5)


def make_figure(data_dir: Path, output_dir: Path, basename: str) -> None:
    configure_style()
    summary = json.loads(
        (data_dir / "annulus_affine_compat_convergence_summary.json").read_text(encoding="utf-8")
    )
    data = np.load(data_dir / "annulus_affine_compat_n384_final.npz")
    X = data["x"]
    Y = data["y"]
    active = data["active"].astype(bool)
    u = data["u"]
    time = float(data["time"])

    ux_exact, uy_exact = exact_displacement(X, Y, time)
    u1 = u[0]
    u2 = u[1]
    u1_err = np.abs(u[0] - ux_exact)
    u2_err = np.abs(u[1] - uy_exact)

    xo, yo, xi, yi = boundary_points()
    uxo, uyo = exact_displacement(xo, yo, time)
    uxi, uyi = exact_displacement(xi, yi, time)
    u1_boundary_outer = uxo
    u1_boundary_inner = uxi
    u2_boundary_outer = uyo
    u2_boundary_inner = uyi
    zero_outer = np.zeros_like(xo)
    zero_inner = np.zeros_like(xi)

    GXu1, GYu1, Zu1 = interpolate_annulus(X, Y, active, u1, (u1_boundary_outer, u1_boundary_inner))
    GXu1e, GYu1e, Zu1e = interpolate_annulus(X, Y, active, u1_err, (zero_outer, zero_inner))
    GXu2, GYu2, Zu2 = interpolate_annulus(X, Y, active, u2, (u2_boundary_outer, u2_boundary_inner))
    GXu2e, GYu2e, Zu2e = interpolate_annulus(X, Y, active, u2_err, (zero_outer, zero_inner))

    GXu1 *= LENGTH_SCALE_CM
    GYu1 *= LENGTH_SCALE_CM
    GXu1e *= LENGTH_SCALE_CM
    GYu1e *= LENGTH_SCALE_CM
    GXu2 *= LENGTH_SCALE_CM
    GYu2 *= LENGTH_SCALE_CM
    GXu2e *= LENGTH_SCALE_CM
    GYu2e *= LENGTH_SCALE_CM
    Zu1 *= LENGTH_SCALE_CM
    Zu1e *= LENGTH_SCALE_CM
    Zu2 *= LENGTH_SCALE_CM
    Zu2e *= LENGTH_SCALE_CM

    fig = plt.figure(figsize=(7.25, 4.05), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.9], height_ratios=[1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[:, 2])

    u_lim = LENGTH_SCALE_CM * max(
        float(np.percentile(np.abs(u1[active]), 99.5)),
        float(np.percentile(np.abs(u2[active]), 99.5)),
    )
    draw_field_panel(
        fig,
        ax0,
        GXu1,
        GYu1,
        Zu1,
        cbar_label=r"$u_{\mathrm{num},1}$ [cm]",
        vmin=-u_lim,
        vmax=u_lim,
    )
    draw_field_panel(
        fig,
        ax1,
        GXu1e,
        GYu1e,
        Zu1e,
        cbar_label=r"$|u_{\mathrm{num},1}-u_{\mathrm{ex},1}|$ [cm]",
        cbar_format="%.1e",
    )
    draw_field_panel(
        fig,
        ax2,
        GXu2,
        GYu2,
        Zu2,
        cbar_label=r"$u_{\mathrm{num},2}$ [cm]",
        vmin=-u_lim,
        vmax=u_lim,
    )
    draw_field_panel(
        fig,
        ax3,
        GXu2e,
        GYu2e,
        Zu2e,
        cbar_label=r"$|u_{\mathrm{num},2}-u_{\mathrm{ex},2}|$ [cm]",
        cbar_format="%.1e",
    )
    draw_convergence(ax4, summary)

    panel_label(ax0, "(a)")
    panel_label(ax1, "(b)")
    panel_label(ax2, "(c)")
    panel_label(ax3, "(d)")
    panel_label(ax4, "(e)")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{basename}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--basename", default="fig1_dirichlet_annulus")
    args = parser.parse_args()
    make_figure(Path(args.data_dir), Path(args.output_dir), str(args.basename))


if __name__ == "__main__":
    main()
