#!/usr/bin/env python3
"""Draw CMAME-style Fig. 4 for the large-deformation superellipse mixed case."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR
LENGTH_SCALE_CM = 10.0


class Superellipse:
    def __init__(
        self, cx: float, cy: float, a: float, b: float, theta_deg: float, p: float
    ) -> None:
        self.cx = float(cx)
        self.cy = float(cy)
        self.a = float(a)
        self.b = float(b)
        self.theta = math.radians(float(theta_deg))
        self.p = float(p)

    def local(self, x: np.ndarray | float, y: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        ct = math.cos(self.theta)
        st = math.sin(self.theta)
        dx = np.asarray(x, dtype=np.float64) - self.cx
        dy = np.asarray(y, dtype=np.float64) - self.cy
        xi = ct * dx + st * dy
        eta = -st * dx + ct * dy
        return xi, eta

    def phi(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        xi, eta = self.local(x, y)
        return (np.abs(xi) / self.a) ** self.p + (np.abs(eta) / self.b) ** self.p - 1.0

    def point(self, t: float) -> tuple[float, float]:
        ct = math.cos(self.theta)
        st = math.sin(self.theta)
        c = math.cos(t)
        s = math.sin(t)
        xi = self.a * math.copysign(abs(c) ** (2.0 / self.p), c)
        eta = self.b * math.copysign(abs(s) ** (2.0 / self.p), s)
        x = self.cx + ct * xi - st * eta
        y = self.cy + st * xi + ct * eta
        return x, y


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


def compact_sci(value: float, _position: int | None = None) -> str:
    if not np.isfinite(value) or abs(value) < 1.0e-14:
        return "0"
    if 1.0e-3 <= abs(value) < 1.0e4:
        return f"{value:.3g}"
    text = f"{value:.1e}"
    mantissa, exponent = text.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent):+d}"


def field_limits(
    values: np.ndarray, *, zero_min: bool = False, symmetric: bool = False
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


def error_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    hi = max(float(np.max(finite)), 1.0e-14)
    return 0.0, hi


def shell_mask(
    x: np.ndarray, y: np.ndarray, outer: Superellipse, inner: Superellipse
) -> np.ndarray:
    return np.maximum(outer.phi(x, y), -inner.phi(x, y)) <= 0.0


def boundary_points(
    outer: Superellipse, inner: Superellipse
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    t = np.linspace(0.0, 2.0 * math.pi, 1000)
    outer_xy = np.asarray([outer.point(float(v)) for v in t])
    inner_xy = np.asarray([inner.point(float(v)) for v in t])
    xmin = min(float(np.min(outer_xy[:, 0])), float(np.min(inner_xy[:, 0]))) - 0.03
    xmax = max(float(np.max(outer_xy[:, 0])), float(np.max(inner_xy[:, 0]))) + 0.03
    ymin = min(float(np.min(outer_xy[:, 1])), float(np.min(inner_xy[:, 1]))) - 0.03
    ymax = max(float(np.max(outer_xy[:, 1])), float(np.max(inner_xy[:, 1]))) + 0.03
    return outer_xy, inner_xy, (xmin, xmax, ymin, ymax)


def interpolate_shell(
    X: np.ndarray,
    Y: np.ndarray,
    mask: np.ndarray,
    values: np.ndarray,
    outer: Superellipse,
    inner: Superellipse,
    bounds: tuple[float, float, float, float],
    *,
    res: int = 680,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    xmin, xmax, ymin, ymax = bounds
    gx = np.linspace(xmin, xmax, res)
    gy = np.linspace(ymin, ymax, res)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    inside = shell_mask(GX, GY, outer, inner)
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


def draw_outline(
    ax: plt.Axes,
    outer_xy: np.ndarray,
    inner_xy: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> None:
    xmin, xmax, ymin, ymax = bounds
    ax.plot(
        LENGTH_SCALE_CM * outer_xy[:, 0], LENGTH_SCALE_CM * outer_xy[:, 1], color="black", lw=0.7
    )
    ax.plot(
        LENGTH_SCALE_CM * inner_xy[:, 0], LENGTH_SCALE_CM * inner_xy[:, 1], color="black", lw=0.7
    )
    ax.set_xlim(LENGTH_SCALE_CM * xmin, LENGTH_SCALE_CM * xmax)
    ax.set_ylim(LENGTH_SCALE_CM * ymin, LENGTH_SCALE_CM * ymax)
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
    outer_xy: np.ndarray,
    inner_xy: np.ndarray,
    bounds: tuple[float, float, float, float],
    label: str,
    limits: tuple[float, float],
    cbar_format: FuncFormatter,
) -> None:
    cmap = plt.get_cmap("rainbow").copy()
    cmap.set_bad(alpha=0.0)
    im = ax.pcolormesh(
        X, Y, Z, cmap=cmap, shading="auto", vmin=limits[0], vmax=limits[1], rasterized=True
    )
    draw_outline(ax, outer_xy, inner_xy, bounds)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.0%", pad=0.028)
    cbar = fig.colorbar(im, cax=cax, format=cbar_format)
    cbar.set_ticks(np.linspace(limits[0], limits[1], 5))
    cbar.update_ticks()
    cbar.set_label(label, labelpad=0.8)
    cbar.ax.tick_params(direction="in", length=2.5, width=0.55, labelsize=7.0, pad=1.2)


def make_figure(data_dir: Path, output_dir: Path, basename: str, *, comparison_column=None) -> None:
    configure_style()
    data = np.load(data_dir / "superellipse_mixed_comparison_n384.npz")
    X = data["x"]
    Y = data["y"]
    mask = data["mask"].astype(bool)
    u_lbm = data["u_lbm"]
    u_fem = data["u_fem"]
    F_lbm = data["F_lbm"]
    F_fem = data["F_fem"]
    outer = Superellipse(*map(float, data["outer"]))
    inner = Superellipse(*map(float, data["inner"]))
    outer_xy, inner_xy, bounds = boundary_points(outer, inner)

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
    u_err_limits = error_limits(u_err[mask])
    F_err_limits = error_limits(F_err[mask])
    fmt = FuncFormatter(compact_sci)
    fields = [
        (u_fem_mag, r"$|\mathbf{u}_{\mathrm{FEM}}|$ [cm]", shared_u_limits),
        (u_lbm_mag, r"$|\mathbf{u}_{\mathrm{LBM}}|$ [cm]", shared_u_limits),
        (u_err, r"$\|\mathbf{u}_{\mathrm{LBM}}-\mathbf{u}_{\mathrm{FEM}}\|$ [cm]", u_err_limits),
        (F_fem_inc, r"$\|\mathbf{F}_{\mathrm{FEM}}-\mathbf{I}\|_F$", shared_F_limits),
        (F_lbm_inc, r"$\|\mathbf{F}_{\mathrm{LBM}}-\mathbf{I}\|_F$", shared_F_limits),
        (F_err, r"$\|\mathbf{F}_{\mathrm{LBM}}-\mathbf{F}_{\mathrm{FEM}}\|_F$", F_err_limits),
    ]

    fig, axs = plt.subplots(2, 3, figsize=(7.25, 4.25), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.105, top=0.985, wspace=0.24, hspace=0.30)
    for index, (ax, (field, label, limits), letter) in enumerate(
        zip(axs.ravel(), fields, ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)"))
    ):
        GX, GY, Z = interpolate_shell(X, Y, mask, field, outer, inner, bounds)
        GX *= LENGTH_SCALE_CM
        GY *= LENGTH_SCALE_CM
        draw_panel(
            fig,
            ax,
            GX,
            GY,
            Z,
            outer_xy=outer_xy,
            inner_xy=inner_xy,
            bounds=bounds,
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
    parser.add_argument("--basename", default="fig4_superellipse_mixed_comparison")
    args = parser.parse_args()
    make_figure(Path(args.data_dir), Path(args.output_dir), str(args.basename))


if __name__ == "__main__":
    main()
