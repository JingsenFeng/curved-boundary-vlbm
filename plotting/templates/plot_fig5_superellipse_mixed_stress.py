#!/usr/bin/env python3
"""Draw CMAME-style Fig. 5 for the large-deformation superellipse stress fields."""

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
STRESS_SCALE_MPA = 10.0


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
    F_lbm = data["F_lbm"]
    F_fem = data["F_fem"]
    outer = Superellipse(*map(float, data["outer"]))
    inner = Superellipse(*map(float, data["inner"]))
    mu, poisson = map(float, data["material"])
    outer_xy, inner_xy, bounds = boundary_points(outer, inner)

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
    vm_err_limits = error_limits(vm_err[mask])
    s22_err_limits = error_limits(s22_err[mask])
    fmt = FuncFormatter(compact_sci)
    fields = [
        (vm_fem, r"$\sigma_{\mathrm{vm},\mathrm{FEM}}$ [MPa]", shared_vm_limits),
        (vm_lbm, r"$\sigma_{\mathrm{vm},\mathrm{LBM}}$ [MPa]", shared_vm_limits),
        (
            vm_err,
            r"$|\sigma_{\mathrm{vm},\mathrm{LBM}}-\sigma_{\mathrm{vm},\mathrm{FEM}}|$ [MPa]",
            vm_err_limits,
        ),
        (s22_fem, r"$\sigma_{22,\mathrm{FEM}}$ [MPa]", shared_s22_limits),
        (s22_lbm, r"$\sigma_{22,\mathrm{LBM}}$ [MPa]", shared_s22_limits),
        (s22_err, r"$|\sigma_{22,\mathrm{LBM}}-\sigma_{22,\mathrm{FEM}}|$ [MPa]", s22_err_limits),
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
    parser.add_argument("--basename", default="fig5_superellipse_mixed_stress")
    args = parser.parse_args()
    make_figure(Path(args.data_dir), Path(args.output_dir), str(args.basename))


if __name__ == "__main__":
    main()
