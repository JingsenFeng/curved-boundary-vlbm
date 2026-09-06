#!/usr/bin/env python3
"""Draw CMAME-style Fig. 6 for the 3-D ellipsoid Dirichlet benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR
LENGTH_SCALE_CM = 10.0

CENTER = np.array([0.5, 0.5, 0.5], dtype=np.float64)
AXES = np.array([0.34, 0.25, 0.18], dtype=np.float64)
A_MATRIX = np.array(
    [
        [0.00, 0.80, 0.00],
        [0.00, 0.00, -0.60],
        [0.45, 0.00, 0.08],
    ],
    dtype=np.float64,
)


def rotation_matrix() -> np.ndarray:
    rz = math.radians(25.0)
    ry = math.radians(18.0)
    Rz = np.array(
        [
            [math.cos(rz), -math.sin(rz), 0.0],
            [math.sin(rz), math.cos(rz), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    Ry = np.array(
        [
            [math.cos(ry), 0.0, math.sin(ry)],
            [0.0, 1.0, 0.0],
            [-math.sin(ry), 0.0, math.cos(ry)],
        ],
        dtype=np.float64,
    )
    return Rz @ Ry


ROT = rotation_matrix()


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
        "fontsize": 11.5,
        "fontweight": "bold",
        "bbox": {"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    }
    if is_3d:
        ax.text2D(0.02, 0.98, text, transform=ax.transAxes, **kwargs)
    else:
        ax.text(0.02, 0.98, text, transform=ax.transAxes, **kwargs)


def ellipsoid_phi_array(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    X, Y, Z = np.broadcast_arrays(X, Y, Z)
    dX = np.stack([X - CENTER[0], Y - CENTER[1], Z - CENTER[2]], axis=0)
    xi = np.einsum("ab,b...->a...", ROT.T, dX)
    return np.sqrt(np.sum((xi / AXES[(slice(None),) + (None,) * X.ndim]) ** 2, axis=0)) - 1.0


def exact_displacement(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, time: float) -> np.ndarray:
    dX = np.stack([X - CENTER[0], Y - CENTER[1], Z - CENTER[2]], axis=0)
    return float(time) * np.einsum("ab,b...->a...", A_MATRIX, dX)


def ellipsoid_surface(
    time: float, *, n_theta: int = 96, n_phi: int = 192
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    local = np.stack(
        [
            AXES[0] * np.sin(TH) * np.cos(PH),
            AXES[1] * np.sin(TH) * np.sin(PH),
            AXES[2] * np.cos(TH),
        ],
        axis=0,
    )
    X0 = CENTER[:, None, None] + np.einsum("ab,bmn->amn", ROT, local)
    u = exact_displacement(X0[0], X0[1], X0[2], time)
    return X0, X0 + u, u


def set_equal_3d(ax, points: np.ndarray, *, pad: float = 0.025, zoom: float = 1.28) -> None:
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


def cbar_ticks(cbar, *, side: str = "right") -> None:
    cbar.ax.yaxis.set_ticks_position(side)
    cbar.ax.yaxis.set_label_position(side)
    cbar.ax.tick_params(direction="in", length=2.5, width=0.55, labelsize=7.0, pad=1.2)
    cbar.outline.set_linewidth(0.65)


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


def projected_zlabel(ax, label: str, *, dx_px: float = 15.0) -> None:
    """Place a z-label at a fixed screen offset from the projected visible z-axis."""
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


def format_dx_tick(value: float, _position: int | None = None) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def set_3d_cm_ticks(ax) -> None:
    ax.set_xticks([2.0, 4.0, 6.0, 8.0])
    ax.set_yticks([2.0, 4.0, 6.0, 8.0])
    ax.set_zticks([4.0, 5.0, 6.0])


def load_state(data_dir: Path, summary: dict) -> tuple[int, dict[str, np.ndarray | float]]:
    rows = sorted(summary["rows"], key=lambda row: int(row["n"]))
    n = int(rows[-1]["n"])
    data = np.load(data_dir / f"ellipsoid_affine_dirichlet_n{n}_final.npz")
    storage = str(data["storage"]) if "storage" in data.files else "dense"
    if storage == "sparse_active" and "x_active" in data.files:
        X = data["x_active"]
        Y = data["y_active"]
        Z = data["z_active"]
        active = np.ones_like(X, dtype=bool)
        dx = float(data["dx"])
    else:
        X = data["x"]
        Y = data["y"]
        Z = data["z"]
        active = data["active"].astype(bool)
        dx = float(rows[-1]["dx"])
    return n, {
        "X": X,
        "Y": Y,
        "Z": Z,
        "active": active,
        "u": data["u"],
        "time": float(data["time"]),
        "dx": dx,
    }


def numerical_surface_fields(
    state: dict[str, np.ndarray | float],
    *,
    n_theta: int = 92,
    n_phi: int = 184,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = state["X"]
    Y = state["Y"]
    Z = state["Z"]
    u = state["u"]
    active = np.asarray(state["active"], dtype=bool)
    time = float(state["time"])
    X0, _Xd_exact, u_exact = ellipsoid_surface(time, n_theta=n_theta, n_phi=n_phi)

    mask = active.ravel()
    coords = np.column_stack([X.ravel()[mask], Y.ravel()[mask], Z.ravel()[mask]])
    u_active = u.reshape(3, -1)[:, mask]
    tree = cKDTree(coords)
    query = X0.reshape(3, -1).T
    dist, idx = tree.query(query, k=8)
    weights = 1.0 / np.maximum(dist, 1.0e-12) ** 2
    weights /= np.sum(weights, axis=1, keepdims=True)
    u_num_flat = np.sum(u_active[:, idx] * weights[None, :, :], axis=2)
    u_num = u_num_flat.reshape((3,) + X0.shape[1:])
    for comp in range(3):
        u_num[comp] = gaussian_filter(u_num[comp], sigma=0.45, mode=("nearest", "wrap"))
    Xd_num = X0 + u_num

    return X0, Xd_num, u_num, u_exact


def boundary_error_points(
    state: dict[str, np.ndarray | float],
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = state["X"]
    Y = state["Y"]
    Z = state["Z"]
    u = state["u"]
    active = np.asarray(state["active"], dtype=bool)
    time = float(state["time"])
    dx = float(state["dx"])
    band = (active & (np.abs(ellipsoid_phi_array(X, Y, Z)) <= 2.0 * dx)).ravel()
    if not np.any(band):
        raise RuntimeError("no near-boundary nodes available for Fig. 6")
    Xb = X.ravel()[band]
    Yb = Y.ravel()[band]
    Zb = Z.ravel()[band]
    ub = u.reshape(3, -1)[:, band]
    ub_exact = exact_displacement(Xb, Yb, Zb, time)
    err = np.sqrt(np.sum((ub - ub_exact) * (ub - ub_exact), axis=0))
    if err.size > max_points:
        rng = np.random.default_rng(20260530)
        keep = np.sort(rng.choice(err.size, size=max_points, replace=False))
        Xb = Xb[keep]
        Yb = Yb[keep]
        Zb = Zb[keep]
        ub = ub[:, keep]
        err = err[keep]
    xd = Xb + ub[0]
    yd = Yb + ub[1]
    zd = Zb + ub[2]
    return xd, yd, zd, err


def draw_surface_panel(
    fig: plt.Figure,
    ax,
    cax: plt.Axes,
    X0: np.ndarray,
    Xd: np.ndarray,
    u_num: np.ndarray,
) -> None:
    umag_cm = LENGTH_SCALE_CM * np.sqrt(np.sum(u_num * u_num, axis=0))
    norm = Normalize(float(np.percentile(umag_cm, 0.5)), float(np.percentile(umag_cm, 99.5)))
    colors = plt.cm.rainbow(norm(umag_cm))
    ax.plot_wireframe(
        *(LENGTH_SCALE_CM * X0),
        rstride=12,
        cstride=18,
        color="0.20",
        linewidth=0.22,
        alpha=0.22,
    )
    ax.plot_surface(
        *(LENGTH_SCALE_CM * Xd),
        facecolors=colors,
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=True,
        shade=False,
        alpha=0.96,
        rasterized=True,
    )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap="rainbow")
    mappable.set_array(umag_cm)
    cbar = fig.colorbar(
        mappable, cax=cax, orientation="horizontal", format=FuncFormatter(compact_sci)
    )
    horizontal_cbar_ticks(cbar)
    cbar.set_label(r"$|\mathbf{u}_{\mathrm{num}}|$ [cm]", fontsize=7.2, labelpad=1.0)
    all_points = np.concatenate([X0.reshape(3, -1), Xd.reshape(3, -1)], axis=1)
    set_equal_3d(ax, all_points, pad=0.012, zoom=1.24)
    set_3d_cm_ticks(ax)
    ax.set_xlabel(r"$X_1$ [cm]", labelpad=-4.0)
    ax.set_ylabel(r"$X_2$ [cm]", labelpad=-4.0)
    ax.set_zlabel("")
    ax.tick_params(pad=-3.0)
    clean_3d_axes(ax)
    ax.view_init(elev=22.0, azim=-54.0)
    projected_zlabel(ax, r"$X_3$ [cm]", dx_px=150.0)
    panel_label(ax, "(a)", is_3d=True)


def draw_error_panel(
    fig: plt.Figure,
    ax,
    cax: plt.Axes,
    Xd: np.ndarray,
    state: dict[str, np.ndarray | float],
    *,
    max_points: int,
) -> None:
    xd, yd, zd, err = boundary_error_points(state, max_points=max_points)
    boundary_coords = np.column_stack([xd, yd, zd])
    query = Xd.reshape(3, -1).T
    tree = cKDTree(boundary_coords)
    k = min(12, err.size)
    dist, idx = tree.query(query, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    weights = 1.0 / np.maximum(dist, 1.0e-12) ** 2
    weights /= np.sum(weights, axis=1, keepdims=True)
    e_flat = np.sum((LENGTH_SCALE_CM * err)[idx] * weights, axis=1)
    e_plot = e_flat.reshape(Xd.shape[1:])
    e_plot = gaussian_filter(e_plot, sigma=1.0, mode=("nearest", "wrap"))
    vmin = 0.0
    vmax = float(np.percentile(e_plot, 99.5))
    if vmax <= vmin:
        vmax = float(np.max(e_plot)) if np.max(e_plot) > 0.0 else 1.0e-12
    norm = Normalize(vmin=vmin, vmax=vmax)
    colors = plt.cm.rainbow(norm(np.clip(e_plot, vmin, vmax)))
    ax.plot_surface(
        *(LENGTH_SCALE_CM * Xd),
        facecolors=colors,
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=True,
        shade=False,
        alpha=0.98,
        rasterized=True,
    )
    mappable = plt.cm.ScalarMappable(
        cmap="rainbow",
        norm=norm,
    )
    mappable.set_array(e_plot)
    cbar = fig.colorbar(
        mappable, cax=cax, orientation="horizontal", format=FuncFormatter(compact_sci)
    )
    horizontal_cbar_ticks(cbar)
    cbar.set_ticks(np.linspace(vmin, vmax, 3))
    cbar.set_label(
        r"$|\mathbf{u}_{\mathrm{num}}-\mathbf{u}_{\mathrm{ex}}|$ [cm]", fontsize=7.2, labelpad=1.0
    )
    set_equal_3d(ax, Xd.reshape(3, -1), pad=0.012, zoom=1.24)
    set_3d_cm_ticks(ax)
    ax.set_xlabel(r"$X_1$ [cm]", labelpad=-4.0)
    ax.set_ylabel(r"$X_2$ [cm]", labelpad=-4.0)
    ax.set_zlabel("")
    ax.tick_params(pad=-3.0)
    clean_3d_axes(ax)
    ax.view_init(elev=22.0, azim=-54.0)
    projected_zlabel(ax, r"$X_3$ [cm]", dx_px=150.0)
    panel_label(ax, "(b)", is_3d=True)


def draw_convergence(ax: plt.Axes, summary: dict) -> None:
    rows = sorted(summary["rows"], key=lambda row: int(row["n"]))
    h = LENGTH_SCALE_CM * np.array([float(row["dx"]) for row in rows])
    colors = plt.cm.rainbow(np.linspace(0.10, 0.72, 2))
    metrics = [
        ("rel_l2_u", r"$\mathbf{u}$", "o", colors[0]),
        ("rel_l2_F", r"$\mathbf{F}$", "^", colors[1]),
    ]
    for key, label, marker, color in metrics:
        err = np.array([float(row[key]) for row in rows])
        ax.loglog(h, err, marker=marker, ms=3.4, lw=1.05, color=color, label=label)
    u_err = np.array([float(row["rel_l2_u"]) for row in rows])
    ref = 0.70 * u_err[-1] * (h / h[-1]) ** 2
    ax.loglog(h, ref, "--", color="0.15", lw=0.85, label=r"$O(\Delta X^2)$")
    ax.invert_xaxis()
    ax.set_xlim(float(np.max(h)) * 1.12, float(np.min(h)) * 0.78)
    ax.set_xlabel(r"$\Delta X$ [cm]", fontsize=11.5)
    ax.set_ylabel(r"$E_{L2}$", fontsize=12.5)
    ax.grid(True, which="both", color="0.82", lw=0.35, ls=":")
    ax.tick_params(top=True, right=True)
    tick_values = h[[0, 2, 4]]
    ax.xaxis.set_major_locator(FixedLocator(tick_values))
    ax.xaxis.set_major_formatter(FuncFormatter(format_dx_tick))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(
        frameon=False,
        loc="upper right",
        handlelength=1.55,
        borderpad=0.15,
        labelspacing=0.35,
        fontsize=8.5,
    )
    panel_label(ax, "(c)")


def make_figure(data_dir: Path, output_dir: Path, basename: str, *, max_points: int) -> None:
    configure_style()
    summary = json.loads(
        (data_dir / "ellipsoid_affine_dirichlet_summary.json").read_text(encoding="utf-8")
    )
    _n, state = load_state(data_dir, summary)

    X0, Xd, u_num, _u_exact = numerical_surface_fields(state)

    fig = plt.figure(figsize=(10.20, 3.55), constrained_layout=False)
    # Manual placement is deliberate here: 3-D tick labels and z-axis labels
    # extend outside their Axes boxes, so GridSpec spacing still looks crowded.
    ax0 = fig.add_axes([0.035, 0.275, 0.250, 0.650], projection="3d")
    cax0 = fig.add_axes([0.062, 0.105, 0.196, 0.045])
    ax1 = fig.add_axes([0.392, 0.275, 0.250, 0.650], projection="3d")
    cax1 = fig.add_axes([0.419, 0.105, 0.196, 0.045])
    ax2 = fig.add_axes([0.780, 0.175, 0.205, 0.750])
    draw_surface_panel(fig, ax0, cax0, X0, Xd, u_num)
    draw_error_panel(fig, ax1, cax1, Xd, state, max_points=max_points)
    draw_convergence(ax2, summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{basename}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--basename", default="fig6_ellipsoid_dirichlet_3d")
    parser.add_argument("--max-points", type=int, default=70000)
    args = parser.parse_args()
    make_figure(
        Path(args.data_dir),
        Path(args.output_dir),
        str(args.basename),
        max_points=int(args.max_points),
    )


if __name__ == "__main__":
    main()
