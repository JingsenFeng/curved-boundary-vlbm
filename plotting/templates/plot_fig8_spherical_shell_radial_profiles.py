#!/usr/bin/env python3
"""Draw Fig. 8 radial profiles for the 3-D spherical shell benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
FIG7_DIR = SCRIPT_DIR.parent / "fig7_spherical_shell_radial_bvp"
DATA_DIR = FIG7_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR
LENGTH_SCALE_CM = 10.0
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
            "legend.fontsize": 7.5,
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


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.025,
        0.965,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )


def radial_stretches(F: np.ndarray, radial_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lambda_r = np.einsum("in,ijn,jn->n", radial_unit, F, radial_unit)
    trace_F = F[0, 0] + F[1, 1] + F[2, 2]
    lambda_theta = 0.5 * (trace_F - lambda_r)
    return lambda_r, lambda_theta


def radial_bin_mean(
    radius: np.ndarray, values: np.ndarray, ri: float, ro: float, *, nbins: int = 92
) -> tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(ri, ro, nbins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    which = np.digitize(radius, bins) - 1
    out = np.full(centers.shape, np.nan, dtype=np.float64)
    for index in range(centers.size):
        mask = which == index
        if np.any(mask):
            out[index] = float(np.mean(values[mask]))
    valid = np.isfinite(out)
    return LENGTH_SCALE_CM * centers[valid], out[valid]


def load_profiles(data_dir: Path) -> dict[str, np.ndarray]:
    summary = json.loads(
        (data_dir / "spherical_shell_radial_bvp_summary.json").read_text(encoding="utf-8")
    )
    rows = sorted(summary["rows"], key=lambda row: int(row["n"]))
    n = int(rows[-1]["n"])
    num = np.load(data_dir / f"spherical_shell_radial_bvp_n{n}_final.npz")
    exact = np.load(data_dir / f"spherical_shell_radial_bvp_reference_n{n}.npz")

    coords = np.vstack([num["x_active"], num["y_active"], num["z_active"]]).astype(np.float64)
    radial = coords - CENTER[:, None]
    radius = np.sqrt(np.sum(radial * radial, axis=0))
    radial_unit = radial / np.maximum(radius, 1.0e-30)

    u_num = LENGTH_SCALE_CM * np.sqrt(np.sum(np.asarray(num["u"], dtype=np.float64) ** 2, axis=0))
    u_exact = LENGTH_SCALE_CM * np.sqrt(
        np.sum(np.asarray(exact["u"], dtype=np.float64) ** 2, axis=0)
    )
    lambda_r_num, lambda_theta_num = radial_stretches(
        np.asarray(num["F"], dtype=np.float64), radial_unit
    )
    lambda_r_exact, lambda_theta_exact = radial_stretches(
        np.asarray(exact["F"], dtype=np.float64), radial_unit
    )

    ri = float(summary["geometry"]["inner_radius"])
    ro = float(summary["geometry"]["outer_radius"])
    R_cm, u_num_mean = radial_bin_mean(radius, u_num, ri, ro)
    _R, u_exact_mean = radial_bin_mean(radius, u_exact, ri, ro)
    _R, lambda_r_num_mean = radial_bin_mean(radius, lambda_r_num, ri, ro)
    _R, lambda_r_exact_mean = radial_bin_mean(radius, lambda_r_exact, ri, ro)
    _R, lambda_theta_num_mean = radial_bin_mean(radius, lambda_theta_num, ri, ro)
    _R, lambda_theta_exact_mean = radial_bin_mean(radius, lambda_theta_exact, ri, ro)

    return {
        "R_cm": R_cm,
        "u_num": u_num_mean,
        "u_exact": u_exact_mean,
        "lambda_r_num": lambda_r_num_mean,
        "lambda_r_exact": lambda_r_exact_mean,
        "lambda_theta_num": lambda_theta_num_mean,
        "lambda_theta_exact": lambda_theta_exact_mean,
    }


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, ls=":", lw=0.55, color="0.78", alpha=0.85)
    ax.tick_params(top=True, right=True)
    ax.set_xlabel(r"$R$ [cm]")


def make_figure(data_dir: Path, output_dir: Path, basename: str) -> None:
    configure_style()
    profiles = load_profiles(data_dir)
    R = profiles["R_cm"]
    color_u = "#8a5ccf"
    color_r = "#4057ff"
    color_theta = "#ff8c1a"

    fig = plt.figure(figsize=(7.25, 2.72), constrained_layout=False)
    ax_u = fig.add_axes([0.075, 0.270, 0.390, 0.660])
    ax_lam = fig.add_axes([0.575, 0.270, 0.390, 0.660])

    ax_u.plot(R, profiles["u_exact"], color=color_u, lw=1.75, label="exact")
    ax_u.plot(
        R,
        profiles["u_num"],
        linestyle="None",
        marker="o",
        ms=5.2,
        mfc="white",
        mec=color_u,
        mew=1.15,
        markevery=6,
        label="num",
    )
    ax_u.set_ylabel(r"$|\mathbf{u}|$ [cm]")
    handles, labels = ax_u.get_legend_handles_labels()
    ax_u.legend(
        [handles[1], handles[0]],
        [labels[1], labels[0]],
        loc="upper right",
        frameon=False,
        handlelength=2.2,
    )
    panel_label(ax_u, "(a)")

    (exact_r,) = ax_lam.plot(
        R, profiles["lambda_r_exact"], color=color_r, lw=1.75, label=r"$\lambda_{r,\mathrm{exact}}$"
    )
    (num_r,) = ax_lam.plot(
        R,
        profiles["lambda_r_num"],
        linestyle="None",
        marker="o",
        ms=5.2,
        mfc="white",
        mec=color_r,
        mew=1.15,
        markevery=6,
        label=r"$\lambda_{r,\mathrm{num}}$",
    )
    (exact_theta,) = ax_lam.plot(
        R,
        profiles["lambda_theta_exact"],
        color=color_theta,
        lw=1.75,
        label=r"$\lambda_{\theta,\mathrm{exact}}$",
    )
    (num_theta,) = ax_lam.plot(
        R,
        profiles["lambda_theta_num"],
        linestyle="None",
        marker="s",
        ms=5.0,
        mfc="white",
        mec=color_theta,
        mew=1.15,
        markevery=6,
        label=r"$\lambda_{\theta,\mathrm{num}}$",
    )
    ax_lam.set_ylabel("principal stretch")
    ax_lam.set_ylim(0.50, 1.47)
    ax_lam.legend(
        [num_r, exact_r, num_theta, exact_theta],
        [
            r"$\lambda_{r,\mathrm{num}}$",
            r"$\lambda_{r,\mathrm{exact}}$",
            r"$\lambda_{\theta,\mathrm{num}}$",
            r"$\lambda_{\theta,\mathrm{exact}}$",
        ],
        loc="upper center",
        bbox_to_anchor=(0.50, 0.995),
        ncol=4,
        frameon=False,
        columnspacing=0.62,
        handlelength=1.45,
        handletextpad=0.35,
    )
    panel_label(ax_lam, "(b)")

    for ax in (ax_u, ax_lam):
        style_axes(ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{basename}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--basename", default="fig8_spherical_shell_radial_profiles")
    args = parser.parse_args()
    make_figure(Path(args.data_dir), Path(args.output_dir), str(args.basename))


if __name__ == "__main__":
    main()
