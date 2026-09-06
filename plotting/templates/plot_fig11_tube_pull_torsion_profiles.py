#!/usr/bin/env python3
"""Draw Fig. 11 axial profiles for the 3-D tube pull-torsion benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR
PROFILE_NAME = "tube_strain50_resolution_T100_compare_scaled_margin.csv"
LENGTH_SCALE_CM = 10.0
LBM_CASES = (
    ("r10", r"LBM $26\times26\times100$"),
    ("r20", r"LBM $50\times50\times200$"),
    ("r30", r"LBM $76\times76\times300$"),
)


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


def load_profiles(profile_path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(profile_path, delimiter=",", names=True)
    profiles = {
        "z_cm": LENGTH_SCALE_CM * np.asarray(data["z"], dtype=np.float64),
        "fem_uz_cm": LENGTH_SCALE_CM * np.asarray(data["fem_mean_uz"], dtype=np.float64),
        "fem_utheta_over_r": np.asarray(data["fem_mean_utheta_over_r"], dtype=np.float64),
    }
    for key, _label in LBM_CASES:
        profiles[f"{key}_uz_cm"] = LENGTH_SCALE_CM * np.asarray(
            data[f"{key}_mean_uz"], dtype=np.float64
        )
        profiles[f"{key}_utheta_over_r"] = np.asarray(
            data[f"{key}_mean_utheta_over_r"], dtype=np.float64
        )
    return profiles


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, ls=":", lw=0.55, color="0.78", alpha=0.85)
    ax.tick_params(top=True, right=True)
    ax.set_xlabel(r"$z$ [cm]")


def plot_profiles(
    ax: plt.Axes,
    x: np.ndarray,
    *,
    profiles: dict[str, np.ndarray],
    field: str,
) -> list[plt.Line2D]:
    (fem_line,) = ax.plot(x, profiles[f"fem_{field}"], color="0.12", lw=1.7, label="FEM")
    colors = {"r10": "#8a5ccf", "r20": "#ff8c1a", "r30": "#4057ff"}
    markers = {"r10": "o", "r20": "s", "r30": "^"}
    markevery = {"r10": (0, 6), "r20": (2, 6), "r30": (4, 6)}
    lbm_by_key: dict[str, plt.Line2D] = {}
    for key, label in LBM_CASES:
        (line,) = ax.plot(
            x,
            profiles[f"{key}_{field}"],
            color=colors[key],
            ls="none",
            marker=markers[key],
            markersize=3.5,
            markerfacecolor="white",
            markeredgecolor=colors[key],
            markeredgewidth=0.9,
            markevery=markevery[key],
            label=label,
        )
        lbm_by_key[key] = line
    return [fem_line] + [lbm_by_key[key] for key, _label in LBM_CASES]


def make_figure(profile_path: Path, output_dir: Path, basename: str) -> None:
    configure_style()
    profiles = load_profiles(profile_path)
    z = profiles["z_cm"]

    fig = plt.figure(figsize=(7.25, 2.72), constrained_layout=False)
    ax_uz = fig.add_axes([0.075, 0.270, 0.390, 0.660])
    ax_twist = fig.add_axes([0.575, 0.270, 0.390, 0.660])

    handles = plot_profiles(ax_uz, z, profiles=profiles, field="uz_cm")
    ax_uz.set_ylabel(r"$\langle u_z\rangle$ [cm]")
    ax_uz.set_xlim(0.0, 30.0)
    uz_max = float(
        np.nanmax(
            np.r_[
                profiles["fem_uz_cm"],
                profiles["r10_uz_cm"],
                profiles["r20_uz_cm"],
                profiles["r30_uz_cm"],
            ]
        )
    )
    ax_uz.set_ylim(0.0, 1.05 * uz_max)
    panel_label(ax_uz, "(a)")

    plot_profiles(ax_twist, z, profiles=profiles, field="utheta_over_r")
    ax_twist.set_ylabel(r"$\langle u_\theta/R\rangle$")
    ax_twist.set_xlim(0.0, 30.0)
    twist_max = float(
        np.nanmax(
            np.r_[
                profiles["fem_utheta_over_r"],
                profiles["r10_utheta_over_r"],
                profiles["r20_utheta_over_r"],
                profiles["r30_utheta_over_r"],
            ]
        )
    )
    ax_twist.set_ylim(0.0, 1.06 * twist_max)
    panel_label(ax_twist, "(b)")

    for ax in (ax_uz, ax_twist):
        style_axes(ax)

    ax_uz.legend(
        handles,
        ["FEM"] + [label for _key, label in LBM_CASES],
        loc="upper left",
        bbox_to_anchor=(0.130, 1.012),
        ncol=1,
        frameon=False,
        handlelength=2.4,
        handletextpad=0.55,
        labelspacing=0.18,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{basename}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=str(DATA_DIR / PROFILE_NAME))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--basename", default="fig11_tube_pull_torsion_profiles")
    args = parser.parse_args()
    make_figure(Path(args.profile), Path(args.output_dir), str(args.basename))


if __name__ == "__main__":
    main()
