#!/usr/bin/env python3
"""Planar FEM/LBM field maps and matched line profiles.

Profiles use the same scalar interpolation on the two grid rows
bracketing the geometric centre. Both rows must be active; profiles
do not interpolate through holes or outside the body.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


PAPER_ROOT = Path(__file__).resolve().parents[1]
NEWFIG_ROOT = PAPER_ROOT / "results" / "paper"
LENGTH_SCALE_CM = 10.0
STRESS_SCALE_MPA = 10.0
LBM_COLOR = "#1f77b4"
MARKER_SPACING_PT = 10.0
FIGURE_NAMES = {
    2: "fig2_annulus_radial_comparison",
    3: "fig3_annulus_radial_stress",
    4: "fig4_superellipse_mixed_comparison",
    5: "fig5_superellipse_mixed_stress",
}


def load_plot_template(relative_path: str):
    path = Path(__file__).resolve().parent / "templates" / Path(relative_path).name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.65,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def contiguous_segments(valid: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(valid)
    if not indices.size:
        return []
    return list(np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1))


def equal_arc_markers(x, lbm, valid, *, points_per_data_unit, spacing_pt=MARKER_SPACING_PT) -> dict:
    """Equally spaced markers along each LBM polyline in display-point units.

    Interpolate only along adjacent saved LBM samples in one material segment.
    The FEM values do not enter this operation. Both segment ends are retained;
    the spacing is constant within a segment and close to spacing_pt across
    segments. No rounding to native indices, smoothing, or gap bridging occurs.
    """
    x, lbm = np.asarray(x, dtype=float), np.asarray(lbm, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(x) & np.isfinite(lbm)
    scale = np.asarray(points_per_data_unit, dtype=float)
    if x.ndim != 1 or lbm.shape != x.shape or valid.shape != x.shape:
        raise ValueError("marker inputs must have matching one-dimensional shapes")
    if (
        scale.shape != (2,)
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0)
        or spacing_pt <= 0
    ):
        raise ValueError("display scales and marker spacing must be positive")
    result = {
        key: []
        for key in ("x", "lbm", "segment", "source_left", "source_right", "upper_weight", "arc_pt")
    }
    for segment_id, segment in enumerate(contiguous_segments(valid)):
        if segment.size > 1 and not np.all(np.diff(x[segment]) > 0):
            raise ValueError("profile coordinates must increase within each material segment")
        xy = np.column_stack((x[segment], lbm[segment]))
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0) * scale, axis=1))]
        if segment.size == 1:
            target_arc = np.array([0.0])
            left = right = segment.copy()
            weight = np.array([0.0])
        else:
            intervals = max(1, round(arc[-1] / spacing_pt))
            target_arc = np.linspace(0, arc[-1], intervals + 1)
            bracket = np.clip(np.searchsorted(arc, target_arc, side="right") - 1, 0, len(arc) - 2)
            left, right = segment[bracket], segment[bracket + 1]
            weight = (target_arc - arc[bracket]) / (arc[bracket + 1] - arc[bracket])
        values = {
            "x": (1 - weight) * x[left] + weight * x[right],
            "lbm": (1 - weight) * lbm[left] + weight * lbm[right],
            "segment": np.full(len(left), segment_id),
            "source_left": left,
            "source_right": right,
            "upper_weight": weight,
            "arc_pt": target_arc,
        }
        for key, array in values.items():
            result[key].extend(array.tolist())
    integer_keys = ("segment", "source_left", "source_right")
    return {
        key: np.asarray(values, dtype=int if key in integer_keys else float)
        for key, values in result.items()
    }


def sample_center_cut(
    X: np.ndarray,
    Y: np.ndarray,
    mask: np.ndarray,
    fields: dict[str, np.ndarray],
    *,
    varying_axis: int,
    fixed_coordinate: float,
) -> dict:
    """Piecewise-linear scalar samples on an exact, axis-aligned centre cut."""
    if not (np.allclose(X, X[:, :1]) and np.allclose(Y, Y[:1, :])):
        raise ValueError("expected an ij-indexed Cartesian grid")
    fixed_grid = Y[0, :] if varying_axis == 0 else X[:, 0]
    coordinate = X[:, 0] if varying_axis == 0 else Y[0, :]
    if not np.all(np.diff(fixed_grid) > 0):
        raise ValueError("grid coordinates must be increasing")
    hi = int(np.searchsorted(fixed_grid, fixed_coordinate))
    if hi < fixed_grid.size and np.isclose(fixed_grid[hi], fixed_coordinate, rtol=0, atol=1e-12):
        lo = hi
        alpha = 0.0
    else:
        if hi == 0 or hi == fixed_grid.size:
            raise ValueError("cut lies outside the grid")
        lo = hi - 1
        alpha = float((fixed_coordinate - fixed_grid[lo]) / (fixed_grid[hi] - fixed_grid[lo]))

    def take(array: np.ndarray, index: int) -> np.ndarray:
        return np.take(array, index, axis=1 - varying_axis)

    valid = take(mask, lo) & take(mask, hi)
    sampled = {}
    for name, values in fields.items():
        if not np.all(np.isfinite(values[mask])):
            raise ValueError(f"non-finite active values in {name}")
        result = (1 - alpha) * take(values, lo) + alpha * take(values, hi)
        sampled[name] = np.where(valid, result, np.nan)
    return {
        "coordinate_cm": LENGTH_SCALE_CM * coordinate,
        "fixed_coordinate_cm": LENGTH_SCALE_CM * fixed_coordinate,
        "varying_axis": varying_axis + 1,
        "valid": valid,
        "fields": sampled,
        "bracketing_indices": [lo, hi],
        "upper_weight": alpha,
    }


def method_handles(reference: str = "FEM") -> list[Line2D]:
    return [
        Line2D([], [], color="black", lw=1.1, label=reference),
        Line2D(
            [],
            [],
            color=LBM_COLOR,
            ls="none",
            marker="o",
            markersize=3.6,
            markerfacecolor="none",
            markeredgewidth=0.85,
            label="LBM",
        ),
    ]


def draw_profile(ax, x, reference, lbm, *, valid=None, reference_label="FEM"):
    if valid is None:
        valid = np.isfinite(reference) & np.isfinite(lbm)
    ax.plot(x, np.where(valid, reference, np.nan), color="black", lw=1.1, label=reference_label)
    # Native samples establish autoscaling. Replace these by equal-arc samples
    # after the panel limits, size and layout have been finalized.
    (markers,) = ax.plot(
        x[valid],
        lbm[valid],
        ls="none",
        marker="o",
        markersize=3.6,
        markerfacecolor="none",
        markeredgecolor=LBM_COLOR,
        markeredgewidth=0.85,
        label="LBM",
        zorder=3,
    )
    ax.grid(True, color="0.88", lw=0.45)
    ax.tick_params(top=True, right=True, labelleft=True, labelbottom=True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    return markers


def panel_label(ax, letter: str) -> None:
    ax.text(
        0.025,
        0.975,
        letter,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
    )


def replace_error_column(fig, profiles: list[dict], *, match_column_spacing: bool = False):
    """Keep every first/second-column artist and its position exactly as built.

    The legacy plot builds its normal figure. Only its third-column axes and
    their colorbars are replaced here. The original crop is retained too.
    """
    save_dpi = plt.rcParams["savefig.dpi"]
    if save_dpi != "figure":
        fig.set_dpi(save_dpi)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    original_bbox = fig.get_tightbbox(renderer).padded(plt.rcParams["savefig.pad_inches"])
    maps = [
        (ax, collection.colorbar)
        for ax in fig.axes
        for collection in ax.collections
        if collection.colorbar is not None
    ]
    if len(maps) != 3 * len(profiles):
        raise ValueError("expected three original map columns")
    unchanged = [
        (kept, kept.get_position().frozen())
        for i, (ax, colorbar) in enumerate(maps)
        if i % 3 != 2
        for kept in (ax, colorbar.ax)
    ]
    right_edge = original_bbox.x1 / fig.get_figwidth() - 0.020
    marker_artists = []
    for row, ((ax, colorbar), profile) in enumerate(zip(maps[2::3], profiles)):
        original_position = ax.get_position().frozen()
        colorbar.ax.remove()
        ax.clear()
        ax.set_axes_locator(None)
        ax.set_aspect("auto")
        first, second = (maps[3 * row + column][0].get_position() for column in (0, 1))
        gap12 = second.x0 - first.x1
        left = second.x1 + gap12 if match_column_spacing else original_position.x0 + 0.060
        width = original_position.width if match_column_spacing else right_edge - left
        ax.set_position([left, original_position.y0, width, original_position.height])
        x, reference, numerical = (profile[key] for key in ("x", "reference", "lbm"))
        valid = np.isfinite(reference) & np.isfinite(numerical)
        marker_artists.append(
            draw_profile(
                ax,
                x,
                reference,
                numerical,
                valid=valid,
                reference_label=profile.get("reference_label", "FEM"),
            )
        )
        ax.tick_params(labelsize=7.0)
        ax.set_xlabel(profile["xlabel"], fontsize=8, labelpad=1)
        ax.set_ylabel(profile["ylabel"], fontsize=8, labelpad=1)
        if match_column_spacing:
            # Reuse the old error-colorbar side for the vertical scale, leaving
            # the inter-column gutter as clear as the first/second-column one.
            ax.yaxis.set_label_position("right")
            ax.tick_params(labelleft=False, labelright=True)
        if "xlim" in profile:
            ax.set_xlim(profile["xlim"])
        else:
            span = np.ptp(x[valid])
            ax.set_xlim(x[valid].min() - 0.03 * span, x[valid].max() + 0.03 * span)
        all_values = np.r_[reference[valid], numerical[valid]]
        low, high = float(all_values.min()), float(all_values.max())
        padding = max(high - low, 0.05 * abs(high), 1e-8) * 0.18
        ax.set_ylim(low - padding, high + padding)
        ax.text(
            0.53,
            0.975,
            profile["note"],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.5,
        )
        ax.legend(
            loc="center" if len(contiguous_segments(valid)) > 1 else "best",
            frameon=False,
            fontsize=7,
            handlelength=1.5,
            labelspacing=0.25,
            borderaxespad=0.4,
        )
        panel_label(ax, f"({chr(99 + 3 * row)})")
    fig.canvas.draw()
    # Keep all right-side labels inside the original canvas without shifting
    # either of the original map columns or the new third-column left edge.
    if match_column_spacing:
        for ax, _ in maps[2::3]:
            overflow = ax.get_tightbbox(fig.canvas.get_renderer()).x1 - right_edge * fig.bbox.width
            if overflow > 0:
                position = ax.get_position()
                ax.set_position(
                    [
                        position.x0,
                        position.y0,
                        position.width - overflow / fig.bbox.width,
                        position.height,
                    ]
                )
        fig.canvas.draw()
    for row, ((ax, _), profile, artist) in enumerate(zip(maps[2::3], profiles, marker_artists)):
        valid = np.isfinite(profile["reference"]) & np.isfinite(profile["lbm"])
        scale = (
            np.array(
                [ax.bbox.width / np.ptp(ax.get_xlim()), ax.bbox.height / np.ptp(ax.get_ylim())]
            )
            * 72
            / fig.dpi
        )
        samples = equal_arc_markers(profile["x"], profile["lbm"], valid, points_per_data_unit=scale)
        artist.set_data(samples["x"], samples["lbm"])
        profile["marker_samples"] = samples
        first, second = (maps[3 * row + column][0].get_position() for column in (0, 1))
        profile["layout"] = {
            "gap12_in": (second.x0 - first.x1) * fig.get_figwidth(),
            "gap23_in": (ax.get_position().x0 - second.x1) * fig.get_figwidth(),
            "matched": match_column_spacing,
        }
        if match_column_spacing:
            np.testing.assert_allclose(
                profile["layout"]["gap12_in"], profile["layout"]["gap23_in"], atol=1e-12
            )
    for ax, position in unchanged:
        np.testing.assert_allclose(ax.get_position().bounds, position.bounds, rtol=0, atol=1e-12)
    return original_bbox


def export_marker_samples(output_dir: Path, basename: str, profiles: list[dict]) -> None:
    with (output_dir / f"{basename}_markers.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "panel",
                "source_field",
                "segment",
                "coordinate_cm",
                "lbm",
                "source_left",
                "source_right",
                "upper_weight",
                "arc_pt",
            ]
        )
        for row, profile in enumerate(profiles):
            samples = profile["marker_samples"]
            for i in range(len(samples["x"])):
                writer.writerow(
                    [
                        chr(99 + 3 * row),
                        profile["source_field"],
                        *(
                            samples[key][i]
                            for key in (
                                "segment",
                                "x",
                                "lbm",
                                "source_left",
                                "source_right",
                                "upper_weight",
                                "arc_pt",
                            )
                        ),
                    ]
                )


def export_cuts(
    output_dir: Path, basename: str, cuts: list[dict], source: Path, profiles: list[dict]
) -> None:
    field_names = list(cuts[0]["fields"])
    with (output_dir / f"{basename}_profiles.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["cut", "varying_axis", "coordinate_cm", "fixed_coordinate_cm", "in_solid"]
            + field_names
        )
        for cut_name, cut in zip(("A-Aprime", "B-Bprime"), cuts):
            for i, position in enumerate(cut["coordinate_cm"]):
                writer.writerow(
                    [
                        cut_name,
                        cut["varying_axis"],
                        position,
                        cut["fixed_coordinate_cm"],
                        int(cut["valid"][i]),
                    ]
                    + [cut["fields"][name][i] for name in field_names]
                )
    metadata = {
        "source_npz": str(source.resolve()),
        "reference": "FEM",
        "sampling": "Linear interpolation of each scalar between the two centre-bracketing grid rows; both rows must be active. No extrapolation or smoothing.",
        "markers": "Equal display-arc spacing on each saved LBM polyline, retaining segment endpoints. Piecewise-linear interpolation uses adjacent LBM samples only; no smoothing or extrapolation. Per-field marker positions and interpolation brackets are in *_markers.csv (zero-based indices into *_profiles.csv).",
        "marker_target_spacing_pt": MARKER_SPACING_PT,
        "layout": [profile["layout"] for profile in profiles],
        "cuts": [
            {
                key: value
                for key, value in cut.items()
                if key
                in ("varying_axis", "fixed_coordinate_cm", "bracketing_indices", "upper_weight")
            }
            for cut in cuts
        ],
    }
    (output_dir / f"{basename}_profiles.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    export_marker_samples(output_dir, basename, profiles)


def make_figure(number: int, data_dir: Path, output_dir: Path) -> None:
    # Reuse exactly the stress definition, units, geometry and map interpolation
    # of the existing manuscript plot; do not introduce a new material model.
    helper = load_plot_template(
        "fig5_superellipse_mixed_stress/plot_fig5_superellipse_mixed_stress.py"
    )
    filename = (
        "annulus_traction_comparison_n196.npz"
        if number in (2, 3)
        else "superellipse_mixed_comparison_n384.npz"
    )
    source = data_dir / filename
    with np.load(source) as data:
        X, Y, mask = data["x"], data["y"], data["mask"].astype(bool)
        outer, inner = (helper.Superellipse(*data[key]) for key in ("outer", "inner"))
        fields = {}
        for method in ("fem", "lbm"):
            F = data[f"F_{method}"]
            if number in (2, 4):
                fields[f"u_{method}_cm"] = LENGTH_SCALE_CM * np.linalg.norm(
                    data[f"u_{method}"], axis=0
                )
                fields[f"Finc_{method}"] = np.sqrt(
                    np.sum((F - np.eye(2)[:, :, None, None]) ** 2, axis=(0, 1))
                )
            else:
                stress = STRESS_SCALE_MPA * helper.cauchy_stress(F, *data["material"])
                fields[f"vm_{method}_MPa"] = helper.von_mises_2d(stress)
                fields[f"s22_{method}_MPa"] = stress[1, 1]
    quantities = (
        [("u_{}_cm", r"$|\mathbf{u}|$ [cm]"), ("Finc_{}", r"$\|\mathbf{F}-\mathbf{I}\|_F$")]
        if number in (2, 4)
        else [
            ("vm_{}_MPa", r"$\sigma_{\mathrm{vm}}$ [MPa]"),
            ("s22_{}_MPa", r"$\sigma_{22}$ [MPa]"),
        ]
    )
    cut = sample_center_cut(X, Y, mask, fields, varying_axis=0, fixed_coordinate=outer.cy)
    profiles = [
        {
            "x": cut["coordinate_cm"],
            "reference": cut["fields"][template.format("fem")],
            "lbm": cut["fields"][template.format("lbm")],
            "xlabel": r"$X_1$ [cm]",
            "source_field": template.format("lbm"),
            "ylabel": label,
            "note": rf"$X_2={cut['fixed_coordinate_cm']:g}$ cm",
        }
        for template, label in quantities
    ]
    basename = FIGURE_NAMES[number]
    legacy = load_plot_template(f"{basename}/plot_{basename}.py")
    legacy.make_figure(
        data_dir,
        output_dir,
        basename,
        comparison_column=lambda fig: replace_error_column(
            fig, profiles, match_column_spacing=number in (2, 3)
        ),
    )
    export_cuts(output_dir, basename, [cut], source, profiles)
    print(output_dir / f"{basename}.pdf", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=int, required=True, choices=tuple(FIGURE_NAMES))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make_figure(args.figure, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
