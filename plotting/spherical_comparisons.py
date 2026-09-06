#!/usr/bin/env python3
"""Spherical-shell field maps and radial profiles against the exact BVP."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from planar_comparisons import (
    LENGTH_SCALE_CM,
    STRESS_SCALE_MPA,
    MARKER_SPACING_PT,
    load_plot_template,
    replace_error_column,
    export_marker_samples,
)


def make_figure(data_dir: Path, output_dir: Path) -> None:
    helper = load_plot_template(
        "fig7_spherical_shell_radial_bvp/plot_fig7_spherical_shell_radial_bvp.py"
    )
    summary = json.loads((data_dir / "spherical_shell_radial_bvp_summary.json").read_text())
    n = max(int(row["n"]) for row in summary["rows"])
    center = np.asarray(summary["geometry"]["center"])
    if not np.allclose(center, helper.CENTER):
        raise ValueError("radial-stress definition requires the configured shell centre")
    inner_radius, outer_radius = (
        float(summary["geometry"][key]) for key in ("inner_radius", "outer_radius")
    )
    numerical_path = data_dir / f"spherical_shell_radial_bvp_n{n}_final.npz"
    reference_path = data_dir / f"spherical_shell_radial_bvp_reference_n{n}.npz"
    with np.load(numerical_path) as num, np.load(reference_path) as ref:
        coords = np.array([num[f"{axis}_active"] for axis in "xyz"])
        for axis, key in enumerate("xyz"):
            if not np.array_equal(coords[axis], ref[f"{key}_active"]):
                raise ValueError("LBM and Exact active coordinates do not match")
        native_axes = [np.unique(coord) for coord in coords]
        fixed_y, fixed_z = (
            axis[np.argmin(np.abs(axis - centre))]
            for axis, centre in zip(native_axes[1:], center[1:])
        )
        plane = np.isclose(coords[2], fixed_z, rtol=0, atol=1e-12)
        cut = plane & np.isclose(coords[1], fixed_y, rtol=0, atol=1e-12) & (coords[0] > center[0])
        selection = np.flatnonzero(cut)
        selection = selection[np.argsort(coords[0, selection])]
        radius_cm = LENGTH_SCALE_CM * np.linalg.norm(coords[:, selection] - center[:, None], axis=0)
        if len(selection) < 2 or not np.all(np.diff(radius_cm) > 0):
            raise ValueError("not enough ordered native radial samples")
        profiles = {}
        for method, data in (("lbm", num), ("exact", ref)):
            u = data["u"]
            sigma = STRESS_SCALE_MPA * data["sigma"]
            scalars = {
                "u_cm": LENGTH_SCALE_CM * np.linalg.norm(u, axis=0),
                "srr_MPa": helper.radial_stress(sigma, *coords),
                "vm_MPa": helper.von_mises_3d(sigma),
            }
            for field, values in scalars.items():
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"non-finite saved {method} {field}")
                profiles[f"{field}_{method}"] = values[selection]
    quantities = [
        ("u_cm", r"$|\mathbf{u}|$ [cm]"),
        ("srr_MPa", r"$\sigma_{rr}$ [MPa]"),
        ("vm_MPa", r"$\sigma_{\mathrm{vm}}$ [MPa]"),
    ]
    basename = "fig7_spherical_shell_radial_bvp"
    comparisons = [
        {
            "x": radius_cm,
            "reference": profiles[f"{field}_exact"],
            "lbm": profiles[f"{field}_lbm"],
            "xlabel": r"$R$ [cm]",
            "ylabel": label,
            "note": r"Radial profile",
            "reference_label": "Exact",
            "source_field": f"{field}_lbm",
            "xlim": (LENGTH_SCALE_CM * inner_radius, LENGTH_SCALE_CM * outer_radius),
        }
        for field, label in quantities
    ]
    helper.make_figure(
        data_dir,
        output_dir,
        basename,
        comparison_column=lambda fig: replace_error_column(
            fig, comparisons, match_column_spacing=True
        ),
    )
    with (output_dir / f"{basename}_profiles.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["R_cm", "X1_cm", "X2_cm", "X3_cm"] + list(profiles))
        for i, source_index in enumerate(selection):
            writer.writerow(
                [radius_cm[i], *(LENGTH_SCALE_CM * coords[:, source_index])]
                + [values[i] for values in profiles.values()]
            )
    metadata = {
        "numerical_npz": str(numerical_path.resolve()),
        "reference_npz": str(reference_path.resolve()),
        "reference": "Exact radial BVP (not FEM)",
        "sampling": "The full profiles CSV contains native active lattice nodes along the positive-X1 ray nearest to the centre, with no radial averaging or smoothing. Displayed markers are separately resampled by piecewise-linear interpolation along the LBM polyline, with equal display-arc spacing per segment; no extrapolation.",
        "markers": "Per-field marker values and adjacent source-row indices (zero-based into *_profiles.csv) are in *_markers.csv.",
        "marker_target_spacing_pt": MARKER_SPACING_PT,
        "X2_cm": LENGTH_SCALE_CM * float(fixed_y),
        "X3_cm": LENGTH_SCALE_CM * float(fixed_z),
        "sample_count": len(selection),
        "marker_counts": {
            profile["source_field"]: len(profile["marker_samples"]["x"]) for profile in comparisons
        },
        "layout": [profile["layout"] for profile in comparisons],
    }
    (output_dir / f"{basename}_profiles.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    export_marker_samples(output_dir, basename, comparisons)
    print(output_dir / f"{basename}.pdf", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make_figure(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
