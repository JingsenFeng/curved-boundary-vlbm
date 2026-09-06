#!/usr/bin/env python3
"""Assemble tube profiles at three resolutions and compare with FEM.

FEM values come from the bundled reference table. LBM profiles are
computed from the final fields and axial profiles of each simulation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from _paths import ROOT, RESULTS, REFERENCES, FENICSX_PYTHON, install_reference

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_ROOT = ROOT
TUBE_DATA = RESULTS / "fig10_11_tube" / "data"
OUTPUT_DIR = RESULTS / "fig11_tube_pull_torsion_profiles" / "data"
REFERENCE_PROFILE = REFERENCES / "tube" / "fem_profiles.csv"
PROFILE_NAME = "tube_strain50_resolution_T100_compare_scaled_margin.csv"
SUMMARY_NAME = "tube_strain50_resolution_T100_compare_scaled_margin.json"


def read_named_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if data.dtype.names is None:
        raise ValueError(f"CSV has no named columns: {path}")
    return {name: np.asarray(data[name], dtype=np.float64) for name in data.dtype.names}


def load_summary(case_dir: Path) -> dict:
    return json.loads((case_dir / "tube_pull_torsion_summary.json").read_text(encoding="utf-8"))


def theta_fit_profile(case_dir: Path, z_target: np.ndarray) -> np.ndarray:
    row = load_summary(case_dir)["row"]
    radius_nodes = int(row["radius_nodes"])
    z_nodes = int(row["z_nodes"])
    state_path = case_dir / f"tube_pull_torsion_r{radius_nodes}_z{z_nodes}_final.npz"
    with np.load(state_path) as state:
        x = np.asarray(state["x_active"], dtype=np.float64)
        y = np.asarray(state["y_active"], dtype=np.float64)
        z = np.asarray(state["z_active"], dtype=np.float64)
        u = np.asarray(state["u"], dtype=np.float64)

    load = row["load_resultants"]
    px = x - float(load["center_x"])
    py = y - float(load["center_y"])
    qx = px + u[0]
    qy = py + u[1]
    bins = np.linspace(0.0, float(load["length_z"]), z_target.size + 1)
    theta = np.full(z_target.shape, np.nan, dtype=np.float64)
    for idx, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (z >= lo) & ((z <= hi) if idx == z_target.size - 1 else (z < hi))
        if np.any(mask):
            cross = np.sum(px[mask] * qy[mask] - py[mask] * qx[mask])
            dot = np.sum(px[mask] * qx[mask] + py[mask] * qy[mask])
            theta[idx] = math.atan2(float(cross), float(dot))
    return theta


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[mask] - b[mask])))


def assemble(
    *,
    reference_profile: Path,
    case_dirs: dict[str, Path],
    output_dir: Path,
) -> tuple[Path, Path]:
    reference = read_named_csv(reference_profile)
    z = reference["z"]
    reference_names = (
        "z",
        "kinematic_sin",
        "linear_theta",
        "linear_uz",
        "fem_mean_utheta_over_r",
        "fem_theta_fit",
        "fem_mean_uz",
        "fem_mean_J",
    )
    fields = {name: reference[name] for name in reference_names}
    metrics: dict[str, dict[str, object]] = {}

    for label in ("r10", "r20", "r30"):
        case_dir = case_dirs[label]
        profile = read_named_csv(case_dir / "tube_pull_torsion_axis_profiles.csv")
        fields[f"{label}_mean_utheta_over_r"] = np.interp(
            z, profile["z"], profile["mean_utheta_over_r"]
        )
        fields[f"{label}_theta_fit"] = theta_fit_profile(case_dir, z)
        fields[f"{label}_mean_uz"] = np.interp(z, profile["z"], profile["mean_uz"])
        fields[f"{label}_mean_J"] = np.interp(z, profile["z"], profile["mean_J"])

        summary = load_summary(case_dir)
        row = summary["row"]
        local_method = row["solver_summary"].get("local_method", {})
        metrics[label] = {
            "case_dir": str(case_dir.resolve()),
            "radius_nodes": int(row["radius_nodes"]),
            "z_nodes": int(row["z_nodes"]),
            "dx": float(row["dx"]),
            "steps": int(row["steps"]),
            "time": float(row["time"]),
            "finite": bool(row["finite"]),
            "max_abs_u": float(row["max_abs_u"]),
            "max_abs_v": float(row["max_abs_v"]),
            "min_J": float(row["min_J"]),
            "max_J": float(row["max_J"]),
            "boundary_reconstruction": str(row["boundary_reconstruction"]),
            "compatibility_projection_active": bool(row["compatibility_projection_active"]),
            "local_method": local_method,
            "rmse_mean_utheta_over_r_vs_fem": rmse(
                fields[f"{label}_mean_utheta_over_r"], fields["fem_mean_utheta_over_r"]
            ),
            "rmse_theta_fit_vs_fem": rmse(fields[f"{label}_theta_fit"], fields["fem_theta_fit"]),
            "rmse_mean_uz_vs_fem": rmse(fields[f"{label}_mean_uz"], fields["fem_mean_uz"]),
            "rmse_mean_J_vs_fem": rmse(fields[f"{label}_mean_J"], fields["fem_mean_J"]),
            "maxerr_mean_utheta_over_r_vs_fem": max_abs(
                fields[f"{label}_mean_utheta_over_r"], fields["fem_mean_utheta_over_r"]
            ),
            "maxerr_theta_fit_vs_fem": max_abs(
                fields[f"{label}_theta_fit"], fields["fem_theta_fit"]
            ),
            "maxerr_mean_uz_vs_fem": max_abs(fields[f"{label}_mean_uz"], fields["fem_mean_uz"]),
            "maxerr_mean_J_vs_fem": max_abs(fields[f"{label}_mean_J"], fields["fem_mean_J"]),
        }

    for label in ("r10", "r20"):
        metrics[label].update(
            {
                "rmse_mean_utheta_over_r_vs_r30": rmse(
                    fields[f"{label}_mean_utheta_over_r"], fields["r30_mean_utheta_over_r"]
                ),
                "rmse_theta_fit_vs_r30": rmse(
                    fields[f"{label}_theta_fit"], fields["r30_theta_fit"]
                ),
                "rmse_mean_uz_vs_r30": rmse(fields[f"{label}_mean_uz"], fields["r30_mean_uz"]),
                "rmse_mean_J_vs_r30": rmse(fields[f"{label}_mean_J"], fields["r30_mean_J"]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / PROFILE_NAME
    names = tuple(fields)
    np.savetxt(
        profile_path,
        np.column_stack([fields[name] for name in names]),
        delimiter=",",
        header=",".join(names),
        comments="",
    )
    report = {
        "method": "fully_local_gpu_no_global_compatibility_projection",
        "reference_role": "unchanged FEM reference columns only",
        "reference_profile": str(reference_profile.resolve()),
        "metrics": metrics,
    }
    summary_path = output_dir / SUMMARY_NAME
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return profile_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-profile", default=str(REFERENCE_PROFILE))
    parser.add_argument("--r10-dir", default=str(TUBE_DATA / "r10"))
    parser.add_argument("--r20-dir", default=str(TUBE_DATA / "r20"))
    parser.add_argument("--r30-dir", default=str(TUBE_DATA))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    profile_path, summary_path = assemble(
        reference_profile=Path(args.reference_profile),
        case_dirs={"r10": Path(args.r10_dir), "r20": Path(args.r20_dir), "r30": Path(args.r30_dir)},
        output_dir=Path(args.output_dir),
    )
    print(f"wrote {profile_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
