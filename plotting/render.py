#!/usr/bin/env python3
"""Render manuscript figures from completed case outputs."""

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"
ALL_CASES = [
    "affine_annulus",
    "annulus",
    "superellipse",
    "pulse2d",
    "ellipsoid",
    "shell",
    "tube10",
    "tube20",
    "tube30",
]


def run(script, *args):
    subprocess.run([sys.executable, str(script), *map(str, args)], cwd=ROOT, check=True)


def render(root, selected):
    for case in selected:
        print(f"Rendering {case}", flush=True)
        if case == "affine_annulus":
            run(
                TEMPLATES / "plot_fig1_dirichlet_annulus.py",
                "--data-dir",
                root / "fig1_dirichlet_annulus/data",
                "--output-dir",
                root / "fig1_dirichlet_annulus",
            )
        elif case in ("annulus", "superellipse"):
            if case == "annulus":
                data = (
                    root
                    / "fig2_3_annulus_radial/data/annulus_traction_radial_relax_femdyn_n196_ri0p18_ro0p42_bgk1p8_local_T5"
                )
                figures = {2: "fig2_annulus_radial_comparison", 3: "fig3_annulus_radial_stress"}
            else:
                data = root / "fig4_5_superellipse/data/superellipse_mixed_relax_n384_tx0_ty0p125"
                figures = {
                    4: "fig4_superellipse_mixed_comparison",
                    5: "fig5_superellipse_mixed_stress",
                }
            for number, name in figures.items():
                run(
                    HERE / "planar_comparisons.py",
                    "--figure",
                    number,
                    "--data-dir",
                    data,
                    "--output-dir",
                    root / name,
                )
        elif case == "pulse2d":
            run(
                ROOT / "cases/eccentric_pulse.py",
                "--plot-only",
                "--results-dir",
                root / "figr14_eccentric_pulse/data",
                "--case-name",
                "eccentric_pulse_n128_p1p5_omega1p8_T8",
                "--figure-dir",
                root / "figr14_eccentric_pulse",
            )
        elif case == "ellipsoid":
            run(
                TEMPLATES / "plot_fig6_ellipsoid_dirichlet.py",
                "--data-dir",
                root / "fig6_ellipsoid_dirichlet_3d/data",
                "--output-dir",
                root / "fig6_ellipsoid_dirichlet_3d",
            )
        elif case == "shell":
            data = root / "fig7_8_spherical_shell/data"
            run(
                HERE / "spherical_comparisons.py",
                "--data-dir",
                data,
                "--output-dir",
                root / "fig7_spherical_shell_radial_bvp",
            )
            run(
                TEMPLATES / "plot_fig8_spherical_shell_radial_profiles.py",
                "--data-dir",
                data,
                "--output-dir",
                root / "fig8_spherical_shell_radial_profiles",
            )
        elif case == "tube30":
            run(
                TEMPLATES / "plot_fig10_tube_pull_torsion.py",
                "--data-dir",
                root / "fig10_11_tube/data",
                "--output-dir",
                root / "fig10_tube_pull_torsion",
            )
    if any(k.startswith("tube") for k in selected):
        data = root / "fig10_11_tube/data"
        if all(
            (d / "tube_pull_torsion_summary.json").exists()
            for d in (data / "r10", data / "r20", data)
        ):
            out = root / "fig11_tube_pull_torsion_profiles"
            run(
                ROOT / "cases/tube_profiles.py",
                "--r10-dir",
                data / "r10",
                "--r20-dir",
                data / "r20",
                "--r30-dir",
                data,
                "--output-dir",
                out / "data",
            )
            run(
                TEMPLATES / "plot_fig11_tube_pull_torsion_profiles.py",
                "--profile",
                out / "data/tube_strain50_resolution_T100_compare_scaled_margin.csv",
                "--output-dir",
                out,
            )
        else:
            print(
                "Tube resolution comparison requires completed tube10, tube20, and tube30 outputs."
            )
        spec = importlib.util.spec_from_file_location(
            "tube_schematic", TEMPLATES / "plot_fig9_tube_pull_torsion_schematic.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.SCRIPT_DIR = root / "fig9_tube_pull_torsion_schematic"
        module.SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        module.make_figure()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/paper")
    parser.add_argument("--cases", default="all")
    args = parser.parse_args()
    selected = ALL_CASES if args.cases == "all" else args.cases.split(",")
    if any(k not in ALL_CASES for k in selected):
        parser.error("Unknown case name")
    render(args.root.expanduser().resolve(), selected)


if __name__ == "__main__":
    main()
