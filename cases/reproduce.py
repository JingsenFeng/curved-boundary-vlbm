#!/usr/bin/env python3
"""Reproduce the manuscript benchmarks with fixed paper settings."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from _paths import ROOT, RESULTS
from traction_settings import SETTINGS, RELAXATION_DAMPING, configuration, command_arguments

CASES = Path(__file__).resolve().parent
ANNULUS = "fig2_3_annulus_radial/data/annulus_traction_radial_relax_femdyn_n196_ri0p18_ro0p42_bgk1p8_local_T5"
SUPERELLIPSE = "fig4_5_superellipse/data/superellipse_mixed_relax_n384_tx0_ty0p125"
PULSE = "figr14_eccentric_pulse/data/eccentric_pulse_n128_p1p5_omega1p8_T8"
DIRICHLET = {"affine_annulus", "ellipsoid"}


def definitions(root):
    def mixed(script, relative, *extra):
        data = root / relative
        return (
            [
                script,
                "--results-dir",
                str(data.parent),
                "--case-name",
                data.name,
                "--no-plots",
                *extra,
                *command_arguments(),
            ],
            data,
        )

    cases = {
        "annulus": mixed(
            "radial_annulus.py",
            ANNULUS,
            "--ramp-time",
            "2",
            "--kinetic-filter-strength",
            "0",
            "--check-interval",
            "1000",
            "--lbm-progress-every",
            "1000",
        ),
        "superellipse": mixed(
            "superellipse_shell.py", SUPERELLIPSE, "--ramp-time", "2", "--check-interval", "1000"
        ),
        "pulse2d": mixed(
            "eccentric_pulse.py",
            PULSE,
            "--pulse-time",
            "16",
            "--check-interval",
            "1000",
            "--figure-dir",
            str(root / "figr14_eccentric_pulse"),
        ),
        "shell": (
            [
                "spherical_shell.py",
                "--n",
                "200",
                "--steady-rms-v-tol",
                "0",
                "--steady-check-every",
                "1000",
                "--progress-every",
                "1000",
                "--results-dir",
                str(root / "fig7_8_spherical_shell/data"),
                *command_arguments(),
            ],
            root / "fig7_8_spherical_shell/data",
        ),
    }
    for n, margin in ((10, 3), (20, 5), (30, 8)):
        data = root / "fig10_11_tube/data"
        if n != 30:
            data = data / f"r{n}"
        cases[f"tube{n}"] = (
            [
                "tube_pull_torsion.py",
                "--radius-nodes",
                str(n),
                "--z-nodes",
                str(10 * n),
                "--margin-nodes",
                str(margin),
                "--progress-every",
                "1000",
                "--skip-surface-render",
                "--results-dir",
                str(data),
                *command_arguments(),
            ],
            data,
        )
    cases["affine_annulus"] = (
        [
            "affine_annulus.py",
            "--n-values",
            "64",
            "96",
            "128",
            "192",
            "256",
            "384",
            "--omega",
            "2",
            "--boundary-reconstruction",
            "compat_bfl",
            "--init-order",
            "2",
            "--no-plots",
            "--out-dir",
            str(root / "fig1_dirichlet_annulus/data"),
        ],
        root / "fig1_dirichlet_annulus/data",
    )
    cases["ellipsoid"] = (
        [
            "affine_ellipsoid.py",
            "--n",
            "80,120,160,240,320",
            "--omega",
            "1.8",
            "--init-order",
            "1",
            "--target-max-u",
            "0.1",
            "--boundary-reconstruction",
            "local_f",
            "--local-displacement-interval",
            "0",
            "--local-compatibility-interval",
            "0",
            "--results-dir",
            str(root / "fig6_ellipsoid_dirichlet_3d/data"),
        ],
        root / "fig6_ellipsoid_dirichlet_3d/data",
    )
    order = [
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
    return {name: cases[name] for name in order}


def verify_summary(data):
    files = list(data.glob("*summary*.json"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one case summary in {data}, found {len(files)}")
    summary = json.loads(files[0].read_text())
    rows = summary.get("rows", [summary.get("row", summary)])
    if not rows or not all(row.get("finite", False) for row in rows):
        raise RuntimeError(f"Non-finite result: {files[0]}")
    for row in rows:
        diagnostic = row.get("final", row)
        if diagnostic.get("min_J", 0) <= 0:
            raise RuntimeError(f"Non-positive deformation Jacobian: {files[0]}")
        target = summary.get(
            "run_time", summary.get("run_time_requested", row.get("max_run_time", 100))
        )
        if abs(float(row["time"]) - float(target)) > 0.01:
            raise RuntimeError(f"Incomplete time interval: {files[0]}")
        local = summary.get(
            "local_method",
            summary.get(
                "local_compatibility", row.get("solver_summary", {}).get("local_method", {})
            ),
        )
        required = {
            "displacement_interval": 1,
            "displacement_sweeps": 1,
            "displacement_relax": SETTINGS["local_displacement_relax"],
            "boundary_weight": SETTINGS["local_displacement_boundary_weight"],
            "F_repair_interval": 1,
            "F_repair_blend": SETTINGS["local_compatibility_blend"],
            "kernel_stencil_radius": 2,
            "composed_compatibility_radius": 3,
        }
        for key, value in required.items():
            if local.get(key) != value:
                raise RuntimeError(f"{files[0]}: {key}={local.get(key)}; required {value}")
        for key, value in [
            ("collision_omega", SETTINGS["omega"]),
            ("lattice_speed", SETTINGS["lattice_speed"]),
        ]:
            if row.get(key, summary.get(key)) != value:
                raise RuntimeError(f"{files[0]}: mismatched {key}")
        gamma = 0.0 if "pulse" in str(summary.get("case", "")) else RELAXATION_DAMPING
        if row.get("damping_gamma", summary.get("damping_gamma")) != gamma:
            raise RuntimeError(f"{files[0]}: mismatched damping")
    return files[0]


def verify_result(name, data):
    if name not in DIRICHLET:
        return verify_summary(data)
    paths = list(data.glob("*summary*.json"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one summary in {data}")
    summary = json.loads(paths[0].read_text())
    rows = summary.get("rows", [])
    expected_grids = (
        [64, 96, 128, 192, 256, 384] if name == "affine_annulus" else [80, 120, 160, 240, 320]
    )
    if [r["n"] for r in rows] != expected_grids:
        raise RuntimeError(f"Incomplete grid sequence: {paths[0]}")
    for row in rows:
        if not row["finite"] or row["min_J"] <= 0:
            raise RuntimeError(f"Invalid deformation state: {paths[0]}")
        target = summary.get("T_end", summary.get("run_time_requested"))
        if abs(row["time"] - target) > 0.01:
            raise RuntimeError(f"Incomplete simulation: {paths[0]}")
    return paths[0]


def source_fingerprint():
    files = sorted((ROOT / "src").rglob("*.py")) + sorted(CASES.glob("*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=RESULTS, help="data, plots, and execution records"
    )
    parser.add_argument(
        "--cases", default="all", help="all, traction, dirichlet, or comma-separated case names"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:0")
    parser.add_argument("--list", action="store_true", help="list available cases without running")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the exact commands without running"
    )
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="recompute existing selected cases")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    cases = definitions(root)
    if args.list:
        print("\n".join(cases))
        return
    groups = {
        "all": list(cases),
        "traction": [k for k in cases if k not in DIRICHLET],
        "dirichlet": [k for k in cases if k in DIRICHLET],
    }
    selected = groups.get(args.cases, [k.strip() for k in args.cases.split(",")])
    if not selected or any(k not in cases for k in selected):
        parser.error("Unknown case; use --list to see the available names")
    device = args.device
    if device == "auto":
        import warp as wp

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
    commands = {
        name: [
            sys.executable,
            str(CASES / cases[name][0][0]),
            *cases[name][0][1:],
            "--device",
            device,
        ]
        for name in selected
    }
    if args.dry_run:
        for name, command in commands.items():
            print(f"{name}: {shlex.join(command)}")
        return
    records = root / "validation"
    records.mkdir(parents=True, exist_ok=True)
    fingerprints = source_fingerprint()
    environment = {**os.environ, "XLB_NEUMANN_SOLVER": "newton", "WARP_DEVICE": device}
    if not args.render_only:
        for name, command in commands.items():
            data = cases[name][1]
            manifest = records / f"{name}.json"
            if not args.rerun and list(data.glob("*summary*.json")):
                verify_result(name, data)
                previous = json.loads(manifest.read_text()) if manifest.exists() else {}
                if (
                    previous.get("command") != command
                    or previous.get("source_hashes") != fingerprints
                ):
                    raise RuntimeError(
                        f"{name}: execution record differs; use --rerun to recompute"
                    )
                print(f"{name}: verified completed result", flush=True)
                continue
            record = {
                "case": name,
                "command": command,
                "source_hashes": fingerprints,
                "settings": configuration()
                if name not in DIRICHLET
                else "affine Dirichlet defaults",
                "neumann_solver": "newton",
                "status": "running",
            }
            manifest.write_text(json.dumps(record, indent=2) + "\n")
            started = time.monotonic()
            print(f"{name}: running (log: {records / (name + '.log')})", flush=True)
            try:
                with (records / f"{name}.log").open("w") as log:
                    subprocess.run(
                        command,
                        cwd=ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                record["summary"] = str(verify_result(name, data).relative_to(root))
                record["status"] = "complete"
            except Exception:
                record["status"] = "failed"
                raise
            finally:
                record["elapsed_seconds"] = time.monotonic() - started
                manifest.write_text(json.dumps(record, indent=2) + "\n")
    if not args.no_render:
        for name in selected:
            verify_result(name, cases[name][1])
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "plotting/render.py"),
                "--root",
                str(root),
                "--cases",
                ",".join(selected),
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    main()
