#!/usr/bin/env python3
"""Convert sampled tube FEM fields into the profile table used by tube_profiles.py."""

import argparse
from pathlib import Path

import numpy as np


def export(source, destination):
    with np.load(source) as data:
        points = data["points"]
        displacement = data["u"]
        center = data["center_xy"]
        z = data["profile_z"]
        length = float(data["length_z"])
        angle = float(data["target_twist_angle"])
        extension = float(data["axial_displacement"])
        px, py = points[:, 0] - center[0], points[:, 1] - center[1]
        qx, qy = px + displacement[:, 0], py + displacement[:, 1]
        edges = np.linspace(0, length, z.size + 1)
        fitted = np.empty_like(z)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            upper = points[:, 2] <= hi if i == z.size - 1 else points[:, 2] < hi
            selected = (points[:, 2] >= lo) & upper
            if not selected.any():
                raise ValueError(f"No FEM samples in axial bin {i}")
            cross = np.sum(px[selected] * qy[selected] - py[selected] * qx[selected])
            dot = np.sum(px[selected] * qx[selected] + py[selected] * qy[selected])
            fitted[i] = np.arctan2(cross, dot)
        columns = {
            "z": z,
            "kinematic_sin": np.sin(angle * z / length),
            "linear_theta": angle * z / length,
            "linear_uz": extension * z / length,
            "fem_mean_utheta_over_r": data["profile_utheta_over_r"],
            "fem_theta_fit": fitted,
            "fem_mean_uz": data["profile_uz"],
            "fem_mean_J": data["profile_J"],
        }
        values = np.column_stack(list(columns.values()))
    if not np.isfinite(values).all():
        raise ValueError("Non-finite FEM profile")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        destination, values, delimiter=",", header=",".join(columns), comments="", fmt="%.17g"
    )
    print(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.input, args.output)


if __name__ == "__main__":
    main()
