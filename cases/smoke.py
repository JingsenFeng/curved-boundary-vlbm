#!/usr/bin/env python3
"""Small 2D annulus and 3D sphere calculations with curved traction boundaries."""

import argparse
import json

import numpy as np

from curved_lbm.two_d import curved_boundary_warp as c2
from curved_lbm.three_d import curved_boundary_warp3d as c3
from traction_settings import SETTINGS, RELAXATION_DAMPING


def run(device, steps):
    common = dict(
        device=device, lattice_speed=SETTINGS["lattice_speed"],
        collision_omega=SETTINGS["omega"], boundary_reconstruction="local_f",
        source_mode="damping", damping_gamma=RELAXATION_DAMPING,
        local_displacement_interval=1, local_displacement_sweeps=1,
        local_displacement_relax=SETTINGS["local_displacement_relax"],
        local_displacement_boundary_weight=SETTINGS["local_displacement_boundary_weight"],
        local_compatibility_interval=1,
        local_compatibility_blend=SETTINGS["local_compatibility_blend"],
    )
    geometry2 = c2.CurvedBoundaryGeometry.annulus(
        nx=24, ny=24, inner_radius=0.18, outer_radius=0.42)
    boundary2 = c2.CurvedBoundarySpec(
        kind="neumann", value_mode=c2.CURVED_VALUE_RADIAL_TRACTION, params=(0.01,),
        id_kinds={c2.BOUNDARY_ID_INNER: "dirichlet", c2.BOUNDARY_ID_OUTER: "neumann"},
        id_value_modes={c2.BOUNDARY_ID_INNER: c2.VALUE_ZERO,
                        c2.BOUNDARY_ID_OUTER: c2.CURVED_VALUE_RADIAL_TRACTION},
    )
    geometry3 = c3.CurvedBoundaryGeometry3D.sphere(nx=16, ny=16, nz=16, radius=0.35)
    boundary3 = c3.CurvedBoundarySpec3D(
        kind="neumann", value_mode=c3.CURVED3D_VALUE_NORMAL_TRACTION, params=(0.01,))
    solvers = [
        ("annulus_2d", c2.WarpCurvedBoundaryLBM2D(
            geometry2, material=c2.WarpHyperelasticMaterial.from_poisson("neo_hooke", 0.2, mu=1),
            boundary=boundary2, **common)),
        ("sphere_3d", c3.WarpCurvedBoundaryLBM3D(
            geometry3, material=c3.WarpHyperelasticMaterial3D.from_poisson("neo_hooke", 0.2, mu=1),
            boundary=boundary3, **common)),
    ]
    report = {}
    for name, solver in solvers:
        solver.initialize_identity()
        for _ in range(steps):
            solver.step()
        solver.refresh_current()
        mask = solver.active_mask().astype(bool)
        jacobian = solver.deformation_jacobian()
        # Three-dimensional solvers store active nodes compactly.
        values = jacobian[mask] if jacobian.shape == mask.shape else jacobian
        state = solver.system_state()
        finite = bool(np.isfinite(state).all() and np.isfinite(values).all())
        minimum = float(values.min())
        if not finite or minimum <= 0:
            raise RuntimeError(f"Invalid state in {name}")
        report[name] = {"steps": steps, "time": solver.time, "finite": finite,
                        "min_J": minimum, "max_J": float(values.max())}
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    run(args.device, args.steps)


if __name__ == "__main__":
    main()
