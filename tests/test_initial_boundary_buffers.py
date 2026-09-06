"""Initialization must evaluate every cut link from the same population state."""

import numpy as np
import pytest
import warp as wp

from curved_lbm.two_d import curved_boundary_warp as c2
from curved_lbm.three_d import curved_boundary_warp3d as c3


@pytest.mark.parametrize("storage", ["2d", "3d_sparse", "3d_dense"])
def test_in_place_boundary_matches_frozen_source(storage):
    # Two active nodes across the small body make opposite cut links read
    # populations that other links update. This exposes aliasing deterministically.
    if storage == "2d":
        geometry = c2.CurvedBoundaryGeometry.circle(nx=8, ny=8, radius=0.12)
        material = c2.WarpHyperelasticMaterial.from_poisson("neo_hooke", 0.2, mu=1)
        boundary = c2.CurvedBoundarySpec(
            kind="dirichlet", value_mode=c2.VALUE_CONSTANT, params=(0.001, -0.002)
        )
        solver = c2.WarpCurvedBoundaryLBM2D(
            geometry,
            material=material,
            boundary=boundary,
            boundary_reconstruction="bfl",
            lattice_speed=10,
            collision_omega=1.8,
        )
    else:
        geometry = c3.CurvedBoundaryGeometry3D.sphere(nx=8, ny=8, nz=8, radius=0.12)
        material = c3.WarpHyperelasticMaterial3D.from_poisson("neo_hooke", 0.2, mu=1)
        boundary = c3.CurvedBoundarySpec3D(
            kind="dirichlet",
            value_mode=c3.CURVED3D_VALUE_CONSTANT_VECTOR,
            params=(0.001, -0.002, 0.003),
        )
        cls = (
            c3.WarpSparseCurvedBoundaryLBM3D
            if storage == "3d_sparse"
            else c3.WarpDenseCurvedBoundaryLBM3D
        )
        solver = cls(
            geometry,
            material=material,
            boundary=boundary,
            boundary_reconstruction="bfl",
            lattice_speed=10,
            collision_omega=1.8,
        )
    before = solver.f0.numpy() + np.random.default_rng(501).normal(0, 0.001, solver.f0.shape)
    frozen = wp.array(before, dtype=wp.float64, device=solver.device)
    wp.copy(solver.f0, frozen)
    if hasattr(solver, "fpost"):
        wp.copy(solver.fpost, frozen)
    wp.copy(
        solver.f1, wp.full(solver.f1.shape, float("nan"), dtype=wp.float64, device=solver.device)
    )
    solver.apply_boundary_reconstruction(0.03, in_place=False)
    separate = solver.f1.numpy()
    written = np.isfinite(separate)
    assert written.any()
    expected = before.copy()
    expected[written] = separate[written]
    wp.copy(solver.f0, frozen)
    solver.apply_boundary_reconstruction(0.03, in_place=True)
    np.testing.assert_array_equal(solver.f0.numpy(), expected)
