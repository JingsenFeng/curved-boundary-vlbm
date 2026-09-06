#!/usr/bin/env python3
"""Independent geometry, constitutive-boundary and population checks."""

from pathlib import Path
import json
import sys
import unittest
import numpy as np

from curved_lbm.three_d import curved_boundary_warp3d as cb
from curved_lbm.three_d.characteristic_stencil3d import build_characteristic_weights


def sphere(n=24):
    domain = cb.SphereDomain(radius=0.35)
    return cb.CurvedBoundaryGeometry3D.from_level_set(
        nx=n,
        ny=n,
        nz=n,
        length_x=1.0,
        length_y=1.0,
        length_z=1.0,
        phi=domain.phi,
        normal=domain.normal,
    )


def flux(U, material, axis):
    F = U[3:].reshape(3, 3)
    P = cb._first_piola_np(F, material)
    out = np.zeros(12)
    out[:3] = -P[:, axis]
    out[3:].reshape(3, 3)[:, axis] = -U[:3]
    return out


def derivative(U, W, material, axis):
    eps = 2e-6 / max(1.0, np.max(np.abs(W)))
    return (flux(U + eps * W, material, axis) - flux(U - eps * W, material, axis)) / (2 * eps)


class CharacteristicChecks(unittest.TestCase):
    def test_dirichlet_bfl_regression(self):
        reference = np.load(Path(__file__).parent / "data" / "dirichlet_bfl_3d_reference.npz")
        g = cb.CurvedBoundaryGeometry3D.sphere(nx=20, ny=20, nz=20, radius=0.35)
        material = cb.WarpHyperelasticMaterial3D.from_poisson("neo_hooke", 0.2, mu=1.0)
        kinds = ("dirichlet",) * 3
        boundary = cb.CurvedBoundarySpec3D(
            kind="neumann",
            component_kinds=kinds,
            value_mode=cb.CURVED3D_VALUE_CONSTANT_VECTOR,
            params=(0.001, 0.002, -0.001),
        )
        s = cb.WarpCurvedBoundaryLBM3D(
            g,
            material=material,
            boundary=boundary,
            boundary_reconstruction="local_f",
            lattice_speed=10.0,
            collision_omega=1.8,
        )
        s.initialize_from_numpy(U=reference["U"], apply_initial_boundaries=False)
        for _ in range(25):
            s.step()
        np.testing.assert_allclose(s.f0.numpy(), reference["f25"], rtol=0, atol=1e-13)
        self.assertIsNone(s.characteristic_stencil)
        self.assertEqual(s.summary()["characteristic_cut_links"], 0)

    def test_quadratic_gradients_and_affine_predictor(self):
        g = sphere()
        tables = cb._sparse_tables_from_geometry(g)
        fit = build_characteristic_weights(g, tables)
        xyz = np.column_stack([(tables[f"active_{a}"] + 0.5) * g.dx for a in ("i", "j", "k")])
        x, y, z = xyz.T
        values = (
            1
            + 0.2 * x
            - 0.3 * y
            + 0.1 * z
            + 0.4 * x * x
            + 0.7 * x * y
            - 0.2 * x * z
            + 0.3 * y * y
            + 0.6 * y * z
            - 0.1 * z * z
        )
        ids = fit["indices"]
        sampled = values[np.maximum(ids, 0)]
        got = np.einsum("akl,kl->al", fit["weights"][1:], sampled)
        x, y, z = g.link_xb, g.link_yb, g.link_zb
        exact = np.array(
            (
                0.2 + 0.8 * x + 0.7 * y - 0.2 * z,
                -0.3 + 0.7 * x + 0.6 * y + 0.6 * z,
                0.1 - 0.2 * x + 0.6 * y - 0.2 * z,
            )
        )
        np.testing.assert_allclose(got, exact, rtol=2e-9, atol=2e-9)
        affine = 1 + 0.2 * xyz[:, 0] - 0.3 * xyz[:, 1] + 0.1 * xyz[:, 2]
        predicted = np.einsum("kl,kl->l", fit["weights"][0], affine[np.maximum(ids, 0)])
        good = fit["probe_modes"] != 1
        np.testing.assert_allclose(
            predicted[good], (1 + 0.2 * x - 0.3 * y + 0.1 * z)[good], rtol=1e-12, atol=1e-12
        )
        selection = g.link_zb > 0.5
        restricted = build_characteristic_weights(g, tables, link_mask=selection)
        np.testing.assert_allclose(
            restricted["weights"][:, :, selection],
            fit["weights"][:, :, selection],
            atol=1e-13,
            rtol=1e-12,
        )
        self.assertTrue(np.all(restricted["indices"][:, ~selection] == -1))
        self.assertTrue(np.all(restricted["weights"][:, :, ~selection] == 0))
        used = np.abs(fit["weights"]).sum(axis=0) > 0
        ctr = np.column_stack((g.link_i, g.link_j, g.link_k))
        sample_xyz = xyz[np.maximum(ids, 0)] / g.dx - 0.5
        self.assertLessEqual(float(np.max(np.abs(sample_xyz - ctr[None, :, :])[used])), 3.00000001)

    def test_boundary_mechanics_and_population_formula(self):
        g = cb.CurvedBoundaryGeometry3D.sphere(nx=20, ny=20, nz=20, radius=0.35)
        original = np.load(Path(__file__).parent / "data" / "dirichlet_bfl_3d_reference.npz")
        material = cb.WarpHyperelasticMaterial3D.from_poisson("neo_hooke", 0.2, mu=1.0)
        for kinds in [("neumann",) * 3, ("dirichlet", "neumann", "neumann")]:
            boundary = cb.CurvedBoundarySpec3D(
                kind="neumann",
                component_kinds=kinds,
                value_mode=cb.CURVED3D_VALUE_CONSTANT_VECTOR,
                params=(0.001, 0.002, -0.001),
            )
            s = cb.WarpCurvedBoundaryLBM3D(
                g,
                material=material,
                boundary=boundary,
                boundary_reconstruction="local_f",
                lattice_speed=10.0,
                source_mode="damping",
                damping_gamma=0.7,
            )
            x, y, z = s.active_coordinates()
            U = np.zeros((12, s.n_active))
            U[0] = 0.03 * x * y
            U[1] = 0.02 * y * z
            U[2] = -0.01 * z * x
            U[3] = 1 + 0.01 * x
            U[7] = 1 + 0.01 * y
            U[11] = 1 + 0.01 * z
            U[4] = 0.003 * y
            s.initialize_from_numpy(
                U=U, displacement=np.zeros((3, s.n_active)), apply_initial_boundaries=False
            )
            cb.wp.copy(
                s.f0, cb._as_warp(original["f_source"], dtype=cb.wp.float64, device=s.device)
            )
            s.apply_boundary_reconstruction(0.03)
            state = s.char_state.numpy()
            Fb = s.link_Fb.numpy()
            Pb = s.link_Pb.numpy()
            bv = s.link_bv.numpy()
            current = s.U.numpy()
            output = s.f1.numpy()
            nodes = s.link_p.numpy()
            normal = np.array((g.link_nx, g.link_ny, g.link_nz))
            traction = np.einsum("ijl,jl->il", Pb.reshape(3, 3, -1), normal)
            for a, kind in enumerate(kinds):
                if kind == "neumann":
                    np.testing.assert_allclose(traction[a], bv[a], atol=2e-9, rtol=1e-7)
            for ell in np.linspace(0, g.n_links - 1, 31, dtype=int):
                qi = (g.link_q[ell] + 3) % 6
                d = cb.DIRS3[qi].astype(float)
                Ub = np.r_[current[:3, nodes[ell]], Fb[:, ell]]
                for a, kind in enumerate(kinds):
                    if kind == "dirichlet":
                        Ub[a] = bv[a, ell]
                G = state[1:, :, ell]
                Uc = Ub + g.link_eta[ell] * s.dt * s.lattice_speed * (d @ G)
                B = np.zeros(12)
                B[:3] = -0.7 * Uc[:3]
                w = (
                    B
                    - sum(derivative(Uc, G[a], material, a) for a in range(3))
                    + s.lattice_speed * (d @ G)
                )
                feq = (
                    Uc + 3 / s.lattice_speed * sum(d[a] * flux(Uc, material, a) for a in range(3))
                ) / 6
                deq = (
                    w
                    + 3
                    / s.lattice_speed
                    * sum(d[a] * derivative(Uc, w, material, a) for a in range(3))
                ) / 6
                expected = (
                    feq
                    - s.dt / s.collision_omega * deq
                    + s.dt * (1 / s.collision_omega - 0.5) * B / 6
                )
                rows = np.array([a if a < 3 else (a - 3) // 3 for a in range(12)])
                neumann = np.array(kinds)[rows] == "neumann"
                np.testing.assert_allclose(
                    output[12 * qi : 12 * (qi + 1), nodes[ell]][neumann],
                    expected[neumann],
                    rtol=2e-8,
                    atol=2e-10,
                )
            slots = 12 * ((g.link_q + 3) % 6)[:, None] + np.arange(12)[None, :]
            np.testing.assert_allclose(
                output[slots[:, ~neumann], nodes[:, None]],
                original["dirichlet_incoming"][:, ~neumann],
                rtol=0,
                atol=1e-13,
            )
            # Neumann reconstruction uses only local U; Dirichlet retains its population history.
            before = s.f1.numpy().copy()
            cb.wp.copy(
                s.f0, cb._as_warp(np.full(s.f0.shape, 123.0), dtype=cb.wp.float64, device=s.device)
            )
            s.apply_boundary_reconstruction(0.03)
            after = s.f1.numpy()
            slots = 12 * ((g.link_q + 3) % 6)[:, None] + np.arange(12)[None, :]
            np.testing.assert_array_equal(
                before[slots[:, neumann], nodes[:, None]], after[slots[:, neumann], nodes[:, None]]
            )
            if np.any(~neumann):
                self.assertGreater(
                    np.max(
                        np.abs(
                            before[slots[:, ~neumann], nodes[:, None]]
                            - after[slots[:, ~neumann], nodes[:, None]]
                        )
                    ),
                    1.0,
                )


if __name__ == "__main__":
    unittest.main()
