#!/usr/bin/env python3
"""Spherical-shell radial BVP benchmark for the 3-D curved-boundary solver.

The reference body is a hollow sphere,

    Ri < |X-C| < Ro.

The exact/reference solution is radially symmetric:

    x(X) = C + r(R) e_R,  R=|X-C|.

The 1-D reference solves the total-Lagrangian spherical equilibrium equation

    dP_RR/dR + 2 (P_RR - P_TT) / R = 0

for the same compressible neo-Hookean model used by the LBM code.  The inner
hole is expanded and the outer sphere is traction-free.  The resulting radial
tractions are applied through boundary ids: outer=1, inner=2.
"""

from __future__ import annotations

import argparse
from traction_settings import apply_defaults
from dataclasses import dataclass
import json
import math
from pathlib import Path
from _paths import ROOT, RESULTS, REFERENCES, FENICSX_PYTHON, install_reference
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp

SCRIPT_DIR = Path(__file__).resolve().parent
NEWCODE_ROOT = ROOT / "src"
PAPER_ROOT = ROOT


from curved_lbm.three_d import curved_boundary_warp3d as cb


CENTER = np.array([0.5, 0.5, 0.5], dtype=np.float64)
RI = 0.20
RO = 0.38
INNER_DEFORMED_RADIUS = 0.28
T_RUN = 5.0
RAMP_TIME = 2.0
DAMPING_GAMMA = 10.0
LATTICE_SPEED = 10.0
COLLISION_OMEGA = 1.8
DATA_DIR = RESULTS / "fig7_8_spherical_shell" / "data"


def effective_omega(base_omega: float, dx: float, mode: str, kappa: float) -> float:
    if mode == "constant":
        omega = float(base_omega)
    elif mode == "dx":
        omega = 2.0 - float(kappa) * float(dx)
    else:
        raise ValueError(f"unsupported omega scaling: {mode}")
    if not (0.0 < omega <= 2.0):
        raise ValueError(f"effective omega={omega} outside (0,2]")
    return omega


def radius_array(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    X, Y, Z = np.broadcast_arrays(X, Y, Z)
    return np.sqrt((X - CENTER[0]) ** 2 + (Y - CENTER[1]) ** 2 + (Z - CENTER[2]) ** 2)


def shell_phi(x: float, y: float, z: float) -> float:
    r = float(radius_array(np.asarray(x), np.asarray(y), np.asarray(z)))
    return max(r - RO, RI - r)


def shell_phi_array(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    R = radius_array(X, Y, Z)
    return np.maximum(R - RO, RI - R)


def shell_normal(x: float, y: float, z: float) -> tuple[float, float, float]:
    d = np.array([x, y, z], dtype=np.float64) - CENTER
    r = max(float(np.linalg.norm(d)), 1.0e-30)
    e = d / r
    if abs(r - RI) <= abs(r - RO):
        e = -e
    return float(e[0]), float(e[1]), float(e[2])


def shell_normal_array(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Y, Z = np.broadcast_arrays(X, Y, Z)
    dx = X - CENTER[0]
    dy = Y - CENTER[1]
    dz = Z - CENTER[2]
    R = np.maximum(np.sqrt(dx * dx + dy * dy + dz * dz), 1.0e-30)
    sign = np.where(np.abs(R - RI) <= np.abs(R - RO), -1.0, 1.0)
    return sign * dx / R, sign * dy / R, sign * dz / R


def shell_boundary_id(x: float, y: float, z: float, nx: float, ny: float, nz: float) -> int:
    r = float(radius_array(np.asarray(x), np.asarray(y), np.asarray(z)))
    return cb.BOUNDARY_ID_INNER if abs(r - RI) <= abs(r - RO) else cb.BOUNDARY_ID_OUTER


@dataclass
class SphericalRadialReference:
    material: cb.WarpHyperelasticMaterial3D
    ri: float = RI
    ro: float = RO
    inner_deformed_radius: float = INNER_DEFORMED_RADIUS
    tol: float = 1.0e-9

    def __post_init__(self) -> None:
        self._solve()

    def piola_principal(
        self, lam_r: np.ndarray, lam_t: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mu = float(self.material.mu)
        lam = float(self.material.lam)
        J = lam_r * lam_t * lam_t
        p_r = mu * (lam_r - 1.0 / lam_r) + 0.5 * lam * (J * J - 1.0) / lam_r
        p_t = mu * (lam_t - 1.0 / lam_t) + 0.5 * lam * (J * J - 1.0) / lam_t
        return p_r, p_t

    def _rhs(self, R: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = y[0]
        lam_r = y[1]
        lam_t = r / R
        p_r, p_t = self.piola_principal(lam_r, lam_t)
        mu = float(self.material.mu)
        lam = float(self.material.lam)
        dpr_dlamr = mu + 0.5 * lam * lam_t**4 + (mu + 0.5 * lam) / (lam_r * lam_r)
        dpr_dlamt = 2.0 * lam * lam_r * lam_t**3
        dlamt_dR = (lam_r - lam_t) / R
        dlamr_dR = (-2.0 * (p_r - p_t) / R - dpr_dlamt * dlamt_dR) / dpr_dlamr
        return np.vstack((lam_r, dlamr_dR))

    def _bc(self, ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        p_ro, _p_to = self.piola_principal(np.asarray(yb[1]), np.asarray(yb[0] / self.ro))
        return np.array([ya[0] - self.inner_deformed_radius, float(p_ro)])

    def _solve(self) -> None:
        R = np.linspace(self.ri, self.ro, 260)
        outer_guess = self.ro + 0.35 * (self.inner_deformed_radius - self.ri)
        r_guess = self.inner_deformed_radius + (outer_guess - self.inner_deformed_radius) * (
            R - self.ri
        ) / (self.ro - self.ri)
        lam_guess = np.gradient(r_guess, R)
        sol = solve_bvp(
            self._rhs, self._bc, R, np.vstack((r_guess, lam_guess)), tol=self.tol, max_nodes=20000
        )
        if not sol.success:
            raise RuntimeError(f"solve_bvp failed: {sol.message}")
        self.sol = sol
        ri_state = sol.sol(np.asarray([self.ri]))
        ro_state = sol.sol(np.asarray([self.ro]))
        self.outer_deformed_radius = float(ro_state[0, 0])
        self.inner_lam_r = float(ri_state[1, 0])
        self.outer_lam_r = float(ro_state[1, 0])
        self.inner_lam_t = self.inner_deformed_radius / self.ri
        self.outer_lam_t = self.outer_deformed_radius / self.ro
        self.inner_traction = float(
            self.piola_principal(np.asarray(self.inner_lam_r), np.asarray(self.inner_lam_t))[0]
        )
        self.outer_traction = float(
            self.piola_principal(np.asarray(self.outer_lam_r), np.asarray(self.outer_lam_t))[0]
        )

    def radial_state(self, R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        R = np.asarray(R, dtype=np.float64)
        R_eval = np.clip(R, self.ri, self.ro)
        y = self.sol.sol(R_eval)
        r = y[0]
        lam_r = y[1]
        R_safe = np.maximum(R, 0.5 * self.ri)
        lam_t = r / R_safe
        return r, lam_r, lam_t

    def fields(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X, Y, Z = np.broadcast_arrays(X, Y, Z)
        dx = X - CENTER[0]
        dy = Y - CENTER[1]
        dz = Z - CENTER[2]
        R = np.sqrt(dx * dx + dy * dy + dz * dz)
        R_safe = np.maximum(R, 1.0e-30)
        ex = dx / R_safe
        ey = dy / R_safe
        ez = dz / R_safe
        r, lam_r, lam_t = self.radial_state(R)
        scale = r / np.maximum(R, 0.5 * self.ri)

        u = np.zeros((3,) + X.shape, dtype=np.float64)
        u[0] = (scale - 1.0) * dx
        u[1] = (scale - 1.0) * dy
        u[2] = (scale - 1.0) * dz

        F = np.zeros((3, 3) + X.shape, dtype=np.float64)
        ee = (ex, ey, ez)
        for a in range(3):
            for b in range(3):
                F[a, b] = (lam_r - lam_t) * ee[a] * ee[b]
                if a == b:
                    F[a, b] += lam_t

        U = np.zeros((12,) + X.shape, dtype=np.float64)
        U[3:12] = F.reshape((9,) + X.shape)
        return u, U, F, R, lam_r, lam_t

    def summary(self) -> dict[str, float | int]:
        sample_R = np.linspace(self.ri, self.ro, 2000)
        _r, lam_r, lam_t = self.radial_state(sample_R)
        J = lam_r * lam_t * lam_t
        return {
            "inner_radius": self.ri,
            "outer_radius": self.ro,
            "inner_deformed_radius": self.inner_deformed_radius,
            "outer_deformed_radius": self.outer_deformed_radius,
            "inner_traction_amp": self.inner_traction,
            "outer_traction_amp": self.outer_traction,
            "min_lambda_r": float(np.min(lam_r)),
            "max_lambda_r": float(np.max(lam_r)),
            "min_lambda_theta": float(np.min(lam_t)),
            "max_lambda_theta": float(np.max(lam_t)),
            "min_J": float(np.min(J)),
            "max_J": float(np.max(J)),
            "solve_bvp_nodes": int(self.sol.x.size),
            "solve_bvp_max_rms_residual": float(np.max(self.sol.rms_residuals)),
        }


def radial_traction_boundary(
    ref: SphericalRadialReference, *, ramp_time: float
) -> cb.CurvedBoundarySpec3D:
    value_mode = cb.CURVED3D_VALUE_SINE2_NORMAL_TRACTION
    outer_params = (ref.outer_traction, float(ramp_time), 0.0)
    inner_params = (ref.inner_traction, float(ramp_time), 0.0)
    return cb.CurvedBoundarySpec3D(
        kind="neumann",
        value_mode=value_mode,
        params=(0.0,),
        id_kinds={cb.BOUNDARY_ID_OUTER: "neumann", cb.BOUNDARY_ID_INNER: "neumann"},
        id_value_modes={
            cb.BOUNDARY_ID_OUTER: value_mode,
            cb.BOUNDARY_ID_INNER: value_mode,
        },
        id_params={cb.BOUNDARY_ID_OUTER: outer_params, cb.BOUNDARY_ID_INNER: inner_params},
        label="spherical shell radial BVP tractions: sine2 ramp then hold",
    )


def relative_l2_all(error: np.ndarray, reference: np.ndarray) -> float:
    e = np.asarray(error)
    r = np.asarray(reference)
    return float(np.sqrt(np.sum(e * e)) / max(np.sqrt(np.sum(r * r)), np.finfo(float).eps))


def cauchy_stress_from_F(
    F: np.ndarray, material: cb.WarpHyperelasticMaterial3D
) -> tuple[np.ndarray, np.ndarray]:
    P = cb._first_piola_np(F, material)
    J = np.maximum(cb._det3_np(F), 1.0e-12)
    sigma = np.einsum("ia...,ja...->ij...", P, F) / J
    return P, sigma


def run_one(
    n: int,
    *,
    device: str,
    ref: SphericalRadialReference,
    run_time: float,
    ramp_time: float,
    min_run_time: float,
    steady_rms_v_tol: float,
    steady_check_every: int,
    steady_hold_checks: int,
    progress_every: int,
    lattice_speed: float,
    collision_omega: float,
    damping_gamma: float,
    boundary_reconstruction: str,
    local_displacement_interval: int,
    local_displacement_sweeps: int,
    local_displacement_relax: float,
    local_displacement_boundary_weight: float,
    local_compatibility_interval: int,
    local_compatibility_blend: float,
    results_dir: Path,
    save_state: bool,
) -> dict[str, float | int | bool | str]:
    geom = cb.CurvedBoundaryGeometry3D.from_level_set(
        nx=n,
        ny=n,
        nz=n,
        length_x=1.0,
        length_y=1.0,
        length_z=1.0,
        phi=shell_phi,
        normal=shell_normal,
        boundary_id=shell_boundary_id,
        phi_vectorized=shell_phi_array,
        normal_vectorized=shell_normal_array,
        label="spherical_shell_radial_bvp",
    )
    solver = cb.WarpCurvedBoundaryLBM3D(
        geom,
        material=ref.material,
        boundary=radial_traction_boundary(ref, ramp_time=ramp_time),
        lattice_speed=lattice_speed,
        collision_omega=collision_omega,
        boundary_reconstruction=boundary_reconstruction,
        source_mode="damping",
        damping_gamma=damping_gamma,
        device=device,
        check_linearized_cfl=True,
        local_displacement_interval=local_displacement_interval,
        local_displacement_sweeps=local_displacement_sweeps,
        local_displacement_relax=local_displacement_relax,
        local_displacement_boundary_weight=local_displacement_boundary_weight,
        local_displacement_boundary_id=-1,
        local_compatibility_interval=local_compatibility_interval,
        local_compatibility_blend=local_compatibility_blend,
    )
    X, Y, Z = solver.active_coordinates()
    u_ref, U_ref, F_ref, _R, _lr, _lt = ref.fields(X, Y, Z)
    solver.initialize_identity(init_order=1)

    history: list[dict[str, float | int | bool]] = []
    steady_count = 0
    stopped_by_steady = False
    check_every = max(int(steady_check_every), 0)
    hold_checks = max(int(steady_hold_checks), 1)
    progress_every = int(progress_every)

    def record_state() -> dict[str, float | int | bool]:
        solver.refresh_current()
        U_now = solver.active_system_state()
        u_now = solver.active_displacement()
        F_now = solver.active_deformation_gradient()
        J_now = solver.active_deformation_jacobian()
        v_mag_now = np.sqrt(U_now[0] * U_now[0] + U_now[1] * U_now[1] + U_now[2] * U_now[2])
        state = {
            "step": int(solver.steps),
            "time": float(solver.time),
            "load_scale": float(
                1.0
                if ramp_time <= 0.0 or solver.time >= ramp_time
                else math.sin(math.pi * solver.time / (2.0 * ramp_time)) ** 2
            ),
            "max_abs_u": float(np.max(np.sqrt(np.sum(u_now * u_now, axis=0)))),
            "rms_v": float(np.sqrt(np.mean(v_mag_now * v_mag_now))),
            "max_abs_v": float(np.max(v_mag_now)),
            "rel_l2_u_to_exact": relative_l2_all(u_now - u_ref, u_ref),
            "rel_l2_F_to_exact": relative_l2_all(F_now - F_ref, F_ref),
            "min_J": float(np.min(J_now)),
            "max_J": float(np.max(J_now)),
            "finite": bool(
                np.isfinite(U_now).all() and np.isfinite(u_now).all() and np.all(J_now > 0.0)
            ),
        }
        history.append(state)
        return state

    record_state()
    while solver.time < float(run_time) - 0.5 * solver.dt:
        solver.step()
        should_check = (
            check_every > 0 and solver.steps % check_every == 0
        ) or solver.time >= float(run_time) - 0.5 * solver.dt
        if should_check:
            state = record_state()
            if not bool(state["finite"]):
                break
            if progress_every > 0 and (
                solver.steps % progress_every == 0
                or solver.time >= float(run_time) - 0.5 * solver.dt
            ):
                print(
                    f"n={n} step={solver.steps} t={solver.time:.4f} "
                    f"load={float(state['load_scale']):.3f} rms_v={float(state['rms_v']):.3e} "
                    f"rel_u={float(state['rel_l2_u_to_exact']):.3e} rel_F={float(state['rel_l2_F_to_exact']):.3e}",
                    flush=True,
                )
            if (
                check_every > 0
                and solver.time >= float(min_run_time)
                and float(state["rms_v"]) <= float(steady_rms_v_tol)
            ):
                steady_count += 1
            else:
                steady_count = 0
            if steady_count >= hold_checks:
                stopped_by_steady = True
                break

    U_num = solver.active_system_state()
    u_num = solver.active_displacement()
    F_num = solver.active_deformation_gradient()
    sigma_num = solver.active_sigma()
    P_ref, sigma_ref = cauchy_stress_from_F(F_ref, ref.material)
    F_err = F_num - F_ref
    v_mag = np.sqrt(U_num[0] * U_num[0] + U_num[1] * U_num[1] + U_num[2] * U_num[2])
    J = solver.active_deformation_jacobian()
    ids, counts = np.unique(geom.link_boundary_id, return_counts=True)
    row: dict[str, float | int | bool | str] = {
        "case": "spherical_shell_radial_bvp_neumann_ids",
        "n": int(n),
        "dx": float(geom.dx),
        "active_nodes": int(solver.n_active),
        "cut_links": int(geom.n_links),
        "outer_cut_links": int(counts[np.where(ids == cb.BOUNDARY_ID_OUTER)[0][0]])
        if np.any(ids == cb.BOUNDARY_ID_OUTER)
        else 0,
        "inner_cut_links": int(counts[np.where(ids == cb.BOUNDARY_ID_INNER)[0][0]])
        if np.any(ids == cb.BOUNDARY_ID_INNER)
        else 0,
        "steps": int(solver.steps),
        "time": float(solver.time),
        "max_run_time": float(run_time),
        "ramp_time": float(ramp_time),
        "min_run_time": float(min_run_time),
        "steady_rms_v_tol": float(steady_rms_v_tol),
        "steady_check_every": int(steady_check_every),
        "steady_hold_checks": int(steady_hold_checks),
        "stopped_by_steady": bool(stopped_by_steady),
        "lattice_speed": float(lattice_speed),
        "collision_omega": float(collision_omega),
        "damping_gamma": float(damping_gamma),
        "boundary_reconstruction": solver.boundary_reconstruction_name,
        "compatibility_projection_active": bool(
            solver._has_neumann_cut_links and solver.compatibility_projection_interval > 0
        ),
        "init_order": 1,
        "rel_l2_u": relative_l2_all(u_num - u_ref, u_ref),
        "rel_l2_F": relative_l2_all(F_err, F_ref),
        "rel_l2_sigma": relative_l2_all(sigma_num - sigma_ref, sigma_ref),
        "rms_v": float(np.sqrt(np.mean(v_mag * v_mag))),
        "max_abs_v": float(np.max(v_mag)),
        "min_J": float(np.min(J)),
        "max_J": float(np.max(J)),
        "finite": bool(not solver.invalid_state()),
    }
    if save_state:
        solver.save_npz(results_dir / f"spherical_shell_radial_bvp_n{n}_final.npz")
        np.savez_compressed(
            results_dir / f"spherical_shell_radial_bvp_reference_n{n}.npz",
            storage=np.array("sparse_active"),
            x_active=X,
            y_active=Y,
            z_active=Z,
            active_i=solver.active_i,
            active_j=solver.active_j,
            active_k=solver.active_k,
            shape=np.array([solver.nx, solver.ny, solver.nz], dtype=np.int32),
            dx=np.array(solver.dx),
            u=u_ref,
            U=U_ref,
            F=F_ref,
            P=P_ref,
            sigma=sigma_ref,
        )
        (results_dir / f"spherical_shell_radial_bvp_history_n{n}.json").write_text(
            json.dumps(history, indent=2)
        )
    print(json.dumps(row, indent=2))
    return row


def add_orders(rows: list[dict[str, float | int | bool | str]], key: str) -> None:
    for i in range(1, len(rows)):
        e0 = float(rows[i - 1][key])
        e1 = float(rows[i][key])
        h0 = float(rows[i - 1]["dx"])
        h1 = float(rows[i]["dx"])
        rows[i][f"order_{key}"] = (
            float(math.log(e1 / e0) / math.log(h1 / h0)) if e0 > 0.0 and e1 > 0.0 else float("nan")
        )


def plot_convergence(rows: list[dict[str, float | int | bool | str]], results_dir: Path) -> None:
    dx = np.array([float(r["dx"]) for r in rows])
    n = np.array([int(r["n"]) for r in rows])
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for key, label, color in (
        ("rel_l2_u", "u", "tab:blue"),
        ("rel_l2_F", "F", "tab:green"),
        ("rms_v", "rms(v)", "tab:orange"),
    ):
        err = np.array([float(r[key]) for r in rows])
        ax0.loglog(dx, err, marker="o", lw=1.8, color=color, label=label)
        order = np.array([float(r.get(f"order_{key}", np.nan)) for r in rows])
        ax1.plot(n[1:], order[1:], marker="o", lw=1.8, color=color, label=label)
    ref2 = float(rows[-1]["rel_l2_F"]) * (dx / dx[-1]) ** 2
    ax0.loglog(dx, ref2, "--", color="0.35", lw=1.2, label="slope 2")
    ax0.invert_xaxis()
    ax0.grid(True, which="both", ls=":", alpha=0.5)
    ax0.set_xlabel("dx")
    ax0.set_ylabel("error magnitude")
    ax0.set_title("Spherical shell radial BVP")
    ax0.legend()
    ax1.axhline(2.0, color="0.35", ls="--", lw=1.2)
    ax1.grid(True, ls=":", alpha=0.5)
    ax1.set_xlabel("grid n")
    ax1.set_ylabel("observed order")
    ax1.set_xticks(n[1:])
    ax1.set_title("Pairwise observed order")
    ax1.legend()
    fig.savefig(results_dir / "spherical_shell_radial_bvp_convergence.png", dpi=190)
    plt.close(fig)


def sphere_surface(
    radius: float, *, ntheta: int = 80, nphi: int = 120, cutaway: bool = True
) -> np.ndarray:
    theta = np.linspace(0.0, np.pi, ntheta)
    if cutaway:
        phi = np.linspace(0.35 * np.pi, 1.85 * np.pi, nphi)
    else:
        phi = np.linspace(0.0, 2.0 * np.pi, nphi)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    return np.stack(
        [
            CENTER[0] + radius * np.sin(TH) * np.cos(PH),
            CENTER[1] + radius * np.sin(TH) * np.sin(PH),
            CENTER[2] + radius * np.cos(TH),
        ],
        axis=0,
    )


def plot_deformed_surfaces(ref: SphericalRadialReference, results_dir: Path) -> None:
    fig = plt.figure(figsize=(8.2, 7.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for radius, cmap, alpha, label in (
        (ref.ro, "viridis", 0.46, "outer"),
        (ref.ri, "magma", 0.82, "inner"),
    ):
        X0 = sphere_surface(radius, cutaway=True)
        R0 = np.full(X0.shape[1:], radius)
        r, _lr, _lt = ref.radial_state(R0)
        d = X0 - CENTER[:, None, None]
        Xd = CENTER[:, None, None] + d * (r / radius)
        umag = np.abs(r - radius)
        norm = plt.Normalize(
            float(np.min(umag)),
            float(np.max(umag))
            if float(np.max(umag)) > float(np.min(umag))
            else float(np.min(umag)) + 1.0e-12,
        )
        ax.plot_surface(
            Xd[0],
            Xd[1],
            Xd[2],
            facecolors=plt.get_cmap(cmap)(norm(umag)),
            linewidth=0.0,
            antialiased=True,
            shade=False,
            alpha=alpha,
        )
        ax.plot_wireframe(
            X0[0], X0[1], X0[2], rstride=10, cstride=12, color="0.35", linewidth=0.35, alpha=0.45
        )
        ax.text(*(CENTER + np.array([0.0, 0.0, r[0, 0] + 0.02])), label, fontsize=8)
    lo = CENTER - ref.outer_deformed_radius * 1.12
    hi = CENTER + ref.outer_deformed_radius * 1.12
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Cutaway deformed spherical shell and initial wireframes")
    ax.view_init(elev=22.0, azim=-50.0)
    fig.savefig(results_dir / "spherical_shell_radial_bvp_deformed_surfaces.png", dpi=220)
    plt.close(fig)


def plot_radial_profiles(npz_path: Path, ref: SphericalRadialReference, results_dir: Path) -> None:
    data = np.load(npz_path)
    X = data["x_active"]
    Y = data["y_active"]
    Z = data["z_active"]
    u = data["u"]
    U = data["U"]
    F = data["F"]
    u_ref, _U_ref, F_ref, R, lam_r_ref, lam_t_ref = ref.fields(X, Y, Z)
    e = np.stack([X - CENTER[0], Y - CENTER[1], Z - CENTER[2]], axis=0)
    e /= np.maximum(np.sqrt(np.sum(e * e, axis=0)), 1.0e-30)
    lam_r_num = np.einsum("an,abn,bn->n", e, F, e)
    trF = F[0, 0] + F[1, 1] + F[2, 2]
    lam_t_num = 0.5 * (trF - lam_r_num)
    u_mag = np.sqrt(np.sum(u * u, axis=0))
    u_ref_mag = np.sqrt(np.sum(u_ref * u_ref, axis=0))
    F_err = np.sqrt(np.sum((F - F_ref) * (F - F_ref), axis=(0, 1)))
    v_mag = np.sqrt(U[0] * U[0] + U[1] * U[1] + U[2] * U[2])

    bins = np.linspace(ref.ri, ref.ro, 90)
    centers = 0.5 * (bins[:-1] + bins[1:])
    which = np.digitize(R, bins) - 1

    def bmean(vals: np.ndarray) -> np.ndarray:
        out = np.full(centers.shape, np.nan)
        for i in range(centers.size):
            m = which == i
            if np.any(m):
                out[i] = float(np.mean(vals[m]))
        return out

    fig, axs = plt.subplots(2, 2, figsize=(11.8, 8.0), constrained_layout=True)
    axs[0, 0].plot(centers, bmean(u_mag), lw=1.8, label="LBM")
    axs[0, 0].plot(centers, bmean(u_ref_mag), "--", lw=1.5, label="BVP")
    axs[0, 0].set_ylabel("|u|")
    axs[0, 0].legend()
    axs[0, 1].plot(centers, bmean(lam_r_num), lw=1.8, label="lambda_r LBM")
    axs[0, 1].plot(centers, bmean(lam_r_ref), "--", lw=1.5, label="lambda_r BVP")
    axs[0, 1].plot(centers, bmean(lam_t_num), lw=1.8, label="lambda_t LBM")
    axs[0, 1].plot(centers, bmean(lam_t_ref), "--", lw=1.5, label="lambda_t BVP")
    axs[0, 1].legend(fontsize=8)
    axs[1, 0].semilogy(centers, bmean(F_err), lw=1.8)
    axs[1, 0].set_ylabel("mean ||F-F_ref||")
    axs[1, 1].semilogy(centers, bmean(v_mag), lw=1.8)
    axs[1, 1].set_ylabel("mean |v|")
    for ax in axs.ravel():
        ax.grid(True, ls=":", alpha=0.5)
        ax.set_xlabel("R")
    fig.suptitle(f"Spherical shell radial profiles, n={int(data['shape'][0])}")
    fig.savefig(results_dir / "spherical_shell_radial_bvp_radial_profiles.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n", default="200")
    parser.add_argument("--inner-deformed-radius", type=float, default=INNER_DEFORMED_RADIUS)
    parser.add_argument("--run-time", type=float, default=T_RUN)
    parser.add_argument("--ramp-time", type=float, default=RAMP_TIME)
    parser.add_argument("--min-run-time", type=float, default=3.0)
    parser.add_argument("--steady-rms-v-tol", type=float, default=0.0)
    parser.add_argument(
        "--steady-check-every",
        type=int,
        default=0,
        help="optional CPU diagnostic interval; 0 keeps the time loop GPU-only",
    )
    parser.add_argument("--steady-hold-checks", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--lattice-speed", type=float, default=LATTICE_SPEED)
    parser.add_argument("--omega", type=float, default=COLLISION_OMEGA)
    parser.add_argument("--omega-scaling", choices=("constant", "dx"), default="constant")
    parser.add_argument("--omega-kappa", type=float, default=50.0)
    parser.add_argument("--damping-gamma", type=float, default=DAMPING_GAMMA)
    parser.add_argument(
        "--boundary-reconstruction",
        choices=("bfl", "halfway", "compat_bfl", "local_f"),
        default="local_f",
    )
    parser.add_argument("--local-displacement-interval", type=int, default=1)
    parser.add_argument("--local-displacement-sweeps", type=int, default=1)
    parser.add_argument("--local-displacement-relax", type=float, default=0.85)
    parser.add_argument("--local-displacement-boundary-weight", type=float, default=100.0)
    parser.add_argument("--local-compatibility-interval", type=int, default=1)
    parser.add_argument("--local-compatibility-blend", type=float, default=0.5)
    parser.add_argument("--results-dir", default=str(DATA_DIR))
    apply_defaults(parser)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    material = cb.WarpHyperelasticMaterial3D.from_poisson("neo_hooke", 0.20, mu=1.0)
    ref = SphericalRadialReference(
        material=material, inner_deformed_radius=float(args.inner_deformed_radius)
    )
    n_values = [int(v) for v in str(args.n).split(",") if v.strip()]
    rows: list[dict[str, float | int | bool | str]] = []
    for n in n_values:
        omega_n = effective_omega(
            float(args.omega), 1.0 / float(n), str(args.omega_scaling), float(args.omega_kappa)
        )
        rows.append(
            run_one(
                n,
                device=str(args.device),
                ref=ref,
                run_time=float(args.run_time),
                ramp_time=float(args.ramp_time),
                min_run_time=float(args.min_run_time),
                steady_rms_v_tol=float(args.steady_rms_v_tol),
                steady_check_every=int(args.steady_check_every),
                steady_hold_checks=int(args.steady_hold_checks),
                progress_every=int(args.progress_every),
                lattice_speed=float(args.lattice_speed),
                collision_omega=omega_n,
                damping_gamma=float(args.damping_gamma),
                boundary_reconstruction=str(args.boundary_reconstruction),
                local_displacement_interval=int(args.local_displacement_interval),
                local_displacement_sweeps=int(args.local_displacement_sweeps),
                local_displacement_relax=float(args.local_displacement_relax),
                local_displacement_boundary_weight=float(args.local_displacement_boundary_weight),
                local_compatibility_interval=int(args.local_compatibility_interval),
                local_compatibility_blend=float(args.local_compatibility_blend),
                results_dir=results_dir,
                save_state=(n == n_values[-1]),
            )
        )
    for key in ("rel_l2_u", "rel_l2_F", "rel_l2_sigma", "rms_v"):
        add_orders(rows, key)

    summary = {
        "case": "spherical_shell_radial_bvp_neumann_ids",
        "geometry": {"center": CENTER.tolist(), "inner_radius": RI, "outer_radius": RO},
        "boundary_ids": {"outer": cb.BOUNDARY_ID_OUTER, "inner": cb.BOUNDARY_ID_INNER},
        "material": {"model": material.model, "lambda": material.lam, "mu": material.mu},
        "reference": ref.summary(),
        "mode": "dynamic_relax_from_identity",
        "run_time_requested": float(args.run_time),
        "ramp_time": float(args.ramp_time),
        "min_run_time": float(args.min_run_time),
        "steady_rms_v_tol": float(args.steady_rms_v_tol),
        "steady_check_every": int(args.steady_check_every),
        "steady_hold_checks": int(args.steady_hold_checks),
        "lattice_speed": float(args.lattice_speed),
        "omega_input": float(args.omega),
        "omega_scaling": str(args.omega_scaling),
        "omega_kappa": float(args.omega_kappa),
        "damping_gamma": float(args.damping_gamma),
        "boundary_reconstruction": str(args.boundary_reconstruction),
        "compatibility_projection": {"active": False, "implementation": "not available in newcode"},
        "local_method": {
            "displacement_interval": int(args.local_displacement_interval),
            "displacement_sweeps": int(args.local_displacement_sweeps),
            "displacement_relax": float(args.local_displacement_relax),
            "boundary_weight": float(args.local_displacement_boundary_weight),
            "F_repair_interval": int(args.local_compatibility_interval),
            "F_repair_blend": float(args.local_compatibility_blend),
            "kernel_stencil_radius": 2,
            "displacement_dependency_radius": int(args.local_displacement_sweeps),
            "composed_compatibility_radius": int(args.local_displacement_sweeps) + 2,
        },
        "rows": rows,
    }
    out = results_dir / "spherical_shell_radial_bvp_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    plot_convergence(rows, results_dir)
    plot_deformed_surfaces(ref, results_dir)
    plot_radial_profiles(
        results_dir / f"spherical_shell_radial_bvp_n{n_values[-1]}_final.npz", ref, results_dir
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
