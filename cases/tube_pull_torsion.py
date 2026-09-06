#!/usr/bin/env python3
"""Finite cylindrical tube with fixed left end and pull-torsion on the right end.

The tube axis is z.  The default grid is chosen so that the outer radius has
30 lattice cells and the length has 300 lattice cells:

    dx = 0.01, Ro = 0.30, L = 3.00.

The left end z=0 is fixed.  The right end z=L is displacement-controlled with
50% axial stretch and a 45 degree finite twist, ramped over 10T and relaxed to
100T.  Inner and outer cylindrical walls are traction-free Neumann boundaries.
"""

from __future__ import annotations

import argparse
from traction_settings import apply_defaults
import json
import math
import sys
from pathlib import Path
from _paths import ROOT, RESULTS, REFERENCES, FENICSX_PYTHON, install_reference

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
NEWCODE_ROOT = ROOT / "src"
PAPER_ROOT = ROOT


from curved_lbm.three_d import curved_boundary_warp3d as cb

DATA_DIR = RESULTS / "fig10_11_tube" / "data"


BOUNDARY_ID_OUTER = cb.BOUNDARY_ID_OUTER
BOUNDARY_ID_INNER = cb.BOUNDARY_ID_INNER
BOUNDARY_ID_LEFT = 3
BOUNDARY_ID_RIGHT = 4


def make_geometry_functions(
    *, center_xy: tuple[float, float], inner_radius: float, outer_radius: float, length_z: float
):
    cx, cy = center_xy

    def rho_array(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X, Y = np.broadcast_arrays(X, Y)
        return np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    def phi(x: float, y: float, z: float) -> float:
        r = float(math.sqrt((x - cx) ** 2 + (y - cy) ** 2))
        return max(r - outer_radius, inner_radius - r, -z, z - length_z)

    def phi_vectorized(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        X, Y, Z = np.broadcast_arrays(X, Y, Z)
        r = rho_array(X, Y)
        return np.maximum.reduce((r - outer_radius, inner_radius - r, -Z, Z - length_z))

    def nearest_id(x: float, y: float, z: float) -> int:
        r = float(math.sqrt((x - cx) ** 2 + (y - cy) ** 2))
        d = np.array([abs(r - outer_radius), abs(r - inner_radius), abs(z), abs(z - length_z)])
        m = int(np.argmin(d))
        return (BOUNDARY_ID_OUTER, BOUNDARY_ID_INNER, BOUNDARY_ID_LEFT, BOUNDARY_ID_RIGHT)[m]

    def normal(x: float, y: float, z: float) -> tuple[float, float, float]:
        r = max(float(math.sqrt((x - cx) ** 2 + (y - cy) ** 2)), 1.0e-30)
        ex = (x - cx) / r
        ey = (y - cy) / r
        bid = nearest_id(x, y, z)
        if bid == BOUNDARY_ID_OUTER:
            return ex, ey, 0.0
        if bid == BOUNDARY_ID_INNER:
            return -ex, -ey, 0.0
        if bid == BOUNDARY_ID_LEFT:
            return 0.0, 0.0, -1.0
        return 0.0, 0.0, 1.0

    def normal_vectorized(
        X: np.ndarray, Y: np.ndarray, Z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, Y, Z = np.broadcast_arrays(X, Y, Z)
        r = np.maximum(rho_array(X, Y), 1.0e-30)
        ex = (X - cx) / r
        ey = (Y - cy) / r
        d_outer = np.abs(r - outer_radius)
        d_inner = np.abs(r - inner_radius)
        d_left = np.abs(Z)
        d_right = np.abs(Z - length_z)
        dmin = np.minimum.reduce((d_outer, d_inner, d_left, d_right))
        nx = np.zeros_like(X, dtype=np.float64)
        ny = np.zeros_like(X, dtype=np.float64)
        nz = np.zeros_like(X, dtype=np.float64)
        m_outer = d_outer == dmin
        m_inner = (~m_outer) & (d_inner == dmin)
        m_left = (~m_outer) & (~m_inner) & (d_left == dmin)
        m_right = (~m_outer) & (~m_inner) & (~m_left)
        nx[m_outer] = ex[m_outer]
        ny[m_outer] = ey[m_outer]
        nx[m_inner] = -ex[m_inner]
        ny[m_inner] = -ey[m_inner]
        nz[m_left] = -1.0
        nz[m_right] = 1.0
        return nx, ny, nz

    def boundary_id(x: float, y: float, z: float, nx: float, ny: float, nz: float) -> int:
        return nearest_id(x, y, z)

    return phi, phi_vectorized, normal, normal_vectorized, boundary_id


def tube_boundary(
    *,
    center_xy: tuple[float, float],
    axial_traction: float,
    twist_alpha: float,
    ramp_time: float,
) -> cb.CurvedBoundarySpec3D:
    return cb.CurvedBoundarySpec3D(
        kind="neumann",
        value_mode=cb.CURVED3D_VALUE_ZERO,
        id_kinds={
            BOUNDARY_ID_OUTER: "neumann",
            BOUNDARY_ID_INNER: "neumann",
            BOUNDARY_ID_LEFT: "dirichlet",
            BOUNDARY_ID_RIGHT: "neumann",
        },
        id_value_modes={
            BOUNDARY_ID_OUTER: cb.CURVED3D_VALUE_ZERO,
            BOUNDARY_ID_INNER: cb.CURVED3D_VALUE_ZERO,
            BOUNDARY_ID_LEFT: cb.CURVED3D_VALUE_ZERO,
            BOUNDARY_ID_RIGHT: cb.CURVED3D_VALUE_Z_PULL_TORSION_TRACTION,
        },
        id_params={
            BOUNDARY_ID_RIGHT: (
                center_xy[0],
                center_xy[1],
                axial_traction,
                twist_alpha,
                ramp_time,
                0.0,
            ),
        },
        label="fixed left end; right end axial pull plus torsion",
    )


def tube_boundary_displacement_control(
    *,
    center_xy: tuple[float, float],
    axial_displacement: float,
    twist_angle: float,
    ramp_time: float,
) -> cb.CurvedBoundarySpec3D:
    params = (center_xy[0], center_xy[1], axial_displacement, twist_angle, ramp_time)
    return cb.CurvedBoundarySpec3D(
        kind="neumann",
        value_mode=cb.CURVED3D_VALUE_ZERO,
        id_kinds={
            BOUNDARY_ID_OUTER: "neumann",
            BOUNDARY_ID_INNER: "neumann",
            BOUNDARY_ID_LEFT: "dirichlet",
            BOUNDARY_ID_RIGHT: "dirichlet",
        },
        id_value_modes={
            BOUNDARY_ID_OUTER: cb.CURVED3D_VALUE_ZERO,
            BOUNDARY_ID_INNER: cb.CURVED3D_VALUE_ZERO,
            BOUNDARY_ID_LEFT: cb.CURVED3D_VALUE_ZERO,
            BOUNDARY_ID_RIGHT: cb.CURVED3D_VALUE_Z_STRETCH_EXACT_TWIST_VELOCITY,
        },
        id_params={BOUNDARY_ID_RIGHT: params},
        label="fixed left end; right end prescribed axial strain and exact finite twist",
    )


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


def initialize_identity_active(solver: cb.WarpCurvedBoundaryLBM3D) -> tuple[np.ndarray, np.ndarray]:
    U = np.zeros((12, solver.n_active), dtype=np.float64)
    U[3] = 1.0
    U[7] = 1.0
    U[11] = 1.0
    displacement = np.zeros((3, solver.n_active), dtype=np.float64)
    return U, displacement


def summarize_loaded_end(
    *,
    center_xy: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    length_z: float,
    axial_traction: float,
    twist_alpha: float,
) -> dict[str, float]:
    area = math.pi * (outer_radius**2 - inner_radius**2)
    polar = 0.5 * math.pi * (outer_radius**4 - inner_radius**4)
    return {
        "right_end_area": area,
        "axial_force": axial_traction * area,
        "torque_z": twist_alpha * polar,
        "max_tangential_traction": abs(twist_alpha) * outer_radius,
        "length_z": length_z,
        "inner_radius": inner_radius,
        "outer_radius": outer_radius,
        "center_x": center_xy[0],
        "center_y": center_xy[1],
    }


def plot_deformed_boundary_band(
    npz_path: Path,
    *,
    center_xy: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    length_z: float,
    results_dir: Path,
    max_points: int = 50000,
) -> None:
    data = np.load(npz_path)
    X = data["x_active"]
    Y = data["y_active"]
    Z = data["z_active"]
    u = data["u"]
    dx = float(data["dx"])
    cx, cy = center_xy
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    band = (
        (np.abs(r - outer_radius) <= 1.5 * dx)
        | (np.abs(r - inner_radius) <= 1.5 * dx)
        | (Z <= 1.5 * dx)
        | (Z >= length_z - 1.5 * dx)
    )
    idx = np.flatnonzero(band)
    if idx.size > max_points:
        rng = np.random.default_rng(20260527)
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))
    xd = X[idx] + u[0, idx]
    yd = Y[idx] + u[1, idx]
    zd = Z[idx] + u[2, idx]
    umag = np.sqrt(np.sum(u[:, idx] * u[:, idx], axis=0))

    fig = plt.figure(figsize=(9.0, 7.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xd, yd, zd, c=umag, s=1.2, cmap="magma", alpha=0.72, linewidths=0.0)
    fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.04, label="|u|")
    ax.set_xlim(cx - 1.15 * outer_radius, cx + 1.15 * outer_radius)
    ax.set_ylim(cy - 1.15 * outer_radius, cy + 1.15 * outer_radius)
    ax.set_zlim(0.0, length_z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Deformed tube boundary-band nodes")
    ax.view_init(elev=18.0, azim=-58.0)
    fig.savefig(results_dir / "tube_pull_torsion_deformed_boundary_band.png", dpi=220)
    plt.close(fig)


def plot_axis_profiles(
    npz_path: Path,
    *,
    center_xy: tuple[float, float],
    length_z: float,
    results_dir: Path,
) -> None:
    data = np.load(npz_path)
    X = data["x_active"]
    Y = data["y_active"]
    Z = data["z_active"]
    u = data["u"]
    U = data["U"]
    F = data["F"]
    cx, cy = center_xy
    dx0 = X - cx
    dy0 = Y - cy
    r2 = np.maximum(dx0 * dx0 + dy0 * dy0, 1.0e-30)
    utheta_over_r = (-dy0 * u[0] + dx0 * u[1]) / r2
    uz = u[2]
    vmag = np.sqrt(U[0] * U[0] + U[1] * U[1] + U[2] * U[2])
    J = (
        F[0, 0] * (F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1])
        - F[0, 1] * (F[1, 0] * F[2, 2] - F[1, 2] * F[2, 0])
        + F[0, 2] * (F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0])
    )
    bins = np.linspace(0.0, length_z, 80)
    centers = 0.5 * (bins[:-1] + bins[1:])
    which = np.digitize(Z, bins) - 1

    def bmean(vals: np.ndarray) -> np.ndarray:
        out = np.full(centers.shape, np.nan)
        for i in range(centers.size):
            m = which == i
            if np.any(m):
                out[i] = float(np.mean(vals[m]))
        return out

    profiles = {
        "z": centers,
        "mean_utheta_over_r": bmean(utheta_over_r),
        "mean_uz": bmean(uz),
        "mean_J": bmean(J),
        "mean_vmag": bmean(vmag),
    }
    names = tuple(profiles)
    np.savetxt(
        results_dir / "tube_pull_torsion_axis_profiles.csv",
        np.column_stack([profiles[name] for name in names]),
        delimiter=",",
        header=",".join(names),
        comments="",
    )

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True)
    axs[0, 0].plot(centers, profiles["mean_uz"], lw=1.8)
    axs[0, 0].set_ylabel("mean u_z")
    axs[0, 1].plot(centers, profiles["mean_utheta_over_r"], lw=1.8)
    axs[0, 1].set_ylabel("mean u_theta / r")
    axs[1, 0].plot(centers, profiles["mean_J"], lw=1.8)
    axs[1, 0].set_ylabel("mean J")
    axs[1, 1].semilogy(centers, profiles["mean_vmag"], lw=1.8)
    axs[1, 1].set_ylabel("mean |v|")
    for ax in axs.ravel():
        ax.grid(True, ls=":", alpha=0.5)
        ax.set_xlabel("z")
    fig.suptitle("Tube pull-torsion axial profiles")
    fig.savefig(results_dir / "tube_pull_torsion_axis_profiles.png", dpi=200)
    plt.close(fig)


def run_case(args: argparse.Namespace) -> dict[str, float | int | bool | str | dict]:
    dx = float(args.outer_radius) / float(args.radius_nodes)
    nx = 2 * (int(args.radius_nodes) + int(args.margin_nodes))
    ny = nx
    nz = int(args.z_nodes)
    length_x = nx * dx
    length_y = ny * dx
    length_z = nz * dx
    center_xy = (0.5 * length_x, 0.5 * length_y)
    inner_radius = float(args.inner_radius)
    if inner_radius < 0.0:
        inner_radius = float(args.inner_ratio) * float(args.outer_radius)
    phi, phi_vectorized, normal, normal_vectorized, boundary_id = make_geometry_functions(
        center_xy=center_xy,
        inner_radius=inner_radius,
        outer_radius=float(args.outer_radius),
        length_z=length_z,
    )
    geom = cb.CurvedBoundaryGeometry3D.from_level_set(
        nx=nx,
        ny=ny,
        nz=nz,
        length_x=length_x,
        length_y=length_y,
        length_z=length_z,
        phi=phi,
        normal=normal,
        boundary_id=boundary_id,
        phi_vectorized=phi_vectorized,
        normal_vectorized=normal_vectorized,
        label="finite_cylindrical_tube_pull_torsion",
    )
    material = cb.WarpHyperelasticMaterial3D.from_poisson(
        "neo_hooke", float(args.poisson), mu=float(args.mu)
    )
    omega = effective_omega(float(args.omega), dx, str(args.omega_scaling), float(args.omega_kappa))
    if str(args.load_mode) == "strain":
        axial_displacement = float(args.target_strain) * length_z
        axial_velocity = axial_displacement / float(args.ramp_time)
        angular_velocity = float(args.target_twist_angle) / float(args.ramp_time)
        boundary = tube_boundary_displacement_control(
            center_xy=center_xy,
            axial_displacement=axial_displacement,
            twist_angle=float(args.target_twist_angle),
            ramp_time=float(args.ramp_time),
        )
    else:
        axial_velocity = 0.0
        angular_velocity = 0.0
        boundary = tube_boundary(
            center_xy=center_xy,
            axial_traction=float(args.axial_traction),
            twist_alpha=float(args.twist_alpha),
            ramp_time=float(args.ramp_time),
        )
    solver = cb.WarpCurvedBoundaryLBM3D(
        geom,
        material=material,
        boundary=boundary,
        lattice_speed=float(args.lattice_speed),
        collision_omega=omega,
        boundary_reconstruction=str(args.boundary_reconstruction),
        source_mode="damping" if float(args.damping_gamma) > 0.0 else "none",
        damping_gamma=float(args.damping_gamma),
        device=str(args.device),
        check_linearized_cfl=True,
        local_displacement_interval=int(args.local_displacement_interval),
        local_displacement_sweeps=int(args.local_displacement_sweeps),
        local_displacement_relax=float(args.local_displacement_relax),
        local_displacement_boundary_weight=float(args.local_displacement_boundary_weight),
        local_displacement_boundary_id=-1,
        local_compatibility_interval=int(args.local_compatibility_interval),
        local_compatibility_blend=float(args.local_compatibility_blend),
    )
    U0, displacement0 = initialize_identity_active(solver)
    solver.initialize_from_numpy(U=U0, displacement=displacement0, init_order=1)
    target_time = float(args.run_time)
    progress_every = int(args.progress_every)
    history = []
    if progress_every > 0:
        while solver.time < target_time - 0.5 * solver.dt:
            chunk_end = min(target_time, solver.time + progress_every * solver.dt)
            solver.run_until(chunk_end)
            solver.refresh_current()
            U_check = solver.active_system_state()
            u_check = solver.active_displacement()
            J_check = solver.active_deformation_jacobian()
            finite = bool(
                np.isfinite(U_check).all()
                and np.isfinite(u_check).all()
                and np.isfinite(J_check).all()
                and np.all(J_check > 0.0)
            )
            state = {
                "step": solver.steps,
                "time": solver.time,
                "finite": finite,
                "min_J": float(np.min(J_check)),
                "max_J": float(np.max(J_check)),
                "rms_v": float(np.sqrt(np.mean(np.sum(U_check[:3] ** 2, axis=0)))),
            }
            history.append(state)
            print(json.dumps(state), flush=True)
            if not finite:
                raise RuntimeError("Tube state is non-finite or has a non-positive Jacobian")
    else:
        solver.run_until(target_time)
    U = solver.active_system_state()
    u = solver.active_displacement()
    J = solver.active_deformation_jacobian()
    ids, counts = np.unique(geom.link_boundary_id, return_counts=True)

    def count_bid(bid: int) -> int:
        return int(counts[np.where(ids == bid)[0][0]]) if np.any(ids == bid) else 0

    row: dict[str, float | int | bool | str | dict] = {
        "case": "finite_cylindrical_tube_pull_torsion",
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "radius_nodes": int(args.radius_nodes),
        "z_nodes": int(args.z_nodes),
        "dx": float(dx),
        "active_nodes": int(solver.n_active),
        "cut_links": int(geom.n_links),
        "outer_cut_links": count_bid(BOUNDARY_ID_OUTER),
        "inner_cut_links": count_bid(BOUNDARY_ID_INNER),
        "left_cut_links": count_bid(BOUNDARY_ID_LEFT),
        "right_cut_links": count_bid(BOUNDARY_ID_RIGHT),
        "steps": int(solver.steps),
        "time": float(solver.time),
        "lattice_speed": float(args.lattice_speed),
        "collision_omega": float(omega),
        "damping_gamma": float(args.damping_gamma),
        "boundary_reconstruction": solver.boundary_reconstruction_name,
        "compatibility_projection_active": bool(
            solver._has_neumann_cut_links and solver.compatibility_projection_interval > 0
        ),
        "load_mode": str(args.load_mode),
        "axial_traction": float(args.axial_traction),
        "twist_alpha": float(args.twist_alpha),
        "target_strain": float(args.target_strain),
        "target_axial_displacement": float(args.target_strain) * length_z,
        "target_twist_angle": float(args.target_twist_angle),
        "average_axial_velocity_during_ramp": float(axial_velocity),
        "average_angular_velocity_during_ramp": float(angular_velocity),
        "ramp_profile": "smooth_cosine_displacement_control"
        if str(args.load_mode) == "strain"
        else "sine2_traction_hold",
        "ramp_time": float(args.ramp_time),
        "run_time": target_time,
        "history": history,
        "max_abs_u": float(np.max(np.sqrt(np.sum(u * u, axis=0)))),
        "max_abs_v": float(np.max(np.sqrt(U[0] * U[0] + U[1] * U[1] + U[2] * U[2]))),
        "min_J": float(np.min(J)),
        "max_J": float(np.max(J)),
        "finite": bool(not solver.invalid_state()),
        "load_resultants": summarize_loaded_end(
            center_xy=center_xy,
            inner_radius=inner_radius,
            outer_radius=float(args.outer_radius),
            length_z=length_z,
            axial_traction=float(args.axial_traction),
            twist_alpha=float(args.twist_alpha),
        ),
        "solver_summary": solver.summary(),
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    state_path = (
        results_dir / f"tube_pull_torsion_r{int(args.radius_nodes)}_z{int(args.z_nodes)}_final.npz"
    )
    solver.save_npz(state_path)
    summary = {
        "case": "finite_cylindrical_tube_pull_torsion",
        "material": {
            "model": material.model,
            "lambda": material.lam,
            "mu": material.mu,
            "poisson": float(args.poisson),
        },
        "boundary_ids": {
            "outer": BOUNDARY_ID_OUTER,
            "inner": BOUNDARY_ID_INNER,
            "left": BOUNDARY_ID_LEFT,
            "right": BOUNDARY_ID_RIGHT,
        },
        "row": row,
    }
    summary_path = results_dir / "tube_pull_torsion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    plot_deformed_boundary_band(
        state_path,
        center_xy=center_xy,
        inner_radius=inner_radius,
        outer_radius=float(args.outer_radius),
        length_z=length_z,
        results_dir=results_dir,
    )
    plot_axis_profiles(state_path, center_xy=center_xy, length_z=length_z, results_dir=results_dir)
    if not bool(args.skip_surface_render):
        try:
            from render_tube_comsol_style import render as render_comsol_style

            angle_deg = math.degrees(float(args.target_twist_angle))
            render_comsol_style(
                argparse.Namespace(
                    state=str(state_path),
                    summary=str(summary_path),
                    out=str(results_dir / "tube_pull_torsion_comsol_style_surface.png"),
                    color_by="|u|",
                    cmap="turbo",
                    warp_scale=1.0,
                    theta_points=260,
                    z_points=340,
                    radial_points=42,
                    cutaway_degrees=70.0,
                    cutaway_center_degrees=-45.0,
                    view_angle_degrees=-18.0,
                    camera_radius=5.4,
                    camera_axis_offset=-0.10,
                    camera_zoom=0.82,
                    projection="perspective",
                    perspective_angle=28.0,
                    interp_neighbors=16,
                    orientation="horizontal",
                    width=1900,
                    height=1100,
                    screenshot_scale=2,
                    title_font_size=24,
                    scalar_title_font_size=34,
                    scalar_label_font_size=30,
                    scalar_title_x_frac=0.915,
                    scalar_title_y_frac=0.885,
                    legend_min_zero=True,
                    title=f"{float(args.target_strain):.0%} axial stretch + {angle_deg:.1f} deg twist, t={float(args.run_time):g}",
                )
            )
        except Exception as exc:
            print(f"warning: skipped smooth surface render: {exc}")
    print(json.dumps(row, indent=2))
    print(f"wrote {summary_path}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--radius-nodes", type=int, default=30)
    parser.add_argument("--z-nodes", type=int, default=300)
    parser.add_argument("--margin-nodes", type=int, default=8)
    parser.add_argument("--outer-radius", type=float, default=0.30)
    parser.add_argument(
        "--inner-radius", type=float, default=-1.0, help="negative value uses --inner-ratio"
    )
    parser.add_argument("--inner-ratio", type=float, default=0.50)
    parser.add_argument("--axial-traction", type=float, default=0.06)
    parser.add_argument("--twist-alpha", type=float, default=0.12)
    parser.add_argument("--load-mode", choices=("traction", "strain"), default="strain")
    parser.add_argument("--target-strain", type=float, default=0.50)
    parser.add_argument("--target-twist-angle", type=float, default=math.pi / 4.0)
    parser.add_argument("--ramp-time", type=float, default=10.0)
    parser.add_argument("--run-time", type=float, default=100.0)
    parser.add_argument("--lattice-speed", type=float, default=10.0)
    parser.add_argument("--omega", type=float, default=1.8)
    parser.add_argument("--omega-scaling", choices=("constant", "dx"), default="constant")
    parser.add_argument("--omega-kappa", type=float, default=50.0)
    parser.add_argument("--damping-gamma", type=float, default=10.0)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--poisson", type=float, default=0.20)
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
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="GPU steps per synchronization-only progress update",
    )
    parser.add_argument("--results-dir", default=str(DATA_DIR))
    parser.add_argument("--skip-surface-render", action="store_true")
    apply_defaults(parser)
    args = parser.parse_args()
    run_case(args)


if __name__ == "__main__":
    main()
