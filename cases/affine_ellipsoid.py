#!/usr/bin/env python3
"""Curved 3-D Dirichlet benchmark on a rotated ellipsoid.

The exact solution is an affine velocity field on an arbitrary curved domain:

    v(X,t) = A (X-C)
    u(X,t) = t A (X-C)
    F(t)   = I + t A

Since F is spatially uniform, P(F) is uniform and div(P)=0.  The test therefore
isolates the curved Dirichlet boundary reconstruction on a non-grid-aligned
surface.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from _paths import ROOT, RESULTS, REFERENCES, FENICSX_PYTHON, install_reference
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
NEWCODE_ROOT = ROOT / "src"
PAPER_ROOT = ROOT


from curved_lbm.three_d import curved_boundary_warp3d as cb


CENTER = np.array([0.5, 0.5, 0.5], dtype=np.float64)
AXES = np.array([0.34, 0.25, 0.18], dtype=np.float64)
A_MATRIX = np.array(
    [
        [0.00, 0.80, 0.00],
        [0.00, 0.00, -0.60],
        [0.45, 0.00, 0.08],
    ],
    dtype=np.float64,
)


def rotation_matrix() -> np.ndarray:
    rz = math.radians(25.0)
    ry = math.radians(18.0)
    Rz = np.array(
        [
            [math.cos(rz), -math.sin(rz), 0.0],
            [math.sin(rz), math.cos(rz), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    Ry = np.array(
        [
            [math.cos(ry), 0.0, math.sin(ry)],
            [0.0, 1.0, 0.0],
            [-math.sin(ry), 0.0, math.cos(ry)],
        ],
        dtype=np.float64,
    )
    return Rz @ Ry


ROT = rotation_matrix()


def ellipsoid_phi(x: float, y: float, z: float) -> float:
    p = np.array([x, y, z], dtype=np.float64) - CENTER
    xi = ROT.T @ p
    return float(np.sqrt(np.sum((xi / AXES) ** 2)) - 1.0)


def ellipsoid_phi_array(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    X, Y, Z = np.broadcast_arrays(X, Y, Z)
    dX = np.stack([X - CENTER[0], Y - CENTER[1], Z - CENTER[2]], axis=0)
    xi = np.einsum("ab,b...->a...", ROT.T, dX)
    return np.sqrt(np.sum((xi / AXES[(slice(None),) + (None,) * X.ndim]) ** 2, axis=0)) - 1.0


def ellipsoid_normal(x: float, y: float, z: float) -> tuple[float, float, float]:
    p = np.array([x, y, z], dtype=np.float64) - CENTER
    xi = ROT.T @ p
    s = max(float(np.sqrt(np.sum((xi / AXES) ** 2))), 1.0e-30)
    grad_local = xi / (AXES * AXES * s)
    grad = ROT @ grad_local
    grad /= max(float(np.linalg.norm(grad)), 1.0e-30)
    return float(grad[0]), float(grad[1]), float(grad[2])


def ellipsoid_normal_array(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Y, Z = np.broadcast_arrays(X, Y, Z)
    dX = np.stack([X - CENTER[0], Y - CENTER[1], Z - CENTER[2]], axis=0)
    xi = np.einsum("ab,b...->a...", ROT.T, dX)
    scale = np.maximum(
        np.sqrt(np.sum((xi / AXES[(slice(None),) + (None,) * X.ndim]) ** 2, axis=0)), 1.0e-30
    )
    grad_local = xi / (AXES[(slice(None),) + (None,) * X.ndim] ** 2 * scale[None, ...])
    grad = np.einsum("ab,b...->a...", ROT, grad_local)
    norm = np.maximum(np.sqrt(np.sum(grad * grad, axis=0)), 1.0e-30)
    grad = grad / norm[None, ...]
    return grad[0], grad[1], grad[2]


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


def affine_velocity_field(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    X, Y, Z = np.broadcast_arrays(X, Y, Z)
    dX = np.stack([X - CENTER[0], Y - CENTER[1], Z - CENTER[2]], axis=0)
    return np.einsum("ab,b...->a...", A_MATRIX, dX)


def exact_fields(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, time: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = affine_velocity_field(X, Y, Z)
    u = float(time) * v
    F0 = np.eye(3, dtype=np.float64) + float(time) * A_MATRIX
    suffix = (None,) * X.ndim
    F = np.broadcast_to(F0[(slice(None), slice(None)) + suffix], (3, 3) + X.shape).copy()
    return v, u, F


def initial_state(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v0 = affine_velocity_field(X, Y, Z)
    U = np.zeros((12,) + X.shape, dtype=np.float64)
    U[0:3] = v0
    U[3] = 1.0
    U[7] = 1.0
    U[11] = 1.0
    displacement = np.zeros((3,) + X.shape, dtype=np.float64)
    return U, displacement


def max_surface_affine_speed(n_theta: int = 200, n_phi: int = 400) -> float:
    theta = np.linspace(0.0, np.pi, int(n_theta))
    phi = np.linspace(0.0, 2.0 * np.pi, int(n_phi))
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    local = np.stack(
        [
            AXES[0] * np.sin(TH) * np.cos(PH),
            AXES[1] * np.sin(TH) * np.sin(PH),
            AXES[2] * np.cos(TH),
        ],
        axis=0,
    )
    X0 = CENTER[:, None, None] + np.einsum("ab,bmn->amn", ROT, local)
    v = np.einsum("ab,bmn->amn", A_MATRIX, X0 - CENTER[:, None, None])
    return float(np.max(np.sqrt(np.sum(v * v, axis=0))))


def relative_l2(error: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    e = np.asarray(error)
    r = np.asarray(reference)
    m = np.asarray(mask, dtype=bool)
    return float(
        np.sqrt(np.sum(e[..., m] ** 2)) / max(np.sqrt(np.sum(r[..., m] ** 2)), np.finfo(float).eps)
    )


def add_orders(rows: list[dict[str, float | int | bool | str]]) -> None:
    for i in range(1, len(rows)):
        h0 = float(rows[i - 1]["dx"])
        h1 = float(rows[i]["dx"])
        for key in ("rel_l2_u", "rel_l2_v", "rel_l2_F"):
            e0 = float(rows[i - 1][key])
            e1 = float(rows[i][key])
            rows[i][f"order_{key}"] = (
                float(math.log(e1 / e0) / math.log(h1 / h0))
                if e0 > 0.0 and e1 > 0.0
                else float("nan")
            )


def run_one(
    n: int,
    *,
    device: str,
    run_time: float,
    lattice_speed: float,
    collision_omega: float,
    init_order: int,
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
        phi=ellipsoid_phi,
        normal=ellipsoid_normal,
        phi_vectorized=ellipsoid_phi_array,
        normal_vectorized=ellipsoid_normal_array,
        label="rotated_ellipsoid_affine_dirichlet",
    )
    params = tuple(
        A_MATRIX.reshape(-1).tolist() + (-A_MATRIX @ CENTER).tolist() + [0.0, 0.0, 0.0, 0.0]
    )
    boundary = cb.CurvedBoundarySpec3D(
        kind="dirichlet",
        value_mode=cb.CURVED3D_VALUE_AFFINE_VELOCITY,
        params=params,
        label="affine velocity v=A(X-C)",
    )
    material = cb.WarpHyperelasticMaterial3D.from_poisson("neo_hooke", 0.20, mu=1.0)
    solver = cb.WarpCurvedBoundaryLBM3D(
        geom,
        material=material,
        boundary=boundary,
        lattice_speed=lattice_speed,
        collision_omega=collision_omega,
        boundary_reconstruction=boundary_reconstruction,
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
    U0, displacement0 = initial_state(X, Y, Z)
    solver.initialize_from_numpy(U=U0, displacement=displacement0, init_order=init_order)
    solver.run_until(run_time)

    active = np.ones_like(X, dtype=bool)
    U_num = solver.active_system_state()
    u_num = solver.active_displacement()
    F_num = solver.active_deformation_gradient()
    v_ref, u_ref, F_ref = exact_fields(X, Y, Z, solver.time)
    v_err = U_num[:3] - v_ref
    u_err = u_num - u_ref
    F_err = F_num - F_ref
    J = solver.active_deformation_jacobian()
    u_num_mag = np.sqrt(np.sum(u_num * u_num, axis=0))
    u_ref_mag = np.sqrt(np.sum(u_ref * u_ref, axis=0))
    row: dict[str, float | int | bool | str] = {
        "case": "rotated_ellipsoid_affine_dirichlet",
        "n": int(n),
        "dx": float(geom.dx),
        "active_nodes": int(solver.n_active),
        "cut_links": int(geom.n_links),
        "steps": int(solver.steps),
        "time": float(solver.time),
        "lattice_speed": float(lattice_speed),
        "collision_omega": float(collision_omega),
        "boundary_reconstruction": solver.boundary_reconstruction_name,
        "compatibility_projection_active": bool(
            solver._has_neumann_cut_links and solver.compatibility_projection_interval > 0
        ),
        "init_order": int(init_order),
        "rel_l2_u": relative_l2(u_err, u_ref, active),
        "rel_l2_v": relative_l2(v_err, v_ref, active),
        "rel_l2_F": relative_l2(F_err, F_ref, active),
        "max_abs_u": float(np.max(np.abs(u_err))),
        "max_abs_v": float(np.max(np.abs(v_err))),
        "max_abs_F": float(np.max(np.abs(F_err))),
        "max_u_num": float(np.max(u_num_mag)),
        "max_u_ref": float(np.max(u_ref_mag)),
        "min_J": float(np.min(J)),
        "max_J": float(np.max(J)),
        "finite": bool(not solver.invalid_state()),
    }
    if save_state:
        solver.save_npz(results_dir / f"ellipsoid_affine_dirichlet_n{n}_final.npz")
        np.savez_compressed(
            results_dir / f"ellipsoid_affine_dirichlet_reference_n{n}.npz",
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
            v=v_ref,
            F=F_ref,
        )
    print(json.dumps(row, indent=2))
    return row


def plot_convergence(rows: list[dict[str, float | int | bool | str]], results_dir: Path) -> None:
    dx = np.array([float(r["dx"]) for r in rows])
    n = np.array([int(r["n"]) for r in rows])
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for key, label, color in (
        ("rel_l2_u", "u", "tab:blue"),
        ("rel_l2_v", "v", "tab:orange"),
        ("rel_l2_F", "F", "tab:green"),
    ):
        err = np.array([float(r[key]) for r in rows])
        ax0.loglog(dx, err, marker="o", lw=1.8, label=label, color=color)
        order = np.array([float(r.get(f"order_{key}", np.nan)) for r in rows])
        ax1.plot(n[1:], order[1:], marker="o", lw=1.8, label=label, color=color)
    ax0.invert_xaxis()
    ax0.grid(True, which="both", ls=":", alpha=0.5)
    ax0.set_xlabel("dx")
    ax0.set_ylabel("relative L2 error")
    ax0.set_title("Rotated ellipsoid affine Dirichlet")
    ax0.legend()
    ax1.axhline(2.0, color="0.35", ls="--", lw=1.1)
    ax1.grid(True, ls=":", alpha=0.5)
    ax1.set_xlabel("grid n")
    ax1.set_ylabel("observed order")
    ax1.set_title("Pairwise observed order")
    ax1.legend()
    fig.savefig(results_dir / "ellipsoid_affine_dirichlet_convergence.png", dpi=190)
    plt.close(fig)


def plot_surface(results_dir: Path, time: float) -> None:
    theta = np.linspace(0.0, np.pi, 80)
    phi = np.linspace(0.0, 2.0 * np.pi, 160)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    local = np.stack(
        [
            AXES[0] * np.sin(TH) * np.cos(PH),
            AXES[1] * np.sin(TH) * np.sin(PH),
            AXES[2] * np.cos(TH),
        ],
        axis=0,
    )
    X0 = CENTER[:, None, None] + np.einsum("ab,bmn->amn", ROT, local)
    disp = float(time) * np.einsum("ab,bmn->amn", A_MATRIX, X0 - CENTER[:, None, None])
    Xd = X0 + disp
    umag = np.sqrt(np.sum(disp * disp, axis=0))

    fig = plt.figure(figsize=(8.0, 6.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(
        X0[0], X0[1], X0[2], rstride=8, cstride=12, color="0.55", linewidth=0.55, alpha=0.55
    )
    norm = plt.Normalize(float(np.min(umag)), float(np.max(umag)))
    colors = plt.cm.magma(norm(umag))
    ax.plot_surface(
        Xd[0],
        Xd[1],
        Xd[2],
        facecolors=colors,
        linewidth=0.0,
        antialiased=True,
        shade=False,
        alpha=0.92,
    )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap="magma")
    mappable.set_array(umag)
    fig.colorbar(mappable, ax=ax, shrink=0.75, pad=0.04, label="|u|")
    all_pts = np.concatenate([X0.reshape(3, -1), Xd.reshape(3, -1)], axis=1)
    lo = np.min(all_pts, axis=1)
    hi = np.max(all_pts, axis=1)
    mid = 0.5 * (lo + hi)
    rad = 0.55 * float(np.max(hi - lo))
    ax.set_xlim(mid[0] - rad, mid[0] + rad)
    ax.set_ylim(mid[1] - rad, mid[1] + rad)
    ax.set_zlim(mid[2] - rad, mid[2] + rad)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Initial wireframe and exact deformed ellipsoid surface")
    ax.view_init(elev=22.0, azim=-52.0)
    fig.savefig(results_dir / "ellipsoid_affine_dirichlet_deformed_surface.png", dpi=220)
    plt.close(fig)


def plot_numeric_boundary_band(results_dir: Path, n: int, *, max_points: int = 35000) -> None:
    state_path = results_dir / f"ellipsoid_affine_dirichlet_n{n}_final.npz"
    if not state_path.exists():
        return
    data = np.load(state_path)
    storage = str(data["storage"]) if "storage" in data.files else "dense"
    u = data["u"]
    time = float(data["time"])
    if storage == "sparse_active" and "x_active" in data.files:
        X = data["x_active"]
        Y = data["y_active"]
        Z = data["z_active"]
        dx = float(data["dx"])
        phi_grid = ellipsoid_phi_array(X, Y, Z)
        band = np.abs(phi_grid) <= 2.0 * dx
        idx = np.flatnonzero(band)
    else:
        X = data["x"]
        Y = data["y"]
        Z = data["z"]
        active = data["active"].astype(bool)
        dx = float(np.mean(np.diff(X[:, 0, 0])))
        phi_grid = ellipsoid_phi_array(X, Y, Z)
        band = active & (np.abs(phi_grid) <= 2.0 * dx)
        idx = np.flatnonzero(band.ravel())
    if idx.size == 0:
        return
    if idx.size > max_points:
        rng = np.random.default_rng(20260527)
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))

    xd = (X + u[0]).ravel()[idx]
    yd = (Y + u[1]).ravel()[idx]
    zd = (Z + u[2]).ravel()[idx]
    umag = np.sqrt(np.sum(u * u, axis=0)).ravel()[idx]

    theta = np.linspace(0.0, np.pi, 36)
    phi = np.linspace(0.0, 2.0 * np.pi, 72)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    local = np.stack(
        [
            AXES[0] * np.sin(TH) * np.cos(PH),
            AXES[1] * np.sin(TH) * np.sin(PH),
            AXES[2] * np.cos(TH),
        ],
        axis=0,
    )
    X0 = CENTER[:, None, None] + np.einsum("ab,bmn->amn", ROT, local)
    Xref = X0 + time * np.einsum("ab,bmn->amn", A_MATRIX, X0 - CENTER[:, None, None])

    fig = plt.figure(figsize=(8.0, 6.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xd, yd, zd, c=umag, s=2.0, cmap="magma", alpha=0.78, linewidths=0.0)
    ax.plot_wireframe(
        Xref[0], Xref[1], Xref[2], rstride=4, cstride=6, color="0.15", linewidth=0.45, alpha=0.42
    )
    fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.04, label="|u_num|")
    pts = np.stack([xd, yd, zd], axis=0)
    lo = np.min(pts, axis=1)
    hi = np.max(pts, axis=1)
    mid = 0.5 * (lo + hi)
    rad = 0.55 * float(np.max(hi - lo))
    ax.set_xlim(mid[0] - rad, mid[0] + rad)
    ax.set_ylim(mid[1] - rad, mid[1] + rad)
    ax.set_zlim(mid[2] - rad, mid[2] + rad)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Numerical deformed near-boundary nodes, n={n}")
    ax.view_init(elev=22.0, azim=-52.0)
    fig.savefig(results_dir / "ellipsoid_affine_dirichlet_numerical_boundary_band.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n", default="80,120,160,240,320")
    parser.add_argument("--run-time", type=float, default=None)
    parser.add_argument("--target-max-u", type=float, default=0.10)
    parser.add_argument("--lattice-speed", type=float, default=10.0)
    parser.add_argument("--omega", type=float, default=1.8)
    parser.add_argument("--omega-scaling", choices=("constant", "dx"), default="constant")
    parser.add_argument("--omega-kappa", type=float, default=16.0)
    parser.add_argument("--init-order", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--boundary-reconstruction",
        choices=("bfl", "halfway", "compat_bfl", "local_f"),
        default="local_f",
    )
    parser.add_argument("--local-displacement-interval", type=int, default=0)
    parser.add_argument("--local-displacement-sweeps", type=int, default=1)
    parser.add_argument("--local-displacement-relax", type=float, default=0.85)
    parser.add_argument("--local-displacement-boundary-weight", type=float, default=20.0)
    parser.add_argument("--local-compatibility-interval", type=int, default=0)
    parser.add_argument("--local-compatibility-blend", type=float, default=0.1)
    parser.add_argument(
        "--results-dir", default=str(RESULTS / "fig6_ellipsoid_dirichlet_3d" / "data")
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "ellipsoid_affine_dirichlet_n*_final.npz",
        "ellipsoid_affine_dirichlet_reference_n*.npz",
    ):
        for old_path in results_dir.glob(pattern):
            old_path.unlink()
    if args.run_time is None:
        max_speed = max_surface_affine_speed()
        run_time = float(args.target_max_u) / max(max_speed, np.finfo(float).eps)
    else:
        max_speed = max_surface_affine_speed()
        run_time = float(args.run_time)
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
                run_time=run_time,
                lattice_speed=float(args.lattice_speed),
                collision_omega=omega_n,
                init_order=int(args.init_order),
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
    add_orders(rows)
    summary = {
        "case": "rotated_ellipsoid_affine_dirichlet",
        "geometry": {
            "center": CENTER.tolist(),
            "axes": AXES.tolist(),
            "rotation": "Rz(25deg) @ Ry(18deg)",
        },
        "A": A_MATRIX.tolist(),
        "run_time_requested": float(run_time),
        "target_max_u": None if args.run_time is not None else float(args.target_max_u),
        "max_surface_affine_speed": float(max_speed),
        "lattice_speed": float(args.lattice_speed),
        "omega_input": float(args.omega),
        "omega_scaling": str(args.omega_scaling),
        "omega_kappa": float(args.omega_kappa),
        "init_order": int(args.init_order),
        "rows": rows,
    }
    (results_dir / "ellipsoid_affine_dirichlet_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    plot_convergence(rows, results_dir)
    plot_surface(results_dir, float(rows[-1]["time"]))
    plot_numeric_boundary_band(results_dir, int(rows[-1]["n"]))
    print(f"wrote {results_dir / 'ellipsoid_affine_dirichlet_summary.json'}")


if __name__ == "__main__":
    main()
