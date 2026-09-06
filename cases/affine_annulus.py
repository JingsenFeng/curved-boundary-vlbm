#!/usr/bin/env python3
"""Affine Dirichlet annulus convergence against an exact solution.

Both curved interfaces prescribe the same affine velocity. The exact
deformation gradient is spatially uniform. The default compat_bfl
option applies the Dirichlet population-pair relation and BFL
interpolation without the traction-specific kinematic corrections.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from _paths import ROOT, RESULTS, REFERENCES, FENICSX_PYTHON, install_reference
import sys
import time
from typing import Any

import numpy as np


SCRIPT = Path(__file__).resolve()
RECON_ROOT = ROOT / "src"
PACKAGE_ROOT = ROOT
NEWFIG_ROOT = RESULTS
DEFAULT_OUT_DIR = NEWFIG_ROOT / "fig1_dirichlet_annulus" / "data"


from curved_lbm.two_d.curved_boundary_warp import (  # noqa: E402
    BOUNDARY_ID_INNER,
    BOUNDARY_ID_OUTER,
    CURVED_VALUE_AFFINE_VELOCITY,
    CurvedBoundaryGeometry,
    CurvedBoundarySpec,
    WarpCurvedBoundaryLBM2D,
    relative_l2,
)
from curved_lbm.two_d.vector_nonlinear_elastic_warp import WarpHyperelasticMaterial, _first_piola_np  # noqa: E402


CENTER = (0.5, 0.5)
INNER_RADIUS = 0.18
OUTER_RADIUS = 0.42
T_END = 0.20
A_FINAL = np.array([[0.35, 0.18], [-0.10, 0.24]], dtype=np.float64)


def effective_relaxation_omega(*, base_omega: float, dx: float, mode: str, kappa: float) -> float:
    """Return the BGK relaxation parameter for this grid level.

    Under acoustic scaling, a fixed ``omega < 2`` leaves an O(dx) kinetic
    viscosity term.  The asymptotically second-order damped sequence is
    ``omega(dx) = 2 - kappa * dx``.
    """

    if mode == "constant":
        omega = float(base_omega)
    elif mode == "dx":
        omega = 2.0 - float(kappa) * float(dx)
    else:
        raise ValueError(f"unsupported omega scaling mode: {mode}")
    if not (0.0 < omega <= 2.0):
        raise ValueError(f"effective omega={omega} outside (0,2]; adjust --omega-kappa or --omega")
    return omega


def affine_velocity_params() -> tuple[float, ...]:
    cx, cy = CENTER
    B = A_FINAL / T_END
    return (
        float(B[0, 0]),
        float(B[0, 1]),
        float(-(B[0, 0] * cx + B[0, 1] * cy)),
        float(B[1, 0]),
        float(B[1, 1]),
        float(-(B[1, 0] * cx + B[1, 1] * cy)),
        0.0,
        0.0,
    )


def annulus_boundary() -> CurvedBoundarySpec:
    params = affine_velocity_params()
    return CurvedBoundarySpec(
        kind="dirichlet",
        value_mode=CURVED_VALUE_AFFINE_VELOCITY,
        params=params,
        id_kinds={BOUNDARY_ID_OUTER: "dirichlet", BOUNDARY_ID_INNER: "dirichlet"},
        id_value_modes={
            BOUNDARY_ID_OUTER: CURVED_VALUE_AFFINE_VELOCITY,
            BOUNDARY_ID_INNER: CURVED_VALUE_AFFINE_VELOCITY,
        },
        id_params={BOUNDARY_ID_OUTER: params, BOUNDARY_ID_INNER: params},
        label="affine Dirichlet velocity on annulus inner and outer boundaries",
    )


def reference_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    dx = 1.0 / float(n)
    x = (np.arange(n, dtype=np.float64) + 0.5) * dx
    return np.meshgrid(x, x, indexing="ij")


def exact_fields(
    X: np.ndarray, Y: np.ndarray, t: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cx, cy = CENTER
    dx = X - cx
    dy = Y - cy
    tau = float(t) / T_END

    u = np.zeros((2,) + X.shape, dtype=np.float64)
    u[0] = tau * (A_FINAL[0, 0] * dx + A_FINAL[0, 1] * dy)
    u[1] = tau * (A_FINAL[1, 0] * dx + A_FINAL[1, 1] * dy)

    U = np.zeros((6,) + X.shape, dtype=np.float64)
    U[0] = (A_FINAL[0, 0] * dx + A_FINAL[0, 1] * dy) / T_END
    U[1] = (A_FINAL[1, 0] * dx + A_FINAL[1, 1] * dy) / T_END
    F = np.eye(2, dtype=np.float64) + tau * A_FINAL
    U[2] = F[0, 0]
    U[3] = F[0, 1]
    U[4] = F[1, 0]
    U[5] = F[1, 1]
    return u, U, F


def exact_cauchy_stress(material: WarpHyperelasticMaterial, F: np.ndarray) -> np.ndarray:
    P = _first_piola_np(np.asarray(F, dtype=np.float64), material)
    J = float(np.linalg.det(F))
    sigma = P @ F.T / J
    return np.array([sigma[0, 0], sigma[0, 1], sigma[1, 0], sigma[1, 1]], dtype=np.float64)


def initial_fields(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, Y = reference_grid(n)
    _u, U0, _F = exact_fields(X, Y, 0.0)
    return U0, np.zeros((2, n, n), dtype=np.float64), X, Y


def run_one(args: argparse.Namespace, n: int, out_dir: Path, *, save_state: bool) -> dict[str, Any]:
    geom = CurvedBoundaryGeometry.annulus(
        nx=int(n),
        ny=int(n),
        inner_radius=INNER_RADIUS,
        outer_radius=OUTER_RADIUS,
        center_x=CENTER[0],
        center_y=CENTER[1],
    )
    material = WarpHyperelasticMaterial.from_poisson(
        str(args.material), float(args.poisson), mu=float(args.mu)
    )
    omega_n = effective_relaxation_omega(
        base_omega=float(args.omega),
        dx=float(geom.dx),
        mode=str(args.omega_scaling),
        kappa=float(args.omega_kappa),
    )
    solver = WarpCurvedBoundaryLBM2D(
        geom,
        material=material,
        boundary=annulus_boundary(),
        lattice_speed=float(args.lattice_speed),
        collision_omega=float(omega_n),
        collision_model=str(args.collision_model),
        ghost_omega=None if args.ghost_omega is None else float(args.ghost_omega),
        kinetic_filter_strength=float(args.kinetic_filter_strength),
        boundary_reconstruction=str(args.boundary_reconstruction),
        source_mode="none",
        local_compatibility_interval=int(args.local_compatibility_interval),
        local_compatibility_blend=float(args.local_compatibility_blend),
        local_compatibility_interior_only=bool(args.local_compatibility_interior_only),
        device=str(args.device),
    )
    U0, u0, X, Y = initial_fields(int(n))
    solver.initialize_from_numpy(U=U0, displacement=u0, init_order=int(args.init_order))

    t0 = time.perf_counter()
    solver.run_until(T_END, check_interval=int(args.check_interval))
    wall_time = time.perf_counter() - t0

    active = solver.active_mask()
    u_exact, U_exact, F_exact = exact_fields(X, Y, solver.time)
    u_num = solver.displacement()
    U_num = solver.system_state()
    F_num = solver.deformation_gradient()
    sigma_num = solver.numpy_field("sigma")
    sigma_exact_vec = exact_cauchy_stress(material, F_exact)
    sigma_exact_field = np.empty_like(sigma_num)
    sigma_exact_field[0] = sigma_exact_vec[0]
    sigma_exact_field[1] = sigma_exact_vec[1]
    sigma_exact_field[2] = sigma_exact_vec[2]
    sigma_exact_field[3] = sigma_exact_vec[3]
    J = solver.deformation_jacobian()
    F_exact_field = np.empty_like(F_num)
    F_exact_field[0, 0] = U_exact[2]
    F_exact_field[0, 1] = U_exact[3]
    F_exact_field[1, 0] = U_exact[4]
    F_exact_field[1, 1] = U_exact[5]
    I = np.eye(2, dtype=np.float64)[:, :, None, None]

    ids, counts = np.unique(geom.link_boundary_id, return_counts=True)
    link_count_by_id = {int(bid): int(count) for bid, count in zip(ids, counts)}
    row: dict[str, Any] = {
        "case": "annulus_affine_compat_dirichlet",
        "n": int(n),
        "dx": float(geom.dx),
        "dt": float(solver.dt),
        "collision_omega": float(omega_n),
        "active_nodes": int(np.count_nonzero(active)),
        "cut_links": int(geom.n_links),
        "outer_cut_links": int(link_count_by_id.get(BOUNDARY_ID_OUTER, 0)),
        "inner_cut_links": int(link_count_by_id.get(BOUNDARY_ID_INNER, 0)),
        "steps": int(solver.steps),
        "time": float(solver.time),
        "wall_time_s": float(wall_time),
        "rel_l2_u": float(relative_l2(u_num - u_exact, u_exact, active)),
        "rel_l2_v": float(relative_l2(U_num[:2] - U_exact[:2], U_exact[:2], active)),
        "rel_l2_F": float(relative_l2(F_num - F_exact_field, F_exact_field, active)),
        "rel_l2_F_minus_I": float(relative_l2(F_num - F_exact_field, F_exact_field - I, active)),
        "rel_l2_sigma": float(
            relative_l2(sigma_num - sigma_exact_field, sigma_exact_field, active)
        ),
        "min_J": float(np.min(J[active])),
        "max_J": float(np.max(J[active])),
        "J_exact": float(np.linalg.det(F_exact)),
        "finite": bool(not solver.invalid_state()),
    }
    if save_state:
        npz_path = out_dir / f"annulus_affine_compat_n{n}_final.npz"
        solver.save_npz(npz_path)
        row["npz"] = str(npz_path)
    print(json.dumps(row), flush=True)
    return row


def add_orders(rows: list[dict[str, Any]], metric: str) -> None:
    last: dict[str, Any] | None = None
    for row in rows:
        row[f"order_{metric}"] = None
        if last is not None:
            e0 = float(last[metric])
            e1 = float(row[metric])
            h0 = float(last["dx"])
            h1 = float(row["dx"])
            if e0 > 0.0 and e1 > 0.0 and h0 > h1:
                row[f"order_{metric}"] = float(math.log(e0 / e1) / math.log(h0 / h1))
        last = row


def write_markdown(rows: list[dict[str, Any]], out_dir: Path) -> None:
    def fmt(value: Any, digits: int = 3) -> str:
        if value is None:
            return ""
        return f"{float(value):.{digits}e}"

    lines = [
        "# Annulus Affine Dirichlet Compat Convergence",
        "",
        "Reference solution: analytic affine dynamic solution, no FEM.",
        "",
        "| n | dx | omega | wall s | rel u | order u | rel sigma | order sigma | rel F-I | order F-I | finite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["n"]),
                    f"{float(row['dx']):.6g}",
                    f"{float(row['collision_omega']):.6g}",
                    f"{float(row['wall_time_s']):.2f}",
                    fmt(row["rel_l2_u"]),
                    fmt(row["order_rel_l2_u"], 2),
                    fmt(row["rel_l2_sigma"]),
                    fmt(row["order_rel_l2_sigma"], 2),
                    fmt(row["rel_l2_F_minus_I"]),
                    fmt(row["order_rel_l2_F_minus_I"], 2),
                    str(bool(row["finite"])).lower(),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Notes:",
        "",
        "- Inner and outer annulus cut links are both affine velocity Dirichlet boundaries.",
        "- The `omega` column is the effective BGK relaxation parameter used at that grid level.",
        "- The run uses link-local `compat_bfl`; no global compatibility projection exists in `newcode`.",
        "- `F-I` is included because the relative error of `F` itself is dominated by the identity tensor.",
    ]
    (out_dir / "annulus_affine_compat_convergence.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def plot_convergence(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = np.asarray([float(r["dx"]) for r in rows])
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for metric, label, marker in (
        ("rel_l2_u", "u", "o"),
        ("rel_l2_sigma", "sigma", "s"),
        ("rel_l2_F_minus_I", "F-I", "^"),
    ):
        err = np.asarray([float(r[metric]) for r in rows])
        ax0.loglog(h, err, marker + "-", lw=1.6, label=label)
        orders = np.asarray(
            [np.nan if r[f"order_{metric}"] is None else float(r[f"order_{metric}"]) for r in rows]
        )
        ax1.plot([int(r["n"]) for r in rows[1:]], orders[1:], marker + "-", lw=1.6, label=label)
    if rows and float(rows[-1]["rel_l2_u"]) > 0.0:
        ref = float(rows[-1]["rel_l2_u"]) * (h / h[-1]) ** 2
        ax0.loglog(h, ref, "--", color="0.35", lw=1.1, label="slope 2")
    ax0.invert_xaxis()
    ax0.set_xlabel("dx")
    ax0.set_ylabel("relative L2 error")
    ax0.grid(True, which="both", alpha=0.35)
    ax0.legend()
    ax1.axhline(2.0, color="0.35", ls="--", lw=1.1)
    ax1.set_xlabel("grid n")
    ax1.set_ylabel("pairwise observed order")
    ax1.grid(True, alpha=0.35)
    ax1.legend()
    fig.savefig(out_dir / "annulus_affine_compat_convergence.png", dpi=200)
    plt.close(fig)


def plot_finest_state(npz_path: Path, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(npz_path)
    X = data["x"]
    Y = data["y"]
    active = data["active"].astype(bool)
    u = data["u"]
    U = data["U"]
    F = data["F"]
    t = float(data["time"])
    u_exact, U_exact, _F_exact = exact_fields(X, Y, t)
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    speed_err = np.sqrt((U[0] - U_exact[0]) ** 2 + (U[1] - U_exact[1]) ** 2)
    disp_err = np.sqrt((u[0] - u_exact[0]) ** 2 + (u[1] - u_exact[1]) ** 2)
    F_err = np.sqrt(
        (F[0, 0] - U_exact[2]) ** 2
        + (F[0, 1] - U_exact[3]) ** 2
        + (F[1, 0] - U_exact[4]) ** 2
        + (F[1, 1] - U_exact[5]) ** 2
    )
    panels = [
        (np.sqrt(u[0] * u[0] + u[1] * u[1]), "|u|"),
        (disp_err, "||u-u_exact||"),
        (speed_err, "||v-v_exact||"),
        (F_err, "||F-F_exact||"),
        (J, "J"),
    ]
    fig, axs = plt.subplots(1, len(panels), figsize=(17.0, 3.6), constrained_layout=True)
    for ax, (field, title) in zip(axs, panels):
        sc = ax.scatter(X[active], Y[active], c=field[active], s=3, cmap="viridis", linewidths=0.0)
        theta = np.linspace(0.0, 2.0 * np.pi, 720)
        for radius in (OUTER_RADIUS, INNER_RADIUS):
            ax.plot(
                CENTER[0] + radius * np.cos(theta),
                CENTER[1] + radius * np.sin(theta),
                color="black",
                lw=0.8,
            )
        ax.set_aspect("equal")
        ax.set_xlim(CENTER[0] - OUTER_RADIUS - 0.02, CENTER[0] + OUTER_RADIUS + 0.02)
        ax.set_ylim(CENTER[1] - OUTER_RADIUS - 0.02, CENTER[1] + OUTER_RADIUS + 0.02)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    fig.savefig(out_dir / f"annulus_affine_compat_n{X.shape[0]}_state.png", dpi=220)
    plt.close(fig)


def parse_n_values(args: argparse.Namespace) -> list[int]:
    if args.n is not None:
        return [int(v) for v in str(args.n).split(",") if v.strip()]
    return [int(v) for v in args.n_values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--n", default=None, help="comma-separated grid sizes; overrides --n-values"
    )
    parser.add_argument("--n-values", type=int, nargs="+", default=[64, 96, 128, 192, 256, 384])
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--material",
        choices=("neo_hooke", "log_neo_hooke", "svk", "mooney_rivlin", "yeoh", "gent"),
        default="neo_hooke",
    )
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--poisson", type=float, default=0.20)
    parser.add_argument("--lattice-speed", type=float, default=10.0)
    parser.add_argument("--omega", type=float, default=2.0)
    parser.add_argument("--omega-scaling", choices=("constant", "dx"), default="constant")
    parser.add_argument("--omega-kappa", type=float, default=20.0)
    parser.add_argument("--collision-model", choices=("bgk", "mrt_ghost"), default="bgk")
    parser.add_argument("--ghost-omega", type=float, default=None)
    parser.add_argument("--kinetic-filter-strength", type=float, default=0.0)
    parser.add_argument(
        "--boundary-reconstruction",
        choices=(
            "bfl",
            "qbfl",
            "lsq_bfl",
            "normal_probe_bfl",
            "regularized_bfl",
            "regularized_nee",
            "pair_nee",
            "same_nee",
            "poly_ibb",
            "lowq_reg_bfl",
            "compat_bfl",
            "compat_nee",
            "local_f",
            "local_f_bfl",
        ),
        default="compat_bfl",
    )
    parser.add_argument("--local-compatibility-interval", type=int, default=0)
    parser.add_argument("--local-compatibility-blend", type=float, default=0.05)
    parser.add_argument(
        "--local-compatibility-all-active",
        dest="local_compatibility_interior_only",
        action="store_false",
        default=True,
    )
    parser.add_argument("--init-order", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument(
        "--check-interval",
        type=int,
        default=0,
        help="optional CPU diagnostic interval; 0 keeps the time loop GPU-only",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_values = parse_n_values(args)
    rows = [
        run_one(args, int(n), out_dir, save_state=(idx == len(n_values) - 1))
        for idx, n in enumerate(n_values)
    ]
    rows.sort(key=lambda item: int(item["n"]))
    for metric in ("rel_l2_u", "rel_l2_v", "rel_l2_F", "rel_l2_F_minus_I", "rel_l2_sigma"):
        add_orders(rows, metric)
    summary = {
        "case": "annulus_affine_compat_dirichlet",
        "description": "pure LBM convergence against analytic affine dynamic Dirichlet solution",
        "device": str(args.device),
        "center": CENTER,
        "inner_radius": INNER_RADIUS,
        "outer_radius": OUTER_RADIUS,
        "T_end": T_END,
        "A_final": A_FINAL.tolist(),
        "F_final": (np.eye(2, dtype=np.float64) + A_FINAL).tolist(),
        "J_final": float(np.linalg.det(np.eye(2, dtype=np.float64) + A_FINAL)),
        "boundary_reconstruction": str(args.boundary_reconstruction),
        "omega_scaling": str(args.omega_scaling),
        "base_omega": float(args.omega),
        "omega_kappa": float(args.omega_kappa),
        "compatibility_projection": {"active": False, "implementation": "not available in newcode"},
        "rows": rows,
    }
    summary_path = out_dir / "annulus_affine_compat_convergence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(rows, out_dir)
    if not args.no_plots:
        plot_convergence(rows, out_dir)
        finest_npz = out_dir / f"annulus_affine_compat_n{n_values[-1]}_final.npz"
        if finest_npz.exists():
            plot_finest_state(finest_npz, out_dir)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
