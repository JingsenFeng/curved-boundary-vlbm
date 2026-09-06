"""FEniCSx quasi-static reference for the stretched and twisted finite tube.

Run in a FEniCSx environment; see docs/reference-data.md for the paper settings.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc


def build_annular_hex_mesh(
    *,
    inner_radius: float,
    outer_radius: float,
    length_z: float,
    center_xy: tuple[float, float],
    nr: int,
    ntheta: int,
    nz: int,
):
    comm = MPI.COMM_WORLD
    cx, cy = center_xy
    rs = np.linspace(inner_radius, outer_radius, int(nr) + 1)
    thetas = np.linspace(0.0, 2.0 * math.pi, int(ntheta), endpoint=False)
    zs = np.linspace(0.0, length_z, int(nz) + 1)
    points = np.empty(((int(nz) + 1) * (int(nr) + 1) * int(ntheta), 3), dtype=np.float64)

    def node(ir: int, it: int, iz: int) -> int:
        return (iz * (int(nr) + 1) + ir) * int(ntheta) + (it % int(ntheta))

    for iz, z in enumerate(zs):
        for ir, r in enumerate(rs):
            for it, th in enumerate(thetas):
                p = node(ir, it, iz)
                points[p] = (cx + r * math.cos(th), cy + r * math.sin(th), z)

    cells = []
    cell_index = 0
    for iz in range(int(nz)):
        for ir in range(int(nr)):
            for it in range(int(ntheta)):
                it1 = (it + 1) % int(ntheta)
                if cell_index % comm.size == comm.rank:
                    cells.append(
                        [
                            node(ir, it, iz),
                            node(ir + 1, it, iz),
                            node(ir, it1, iz),
                            node(ir + 1, it1, iz),
                            node(ir, it, iz + 1),
                            node(ir + 1, it, iz + 1),
                            node(ir, it1, iz + 1),
                            node(ir + 1, it1, iz + 1),
                        ]
                    )
                cell_index += 1
    cell_arr = np.asarray(cells, dtype=np.int64)
    coord_el = basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,))
    return mesh.create_mesh(comm, cell_arr, coord_el, points)


def right_displacement_expr(
    *,
    center_xy: tuple[float, float],
    axial_displacement: float,
    twist_angle: float,
    scale: float,
):
    cx, cy = center_xy
    theta = float(scale) * float(twist_angle)
    uz = float(scale) * float(axial_displacement)
    c = math.cos(theta)
    s = math.sin(theta)

    def value(x: np.ndarray) -> np.ndarray:
        dx0 = x[0] - cx
        dy0 = x[1] - cy
        out = np.empty((3, x.shape[1]), dtype=PETSc.ScalarType)
        out[0] = (c - 1.0) * dx0 - s * dy0
        out[1] = s * dx0 + (c - 1.0) * dy0
        out[2] = uz
        return out

    return value


def det3(F: np.ndarray) -> np.ndarray:
    return (
        F[:, 0, 0] * (F[:, 1, 1] * F[:, 2, 2] - F[:, 1, 2] * F[:, 2, 1])
        - F[:, 0, 1] * (F[:, 1, 0] * F[:, 2, 2] - F[:, 1, 2] * F[:, 2, 0])
        + F[:, 0, 2] * (F[:, 1, 0] * F[:, 2, 1] - F[:, 1, 1] * F[:, 2, 0])
    )


def sample_reference(
    *,
    domain,
    u_fun,
    F_fun,
    inner_radius: float,
    outer_radius: float,
    length_z: float,
    center_xy: tuple[float, float],
    nr_sample: int,
    ntheta_sample: int,
    nz_sample: int,
    profile_bins: int,
) -> dict[str, np.ndarray]:
    comm = domain.comm
    cx, cy = center_xy
    rs = np.linspace(inner_radius, outer_radius, int(nr_sample) + 2)[1:-1]
    thetas = np.linspace(0.0, 2.0 * math.pi, int(ntheta_sample), endpoint=False)
    zs = np.linspace(0.0, length_z, int(nz_sample) + 2)[1:-1]
    pts = np.empty((len(rs) * len(thetas) * len(zs), 3), dtype=np.float64)
    idx = 0
    for z in zs:
        for r in rs:
            for th in thetas:
                pts[idx] = (cx + r * math.cos(th), cy + r * math.sin(th), z)
                idx += 1

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidate_cells = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, candidate_cells, pts)
    valid_local: list[int] = []
    cells: list[int] = []
    for k in range(pts.shape[0]):
        links = colliding_cells.links(k)
        if len(links) > 0:
            valid_local.append(k)
            cells.append(int(links[0]))
    valid_local_arr = np.asarray(valid_local, dtype=np.int64)
    points = pts[valid_local_arr]
    cells_arr = np.asarray(cells, dtype=np.int32)
    if points.size:
        u_values = np.asarray(u_fun.eval(points, cells_arr), dtype=np.float64)
        F_values = np.asarray(F_fun.eval(points, cells_arr), dtype=np.float64)
        if F_values.ndim == 2 and F_values.shape[1] == 9:
            F_values = F_values.reshape((-1, 3, 3))
        J = det3(F_values)
        dx0 = points[:, 0] - cx
        dy0 = points[:, 1] - cy
        r2 = np.maximum(dx0 * dx0 + dy0 * dy0, 1.0e-30)
        utheta_over_r = (-dy0 * u_values[:, 0] + dx0 * u_values[:, 1]) / r2
    else:
        u_values = np.empty((0, 3), dtype=np.float64)
        F_values = np.empty((0, 3, 3), dtype=np.float64)
        J = np.empty((0,), dtype=np.float64)
        utheta_over_r = np.empty((0,), dtype=np.float64)

    gathered = comm.gather((points, u_values, F_values, J, utheta_over_r), root=0)
    centers = 0.5 * (
        np.linspace(0.0, length_z, int(profile_bins) + 1)[:-1]
        + np.linspace(0.0, length_z, int(profile_bins) + 1)[1:]
    )
    if comm.rank != 0:
        return {
            "points": np.empty((0, 3), dtype=np.float64),
            "u": np.empty((0, 3), dtype=np.float64),
            "F": np.empty((0, 3, 3), dtype=np.float64),
            "J": np.empty((0,), dtype=np.float64),
            "utheta_over_r": np.empty((0,), dtype=np.float64),
            "profile_z": centers,
            "profile_utheta_over_r": np.full(centers.shape, np.nan, dtype=np.float64),
            "profile_uz": np.full(centers.shape, np.nan, dtype=np.float64),
            "profile_J": np.full(centers.shape, np.nan, dtype=np.float64),
        }

    points = np.concatenate([item[0] for item in gathered], axis=0)
    u_values = np.concatenate([item[1] for item in gathered], axis=0)
    F_values = np.concatenate([item[2] for item in gathered], axis=0)
    J = np.concatenate([item[3] for item in gathered], axis=0)
    utheta_over_r = np.concatenate([item[4] for item in gathered], axis=0)
    if points.size:
        _, keep = np.unique(np.round(points, decimals=13), axis=0, return_index=True)
        keep = np.sort(keep)
        points = points[keep]
        u_values = u_values[keep]
        F_values = F_values[keep]
        J = J[keep]
        utheta_over_r = utheta_over_r[keep]

    bins = np.linspace(0.0, length_z, int(profile_bins) + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    which = np.digitize(points[:, 2], bins) - 1

    def bmean(values: np.ndarray) -> np.ndarray:
        out = np.full(centers.shape, np.nan, dtype=np.float64)
        for i in range(centers.size):
            mask = which == i
            if np.any(mask):
                out[i] = float(np.mean(values[mask]))
        return out

    return {
        "points": points,
        "u": u_values,
        "F": F_values,
        "J": J,
        "utheta_over_r": utheta_over_r,
        "profile_z": centers,
        "profile_utheta_over_r": bmean(utheta_over_r),
        "profile_uz": bmean(u_values[:, 2]),
        "profile_J": bmean(J),
    }


def solve(args: argparse.Namespace) -> dict[str, object]:
    length_z = float(args.outer_radius) / float(args.radius_nodes) * float(args.z_nodes)
    center_xy = (
        float(args.outer_radius)
        + float(args.margin_nodes) * float(args.outer_radius) / float(args.radius_nodes),
    ) * 2
    inner_radius = float(args.inner_radius)
    if inner_radius < 0.0:
        inner_radius = float(args.inner_ratio) * float(args.outer_radius)
    axial_displacement = float(args.target_strain) * length_z

    domain = build_annular_hex_mesh(
        inner_radius=inner_radius,
        outer_radius=float(args.outer_radius),
        length_z=length_z,
        center_xy=center_xy,
        nr=int(args.nr),
        ntheta=int(args.ntheta),
        nz=int(args.nz),
    )
    comm = domain.comm
    root = comm.rank == 0
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    fdim = domain.topology.dim - 1
    left_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[2], 0.0))
    right_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[2], length_z))

    V = fem.functionspace(domain, ("Lagrange", int(args.degree), (3,)))
    u = fem.Function(V, name="u_fem")
    test = ufl.TestFunction(V)
    trial = ufl.TrialFunction(V)

    left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
    right_dofs = fem.locate_dofs_topological(V, fdim, right_facets)
    zero = np.array((0.0, 0.0, 0.0), dtype=PETSc.ScalarType)
    right_bc_fun = fem.Function(V, name="right_displacement")
    bc_left = fem.dirichletbc(zero, left_dofs, V)
    bc_right = fem.dirichletbc(right_bc_fun, right_dofs)

    mu = float(args.mu)
    poisson = float(args.poisson)
    lam = 2.0 * mu * poisson / (1.0 - 2.0 * poisson)

    I = ufl.Identity(3)
    F = I + ufl.grad(u)
    J = ufl.det(F)
    FinvT = ufl.inv(F).T
    P = mu * (F - FinvT) + 0.5 * lam * (J * J - 1.0) * FinvT
    residual = ufl.inner(P, ufl.grad(test)) * ufl.dx
    jacobian = ufl.derivative(residual, u, trial)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": float(args.newton_rtol),
        "snes_atol": float(args.newton_atol),
        "snes_max_it": int(args.newton_max_it),
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    if str(args.pc_factor_mat_solver_type):
        petsc_options["pc_factor_mat_solver_type"] = str(args.pc_factor_mat_solver_type)

    problem = petsc.NonlinearProblem(
        residual,
        u,
        bcs=[bc_left, bc_right],
        J=jacobian,
        petsc_options_prefix="tube3d_",
        petsc_options=petsc_options,
    )

    load_history = []
    for step in range(1, int(args.load_steps) + 1):
        scale = float(step) / float(args.load_steps)
        right_bc_fun.interpolate(
            right_displacement_expr(
                center_xy=center_xy,
                axial_displacement=axial_displacement,
                twist_angle=float(args.target_twist_angle),
                scale=scale,
            )
        )
        right_bc_fun.x.scatter_forward()
        problem.solve()
        u.x.scatter_forward()
        iterations = int(problem.solver.getIterationNumber())
        reason = int(problem.solver.getConvergedReason())
        load_history.append(
            {
                "step": step,
                "scale": scale,
                "newton_iterations": iterations,
                "reason": reason,
                "converged": reason > 0,
            }
        )
        if root:
            print(json.dumps(load_history[-1]), flush=True)
        if reason <= 0:
            raise RuntimeError(
                f"FEM Newton did not converge at load step {step}; SNES reason {reason}"
            )

    W = fem.functionspace(domain, ("DG", max(int(args.degree) - 1, 0), (3, 3)))
    F_fun = fem.Function(W, name="F_fem")
    interpolation_points = W.element.interpolation_points
    if callable(interpolation_points):
        interpolation_points = interpolation_points()
    F_fun.interpolate(fem.Expression(F, interpolation_points))

    sampled = sample_reference(
        domain=domain,
        u_fun=u,
        F_fun=F_fun,
        inner_radius=inner_radius,
        outer_radius=float(args.outer_radius),
        length_z=length_z,
        center_xy=center_xy,
        nr_sample=int(args.nr_sample),
        ntheta_sample=int(args.ntheta_sample),
        nz_sample=int(args.nz_sample),
        profile_bins=int(args.profile_bins),
    )

    if not root:
        return {}

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "tube_pull_torsion_coarse_fem_reference.npz"
    np.savez_compressed(
        out_npz,
        **sampled,
        length_z=np.array(length_z),
        center_xy=np.asarray(center_xy, dtype=np.float64),
        inner_radius=np.array(inner_radius),
        outer_radius=np.array(float(args.outer_radius)),
        axial_displacement=np.array(axial_displacement),
        target_twist_angle=np.array(float(args.target_twist_angle)),
    )
    meta = {
        "case": "coarse_3d_tube_pull_torsion_quasistatic_fem",
        "dolfinx_version": __import__("dolfinx").__version__,
        "mpi_size": int(comm.size),
        "mesh": {
            "nr": int(args.nr),
            "ntheta": int(args.ntheta),
            "nz": int(args.nz),
            "degree": int(args.degree),
        },
        "sample": {
            "nr": int(args.nr_sample),
            "ntheta": int(args.ntheta_sample),
            "nz": int(args.nz_sample),
            "profile_bins": int(args.profile_bins),
        },
        "geometry": {
            "inner_radius": inner_radius,
            "outer_radius": float(args.outer_radius),
            "length_z": length_z,
            "center_xy": list(center_xy),
        },
        "loading": {
            "target_strain": float(args.target_strain),
            "axial_displacement": axial_displacement,
            "target_twist_angle": float(args.target_twist_angle),
        },
        "material": {"mu": mu, "lambda": lam, "poisson": poisson},
        "solver": {
            "load_steps": int(args.load_steps),
            "newton_rtol": float(args.newton_rtol),
            "newton_atol": float(args.newton_atol),
            "newton_max_it": int(args.newton_max_it),
            "petsc_options": petsc_options,
        },
        "load_history": load_history,
        "out_npz": str(out_npz),
        "max_abs_u": float(np.max(np.sqrt(np.sum(sampled["u"] * sampled["u"], axis=1)))),
        "min_J": float(np.min(sampled["J"])),
        "max_J": float(np.max(sampled["J"])),
    }
    out_json = out_dir / "tube_pull_torsion_coarse_fem_reference.json"
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius-nodes", type=int, default=30)
    parser.add_argument("--z-nodes", type=int, default=300)
    parser.add_argument("--margin-nodes", type=int, default=8)
    parser.add_argument("--outer-radius", type=float, default=0.30)
    parser.add_argument("--inner-radius", type=float, default=-1.0)
    parser.add_argument("--inner-ratio", type=float, default=0.50)
    parser.add_argument("--target-strain", type=float, default=0.50)
    parser.add_argument("--target-twist-angle", type=float, default=math.pi / 4.0)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--poisson", type=float, default=0.20)
    parser.add_argument("--nr", type=int, default=6)
    parser.add_argument("--ntheta", type=int, default=48)
    parser.add_argument("--nz", type=int, default=60)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--load-steps", type=int, default=20)
    parser.add_argument("--newton-rtol", type=float, default=1.0e-8)
    parser.add_argument("--newton-atol", type=float, default=1.0e-9)
    parser.add_argument("--newton-max-it", type=int, default=40)
    parser.add_argument("--pc-factor-mat-solver-type", default="")
    parser.add_argument("--nr-sample", type=int, default=8)
    parser.add_argument("--ntheta-sample", type=int, default=96)
    parser.add_argument("--nz-sample", type=int, default=120)
    parser.add_argument("--profile-bins", type=int, default=79)
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).resolve().parents[2] / "results" / "fem-tube"),
    )
    args = parser.parse_args()
    solve(args)


if __name__ == "__main__":
    main()
