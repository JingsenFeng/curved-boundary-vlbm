#!/usr/bin/env python3
"""Mixed Dirichlet/Neumann superellipse-shell benchmark against FEniCSx.

The reference domain is a non-circular shell bounded by two rotated
superellipses.  The inner boundary is clamped through the LBM velocity
Dirichlet rule, while the outer boundary carries a constant nominal traction.

There is no closed-form solution for this mixed curved-boundary problem.  This
driver therefore generates a FEniCSx static finite-element reference on a gmsh
boundary-fitted mesh, samples that reference on the LBM cell centers, and then
compares a damped curved-boundary LBM run with the sampled FEM fields.

The script is intentionally single-file.  The default Python environment has
Warp/LBM, while the FEniCSx environment usually does not.  The main process runs
LBM and launches this same script with ``--fem-only`` using ``--fenicsx-python``
to generate the FEM reference.
"""

from __future__ import annotations

import argparse
from traction_settings import apply_defaults
from dataclasses import dataclass
import json
import math
from pathlib import Path
from _paths import ROOT, RESULTS, REFERENCES, FENICSX_PYTHON, install_reference
import subprocess
import sys
from typing import Any

import numpy as np


SCRIPT = Path(__file__).resolve()
RECON_ROOT = ROOT / "src"
PACKAGE_ROOT = ROOT
NEWFIG_ROOT = RESULTS
DEFAULT_RESULTS = NEWFIG_ROOT / "fig4_5_superellipse" / "data"
DEFAULT_FENICSX_PYTHON = FENICSX_PYTHON

OUTER_ID = 1
INNER_ID = 2


def tag_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def case_name(args: argparse.Namespace) -> str:
    requested = str(getattr(args, "case_name", "")).strip()
    if requested:
        return requested
    fem_tag = "" if str(getattr(args, "fem_mode", "static")) == "static" else "_femdyn"
    return (
        f"superellipse_mixed_{args.mode}{fem_tag}_n{int(args.n)}"
        f"_tx{tag_float(float(args.traction_x))}_ty{tag_float(float(args.traction_y))}"
    )


def case_output_dir(args: argparse.Namespace) -> Path:
    return Path(args.results_dir) / case_name(args)


@dataclass(frozen=True)
class Superellipse:
    a: float
    b: float
    theta_deg: float
    p: float = 4.0
    cx: float = 0.5
    cy: float = 0.5

    @property
    def theta(self) -> float:
        return math.radians(self.theta_deg)

    def local(self, x: np.ndarray | float, y: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        ct = math.cos(self.theta)
        st = math.sin(self.theta)
        dx = np.asarray(x, dtype=np.float64) - self.cx
        dy = np.asarray(y, dtype=np.float64) - self.cy
        xi = ct * dx + st * dy
        eta = -st * dx + ct * dy
        return xi, eta

    def phi(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        xi, eta = self.local(x, y)
        return (np.abs(xi) / self.a) ** self.p + (np.abs(eta) / self.b) ** self.p - 1.0

    def grad(self, x: float, y: float) -> tuple[float, float]:
        xi, eta = self.local(float(x), float(y))
        xi_f = float(xi)
        eta_f = float(eta)
        ct = math.cos(self.theta)
        st = math.sin(self.theta)
        gx_l = self.p * math.copysign(abs(xi_f) ** (self.p - 1.0), xi_f) / (self.a**self.p)
        gy_l = self.p * math.copysign(abs(eta_f) ** (self.p - 1.0), eta_f) / (self.b**self.p)
        gx = ct * gx_l - st * gy_l
        gy = st * gx_l + ct * gy_l
        return gx, gy

    def point(self, t: float) -> tuple[float, float]:
        ct = math.cos(self.theta)
        st = math.sin(self.theta)
        c = math.cos(t)
        s = math.sin(t)
        # Standard superellipse parametrisation for |xi/a|^p + |eta/b|^p = 1.
        xi = self.a * math.copysign(abs(c) ** (2.0 / self.p), c)
        eta = self.b * math.copysign(abs(s) ** (2.0 / self.p), s)
        x = self.cx + ct * xi - st * eta
        y = self.cy + st * xi + ct * eta
        return x, y


def outer_spec(args: argparse.Namespace) -> Superellipse:
    return Superellipse(
        a=float(args.outer_a),
        b=float(args.outer_b),
        theta_deg=float(args.outer_theta),
        p=float(args.superellipse_power),
        cx=float(args.center_x),
        cy=float(args.center_y),
    )


def inner_spec(args: argparse.Namespace) -> Superellipse:
    return Superellipse(
        a=float(args.inner_a),
        b=float(args.inner_b),
        theta_deg=float(args.inner_theta),
        p=float(args.superellipse_power),
        cx=float(args.center_x),
        cy=float(args.center_y),
    )


def shell_phi(
    x: np.ndarray | float, y: np.ndarray | float, outer: Superellipse, inner: Superellipse
) -> np.ndarray:
    # Material is inside the outer superellipse and outside the inner one.
    return np.maximum(outer.phi(x, y), -inner.phi(x, y))


def shell_normal(
    x: float, y: float, outer: Superellipse, inner: Superellipse
) -> tuple[float, float]:
    outer_phi = float(outer.phi(x, y))
    inner_phi = float(inner.phi(x, y))
    if abs(inner_phi) <= abs(outer_phi):
        gx, gy = inner.grad(x, y)
        gx, gy = -gx, -gy
    else:
        gx, gy = outer.grad(x, y)
    nrm = math.hypot(gx, gy)
    if nrm <= 1.0e-30:
        raise ValueError("zero superellipse normal")
    return gx / nrm, gy / nrm


def boundary_id(
    x: float, y: float, _nx: float, _ny: float, outer: Superellipse, inner: Superellipse
) -> int:
    return INNER_ID if abs(float(inner.phi(x, y))) <= abs(float(outer.phi(x, y))) else OUTER_ID


def reference_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    dx = 1.0 / float(n)
    x = (np.arange(n, dtype=np.float64) + 0.5) * dx
    return np.meshgrid(x, x, indexing="ij")


def active_mask_for_grid(
    n: int, outer: Superellipse, inner: Superellipse
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Y = reference_grid(n)
    return X, Y, shell_phi(X, Y, outer, inner) <= 0.0


def build_lbm_geometry(args: argparse.Namespace):
    from curved_lbm.two_d.curved_boundary_warp import CurvedBoundaryGeometry

    outer = outer_spec(args)
    inner = inner_spec(args)
    return CurvedBoundaryGeometry.from_level_set(
        nx=int(args.n),
        ny=int(args.n),
        length_x=1.0,
        length_y=1.0,
        phi=lambda x, y: float(shell_phi(x, y, outer, inner)),
        normal=lambda x, y: shell_normal(x, y, outer, inner),
        boundary_id=lambda x, y, nx, ny: boundary_id(x, y, nx, ny, outer, inner),
        label=(
            "superellipse_shell("
            f"outer_a={outer.a},outer_b={outer.b},outer_theta={outer.theta_deg},"
            f"inner_a={inner.a},inner_b={inner.b},inner_theta={inner.theta_deg})"
        ),
    )


def build_gmsh_model(args: argparse.Namespace) -> None:
    import gmsh

    outer = outer_spec(args)
    inner = inner_spec(args)
    n_outer = int(args.gmsh_outer_points)
    n_inner = int(args.gmsh_inner_points)
    mesh_size = float(args.fem_mesh_size)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("superellipse_mixed_shell")

    def add_closed_spline(spec: Superellipse, count: int, reverse: bool) -> int:
        if reverse:
            ts = np.linspace(2.0 * math.pi, 0.0, count, endpoint=False)
        else:
            ts = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
        point_tags = []
        for t in ts:
            x, y = spec.point(float(t))
            point_tags.append(gmsh.model.geo.addPoint(x, y, 0.0, mesh_size))
        return gmsh.model.geo.addSpline(point_tags + [point_tags[0]])

    outer_curve = add_closed_spline(outer, n_outer, reverse=False)
    inner_curve = add_closed_spline(inner, n_inner, reverse=True)
    outer_loop = gmsh.model.geo.addCurveLoop([outer_curve])
    inner_loop = gmsh.model.geo.addCurveLoop([inner_curve])
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [surface], 1)
    gmsh.model.setPhysicalName(2, 1, "material")
    gmsh.model.addPhysicalGroup(1, [outer_curve], OUTER_ID)
    gmsh.model.setPhysicalName(1, OUTER_ID, "outer_neumann")
    gmsh.model.addPhysicalGroup(1, [inner_curve], INNER_ID)
    gmsh.model.setPhysicalName(1, INNER_ID, "inner_dirichlet")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.45 * mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    if int(args.geometry_order) > 1:
        gmsh.model.mesh.setOrder(int(args.geometry_order))
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)
        gmsh.model.mesh.optimize("HighOrder")


def sine2_hold_scale(time: float, ramp_time: float) -> float:
    if float(ramp_time) > 0.0 and float(time) < float(ramp_time):
        return math.sin(math.pi * float(time) / (2.0 * float(ramp_time))) ** 2
    return 1.0


def solve_dynamic_fem_reference(args: argparse.Namespace) -> dict[str, Any]:
    from mpi4py import MPI
    from petsc4py import PETSc

    from dolfinx import fem
    from dolfinx.fem import petsc
    from dolfinx.io import gmsh as gmshio
    import dolfinx
    import gmsh
    import ufl

    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("run this FEM reference with one MPI rank")

    build_gmsh_model(args)
    mesh_data = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()

    domain = mesh_data.mesh
    facet_tags = mesh_data.facet_tags
    if facet_tags is None:
        raise RuntimeError("gmsh mesh did not produce facet tags")

    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    degree = int(args.fem_degree)
    V = fem.functionspace(domain, ("Lagrange", degree, (2,)))
    u = fem.Function(V, name="u_fem")
    v_fun = fem.Function(V, name="v_fem")
    test = ufl.TestFunction(V)

    inner_facets = facet_tags.find(INNER_ID)
    inner_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, inner_facets)

    mu = float(args.mu)
    poisson = float(args.poisson)
    lam = 2.0 * mu * poisson / (1.0 - 2.0 * poisson)
    rho0 = float(args.rho0)

    I = ufl.Identity(2)
    F = I + ufl.grad(u)
    J = ufl.det(F)
    FinvT = ufl.inv(F).T
    P = mu * (F - FinvT) + 0.5 * lam * (J * J - 1.0) * FinvT
    internal_form = fem.form(ufl.inner(P, ufl.grad(test)) * ufl.dx)

    one = fem.Function(V)
    one.x.array[:] = 1.0
    mass_vec = petsc.assemble_vector(fem.form(rho0 * ufl.inner(one, test) * ufl.dx))
    mass = np.asarray(mass_vec.array, dtype=np.float64).reshape((-1, 2)).copy()
    mass_vec.destroy()
    mass_method = "row_sum_lumped"
    if not np.all(mass > 0.0):
        trial = ufl.TrialFunction(V)
        mass_matrix = petsc.assemble_matrix(fem.form(rho0 * ufl.inner(trial, test) * ufl.dx))
        mass_matrix.assemble()
        diag_vec = mass_matrix.getDiagonal()
        mass = np.asarray(diag_vec.array, dtype=np.float64).reshape((-1, 2)).copy()
        diag_vec.destroy()
        mass_matrix.destroy()
        mass_method = "consistent_mass_diagonal"
    if not np.all(mass > 0.0):
        raise RuntimeError("dynamic FEM mass diagonal contains non-positive entries")

    load_x = fem.Constant(domain, PETSc.ScalarType(0.0))
    load_y = fem.Constant(domain, PETSc.ScalarType(0.0))
    traction = ufl.as_vector((load_x, load_y))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    external_form = fem.form(ufl.dot(traction, test) * ds(OUTER_ID))

    u_values = np.asarray(u.x.array, dtype=np.float64).reshape((-1, 2))
    v_values = np.asarray(v_fun.x.array, dtype=np.float64).reshape((-1, 2))
    u_values[:, :] = 0.0
    v_values[:, :] = 0.0
    u_values[inner_dofs, :] = 0.0
    v_values[inner_dofs, :] = 0.0
    u.x.scatter_forward()
    v_fun.x.scatter_forward()

    force_history = []

    def acceleration(time: float) -> np.ndarray:
        u.x.scatter_forward()
        fint_vec = petsc.assemble_vector(internal_form)
        fint = np.asarray(fint_vec.array, dtype=np.float64).reshape((-1, 2)).copy()
        fint_vec.destroy()

        scale = sine2_hold_scale(float(time), float(args.ramp_time))
        load_x.value = PETSc.ScalarType(scale * float(args.traction_x))
        load_y.value = PETSc.ScalarType(scale * float(args.traction_y))
        fext_vec = petsc.assemble_vector(external_form)
        fext = np.asarray(fext_vec.array, dtype=np.float64).reshape((-1, 2)).copy()
        fext_vec.destroy()

        acc = (fext - fint) / mass - float(args.damping_gamma) * v_values
        acc[inner_dofs, :] = 0.0
        return acc

    t_end = float(args.run_time)
    dt_requested = float(args.fem_dt)
    if dt_requested <= 0.0:
        dt_requested = (1.0 / float(args.n)) / float(args.lattice_speed)
    nsteps = max(1, int(math.ceil(t_end / dt_requested)))
    dt = t_end / float(nsteps)
    acc = acceleration(0.0)
    progress_every = max(1, int(args.fem_progress_every))

    for step in range(nsteps):
        t_next = (step + 1) * dt
        u_values[:, :] += dt * v_values + 0.5 * dt * dt * acc
        u_values[inner_dofs, :] = 0.0
        u.x.scatter_forward()

        acc_next = acceleration(t_next)
        v_values[:, :] += 0.5 * dt * (acc + acc_next)
        v_values[inner_dofs, :] = 0.0
        v_fun.x.scatter_forward()
        acc = acc_next

        if not (np.isfinite(u_values).all() and np.isfinite(v_values).all()):
            raise RuntimeError(
                f"dynamic FEM produced non-finite values at step {step + 1}, time {t_next}"
            )

        if progress_every and ((step + 1) % progress_every == 0 or step + 1 == nsteps):
            max_u = float(np.max(np.sqrt(np.sum(u_values * u_values, axis=1))))
            max_v = float(np.max(np.sqrt(np.sum(v_values * v_values, axis=1))))
            msg = {
                "fem_dynamic_step": step + 1,
                "steps": nsteps,
                "time": t_next,
                "max_u": max_u,
                "max_v": max_v,
            }
            force_history.append(msg)
            print(json.dumps(msg), flush=True)

    W = fem.functionspace(domain, ("DG", max(degree - 1, 0), (2, 2)))
    F_fun = fem.Function(W, name="F_fem")
    interpolation_points = W.element.interpolation_points
    if callable(interpolation_points):
        interpolation_points = interpolation_points()
    F_expr = fem.Expression(F, interpolation_points)
    F_fun.interpolate(F_expr)

    out_npz = fem_output_path(args)
    sample_fem_to_grid(args, domain, u, F_fun, out_npz, v_fun=v_fun)
    out_json = out_npz.with_suffix(".json")
    meta = {
        "case": "superellipse_shell_mixed_dirichlet_neumann_dynamic_fem",
        "case_name": case_name(args),
        "results_dir": str(case_output_dir(args)),
        "dolfinx_version": dolfinx.__version__,
        "n": int(args.n),
        "fem_mode": "dynamic",
        "fem_degree": degree,
        "geometry_order": int(args.geometry_order),
        "fem_mesh_size": float(args.fem_mesh_size),
        "material": {"mu": mu, "lambda": lam, "poisson": poisson, "rho0": rho0},
        "mass_method": mass_method,
        "traction": [float(args.traction_x), float(args.traction_y)],
        "run_time": t_end,
        "ramp_time": float(args.ramp_time),
        "dt": dt,
        "steps": nsteps,
        "damping_gamma": float(args.damping_gamma),
        "outer": vars(outer_spec(args)),
        "inner": vars(inner_spec(args)),
        "progress": force_history,
        "out": str(out_npz),
    }
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def solve_fem_reference(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.fem_mode) == "dynamic":
        return solve_dynamic_fem_reference(args)

    from mpi4py import MPI
    from petsc4py import PETSc

    from dolfinx import fem
    from dolfinx.fem import petsc
    from dolfinx.io import gmsh as gmshio
    import dolfinx
    import gmsh
    import ufl

    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("run this FEM reference with one MPI rank")

    build_gmsh_model(args)
    mesh_data = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()

    domain = mesh_data.mesh
    facet_tags = mesh_data.facet_tags
    if facet_tags is None:
        raise RuntimeError("gmsh mesh did not produce facet tags")

    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    degree = int(args.fem_degree)
    V = fem.functionspace(domain, ("Lagrange", degree, (2,)))
    u = fem.Function(V, name="u_fem")
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    inner_facets = facet_tags.find(INNER_ID)
    inner_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, inner_facets)
    zero = np.array((0.0, 0.0), dtype=PETSc.ScalarType)
    bc = fem.dirichletbc(zero, inner_dofs, V)

    mu = float(args.mu)
    poisson = float(args.poisson)
    lam = 2.0 * mu * poisson / (1.0 - 2.0 * poisson)

    I = ufl.Identity(2)
    F = I + ufl.grad(u)
    J = ufl.det(F)
    FinvT = ufl.inv(F).T
    P = mu * (F - FinvT) + 0.5 * lam * (J * J - 1.0) * FinvT

    load_x = fem.Constant(domain, PETSc.ScalarType(0.0))
    load_y = fem.Constant(domain, PETSc.ScalarType(0.0))
    traction = ufl.as_vector((load_x, load_y))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    residual = ufl.inner(P, ufl.grad(v)) * ufl.dx - ufl.dot(traction, v) * ds(OUTER_ID)
    jacobian = ufl.derivative(residual, u, du)

    problem = petsc.NonlinearProblem(
        residual,
        u,
        petsc_options_prefix="superellipse_mixed_",
        bcs=[bc],
        J=jacobian,
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": float(args.fem_newton_rtol),
            "snes_atol": float(args.fem_newton_atol),
            "snes_max_it": int(args.fem_newton_max_it),
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    load_history = []
    for step in range(1, int(args.fem_load_steps) + 1):
        scale = float(step) / float(args.fem_load_steps)
        load_x.value = PETSc.ScalarType(scale * float(args.traction_x))
        load_y.value = PETSc.ScalarType(scale * float(args.traction_y))
        problem.solve()
        u.x.scatter_forward()
        iterations = problem.solver.getIterationNumber()
        reason = int(problem.solver.getConvergedReason())
        converged = reason > 0
        load_history.append(
            {
                "step": step,
                "scale": scale,
                "newton_iterations": int(iterations),
                "converged": bool(converged),
            }
        )
        if not converged:
            raise RuntimeError(
                f"FEM Newton solve did not converge at load step {step}; SNES reason {reason}"
            )

    W = fem.functionspace(domain, ("DG", max(degree - 1, 0), (2, 2)))
    F_fun = fem.Function(W, name="F_fem")
    interpolation_points = W.element.interpolation_points
    if callable(interpolation_points):
        interpolation_points = interpolation_points()
    F_expr = fem.Expression(F, interpolation_points)
    F_fun.interpolate(F_expr)

    out_npz = fem_output_path(args)
    sample_fem_to_grid(args, domain, u, F_fun, out_npz)
    out_json = out_npz.with_suffix(".json")
    meta = {
        "case": "superellipse_shell_mixed_dirichlet_neumann_fem",
        "case_name": case_name(args),
        "results_dir": str(case_output_dir(args)),
        "dolfinx_version": dolfinx.__version__,
        "n": int(args.n),
        "fem_degree": degree,
        "geometry_order": int(args.geometry_order),
        "fem_mesh_size": float(args.fem_mesh_size),
        "material": {"mu": mu, "lambda": lam, "poisson": poisson},
        "traction": [float(args.traction_x), float(args.traction_y)],
        "outer": vars(outer_spec(args)),
        "inner": vars(inner_spec(args)),
        "load_history": load_history,
        "out": str(out_npz),
    }
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def sample_fem_to_grid(
    args: argparse.Namespace,
    domain: Any,
    u_fun: Any,
    F_fun: Any,
    out_npz: Path,
    *,
    v_fun: Any | None = None,
) -> None:
    from dolfinx import geometry

    n = int(args.n)
    outer = outer_spec(args)
    inner = inner_spec(args)
    X, Y, active = active_mask_for_grid(n, outer, inner)
    points = np.zeros((int(np.count_nonzero(active)), 3), dtype=np.float64)
    active_indices = np.flatnonzero(active.ravel())
    points[:, 0] = X.ravel()[active_indices]
    points[:, 1] = Y.ravel()[active_indices]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidate_cells = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, candidate_cells, points)

    valid_local = []
    cells = []
    for k in range(points.shape[0]):
        links = colliding_cells.links(k)
        if len(links) > 0:
            valid_local.append(k)
            cells.append(int(links[0]))

    valid_local_arr = np.asarray(valid_local, dtype=np.int64)
    cells_arr = np.asarray(cells, dtype=np.int32)
    valid_points = points[valid_local_arr]

    u_values = u_fun.eval(valid_points, cells_arr)
    F_values = F_fun.eval(valid_points, cells_arr)
    v_values = None if v_fun is None else v_fun.eval(valid_points, cells_arr)
    if F_values.ndim == 2 and F_values.shape[1] == 4:
        F_values = F_values.reshape((-1, 2, 2))

    u_grid = np.full((2, n, n), np.nan, dtype=np.float64)
    v_grid = np.full((2, n, n), np.nan, dtype=np.float64)
    F_grid = np.full((2, 2, n, n), np.nan, dtype=np.float64)
    valid = np.zeros((n, n), dtype=bool)
    valid_flat = active_indices[valid_local_arr]
    ii, jj = np.unravel_index(valid_flat, (n, n))
    valid[ii, jj] = True
    u_grid[:, ii, jj] = np.asarray(u_values, dtype=np.float64).T
    if v_values is not None:
        v_grid[:, ii, jj] = np.asarray(v_values, dtype=np.float64).T
    F_grid[:, :, ii, jj] = np.asarray(F_values, dtype=np.float64).transpose(1, 2, 0)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        x=X,
        y=Y,
        active=active.astype(np.int32),
        valid=valid.astype(np.int32),
        u=u_grid,
        v=v_grid,
        F=F_grid,
        traction=np.asarray([float(args.traction_x), float(args.traction_y)], dtype=np.float64),
        outer=np.asarray(
            [
                float(args.outer_a),
                float(args.outer_b),
                float(args.outer_theta),
                float(args.superellipse_power),
            ],
            dtype=np.float64,
        ),
        inner=np.asarray(
            [
                float(args.inner_a),
                float(args.inner_b),
                float(args.inner_theta),
                float(args.superellipse_power),
            ],
            dtype=np.float64,
        ),
    )


def fem_output_path(args: argparse.Namespace) -> Path:
    return case_output_dir(args) / f"superellipse_mixed_fem_n{int(args.n)}.npz"


def ensure_fem_reference(args: argparse.Namespace) -> Path:
    out = fem_output_path(args)
    if not bool(args.force_fem):
        install_reference("superellipse", args, out)
    if out.exists() and not bool(args.force_fem):
        return out

    fenicsx_python = Path(args.fenicsx_python)
    if not fenicsx_python.exists():
        raise FileNotFoundError(f"FEniCSx Python not found: {fenicsx_python}")

    cmd = [
        str(fenicsx_python),
        str(SCRIPT),
        "--fem-only",
        "--n",
        str(int(args.n)),
        "--results-dir",
        str(Path(args.results_dir)),
        "--case-name",
        case_name(args),
        "--center-x",
        str(float(args.center_x)),
        "--center-y",
        str(float(args.center_y)),
        "--outer-a",
        str(float(args.outer_a)),
        "--outer-b",
        str(float(args.outer_b)),
        "--outer-theta",
        str(float(args.outer_theta)),
        "--inner-a",
        str(float(args.inner_a)),
        "--inner-b",
        str(float(args.inner_b)),
        "--inner-theta",
        str(float(args.inner_theta)),
        "--superellipse-power",
        str(float(args.superellipse_power)),
        "--traction-x",
        str(float(args.traction_x)),
        "--traction-y",
        str(float(args.traction_y)),
        "--rho0",
        str(float(args.rho0)),
        "--fem-mode",
        str(args.fem_mode),
        "--run-time",
        str(float(args.run_time)),
        "--ramp-time",
        str(float(args.ramp_time)),
        "--lattice-speed",
        str(float(args.lattice_speed)),
        "--damping-gamma",
        str(float(args.damping_gamma)),
        "--fem-dt",
        str(float(args.fem_dt)),
        "--fem-progress-every",
        str(int(args.fem_progress_every)),
        "--mu",
        str(float(args.mu)),
        "--poisson",
        str(float(args.poisson)),
        "--fem-degree",
        str(int(args.fem_degree)),
        "--geometry-order",
        str(int(args.geometry_order)),
        "--fem-mesh-size",
        str(float(args.fem_mesh_size)),
        "--gmsh-outer-points",
        str(int(args.gmsh_outer_points)),
        "--gmsh-inner-points",
        str(int(args.gmsh_inner_points)),
        "--fem-load-steps",
        str(int(args.fem_load_steps)),
        "--fem-newton-rtol",
        str(float(args.fem_newton_rtol)),
        "--fem-newton-atol",
        str(float(args.fem_newton_atol)),
        "--fem-newton-max-it",
        str(int(args.fem_newton_max_it)),
    ]
    subprocess.run(cmd, check=True)
    if not out.exists():
        raise FileNotFoundError(f"FEM reference was not written: {out}")
    return out


def initial_from_fem(
    fem_npz: Path, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(fem_npz)
    u = np.asarray(data["u"], dtype=np.float64)
    F = np.asarray(data["F"], dtype=np.float64)
    active = data["active"].astype(bool)
    valid = data["valid"].astype(bool)
    U0 = np.zeros((6, n, n), dtype=np.float64)
    U0[2] = 1.0
    U0[5] = 1.0
    for a, (i, j) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1)), start=2):
        values = F[i, j]
        U0[a] = np.where(np.isfinite(values), values, U0[a])
    disp0 = np.where(np.isfinite(u), u, 0.0)
    return U0, disp0, u, F, (active & valid)


def run_lbm_comparison(args: argparse.Namespace) -> dict[str, Any]:
    from curved_lbm.two_d.curved_boundary_warp import (
        BOUNDARY_ID_INNER,
        BOUNDARY_ID_OUTER,
        CURVED_VALUE_CONSTANT_VECTOR,
        CURVED_VALUE_ZERO,
        CurvedBoundarySpec,
        WarpCurvedBoundaryLBM2D,
        boundary_reconstruction_id,
        boundary_reconstruction_name,
        relative_l2,
    )
    from curved_lbm.two_d.vector_nonlinear_elastic_warp import (
        VALUE_SINE2_HOLD,
        WarpHyperelasticMaterial,
    )

    fem_npz = ensure_fem_reference(args)
    n = int(args.n)
    geom = build_lbm_geometry(args)
    material = WarpHyperelasticMaterial.from_poisson(
        "neo_hooke", float(args.poisson), mu=float(args.mu)
    )

    if args.mode in ("relax", "staged"):
        outer_mode = VALUE_SINE2_HOLD
        outer_params = (float(args.traction_x), float(args.traction_y), float(args.ramp_time), 0.0)
        U0 = np.zeros((6, n, n), dtype=np.float64)
        U0[2] = 1.0
        U0[5] = 1.0
        disp0 = np.zeros((2, n, n), dtype=np.float64)
        fem_data = np.load(fem_npz)
        u_fem = np.asarray(fem_data["u"], dtype=np.float64)
        F_fem = np.asarray(fem_data["F"], dtype=np.float64)
        compare_mask = fem_data["active"].astype(bool) & fem_data["valid"].astype(bool)
    else:
        outer_mode = CURVED_VALUE_CONSTANT_VECTOR
        outer_params = (float(args.traction_x), float(args.traction_y))
        U0, disp0, u_fem, F_fem, compare_mask = initial_from_fem(fem_npz, n)
    fem_data_for_velocity = np.load(fem_npz)
    v_fem = (
        np.asarray(fem_data_for_velocity["v"], dtype=np.float64)
        if "v" in fem_data_for_velocity.files
        else np.full((2, n, n), np.nan, dtype=np.float64)
    )

    boundary = CurvedBoundarySpec(
        kind="neumann",
        value_mode=outer_mode,
        params=outer_params,
        id_kinds={BOUNDARY_ID_OUTER: "neumann", BOUNDARY_ID_INNER: "dirichlet"},
        id_value_modes={BOUNDARY_ID_OUTER: outer_mode, BOUNDARY_ID_INNER: CURVED_VALUE_ZERO},
        id_params={BOUNDARY_ID_OUTER: outer_params, BOUNDARY_ID_INNER: ()},
        label="inner clamped Dirichlet velocity, outer constant nominal traction",
    )

    def make_solver(reconstruction: str) -> WarpCurvedBoundaryLBM2D:
        return WarpCurvedBoundaryLBM2D(
            geom,
            material=material,
            boundary=boundary,
            lattice_speed=float(args.lattice_speed),
            collision_omega=float(args.omega),
            collision_model=str(args.collision_model),
            ghost_omega=None if args.ghost_omega is None else float(args.ghost_omega),
            kinetic_filter_strength=float(args.kinetic_filter_strength),
            boundary_reconstruction=str(reconstruction),
            source_mode="damping",
            damping_gamma=float(args.damping_gamma),
            local_compatibility_interval=int(args.local_compatibility_interval),
            local_compatibility_blend=float(args.local_compatibility_blend),
            local_compatibility_interior_only=bool(args.local_compatibility_interior_only),
            local_displacement_interval=int(args.local_displacement_interval),
            local_displacement_sweeps=int(args.local_displacement_sweeps),
            local_displacement_relax=float(args.local_displacement_relax),
            local_displacement_boundary_weight=float(args.local_displacement_boundary_weight),
            local_displacement_boundary_id=BOUNDARY_ID_INNER,
            device=str(args.device),
        )

    stage_summary: dict[str, Any] | None = None
    if args.mode == "staged":
        solver_stage1 = make_solver(str(args.stage1_boundary_reconstruction))
        solver_stage1.initialize_from_numpy(
            U=U0, displacement=disp0, init_order=int(args.init_order)
        )
        solver_stage1.run_until(float(args.stage1_time), check_interval=int(args.check_interval))
        stage_summary = {
            "stage1_boundary_reconstruction": str(args.stage1_boundary_reconstruction),
            "stage1_time": float(solver_stage1.time),
            "stage1_steps": int(solver_stage1.steps),
            "stage1_finite": bool(not solver_stage1.invalid_state()),
            "stage2_boundary_reconstruction": str(args.boundary_reconstruction),
            "stage2_target_time": float(args.stage1_time) + float(args.stage2_time),
            "stage_transition": str(args.stage_transition),
        }
        results_dir_stage = case_output_dir(args)
        results_dir_stage.mkdir(parents=True, exist_ok=True)
        solver_stage1.save_npz(
            results_dir_stage
            / f"superellipse_mixed_stage1_{args.stage1_boundary_reconstruction}_n{n}.npz"
        )
        if str(args.stage_transition) == "restart":
            stage1_state = solver_stage1.system_state()
            stage1_disp = solver_stage1.displacement()
            solver = make_solver(str(args.boundary_reconstruction))
            solver.initialize_from_numpy(
                U=stage1_state,
                displacement=stage1_disp,
                init_order=int(args.stage_restart_init_order),
                initial_boundary_time=float(solver_stage1.time),
            )
            solver.time = float(solver_stage1.time)
            solver.steps = int(solver_stage1.steps)
        else:
            solver = solver_stage1
            solver.boundary_reconstruction = boundary_reconstruction_id(
                str(args.boundary_reconstruction)
            )
            solver.boundary_reconstruction_name = boundary_reconstruction_name(
                solver.boundary_reconstruction
            )
        solver.run_until(
            float(args.stage1_time) + float(args.stage2_time),
            check_interval=int(args.check_interval),
        )
    else:
        solver = make_solver(str(args.boundary_reconstruction))
        solver.initialize_from_numpy(U=U0, displacement=disp0, init_order=int(args.init_order))
        solver.run_until(float(args.run_time), check_interval=int(args.check_interval))

    mask = solver.active_mask() & compare_mask
    u_lbm = solver.displacement()
    F_lbm = solver.deformation_gradient()
    U_lbm = solver.system_state()
    J = solver.deformation_jacobian()

    F_err = F_lbm - F_fem
    v_mag = np.sqrt(U_lbm[0] * U_lbm[0] + U_lbm[1] * U_lbm[1])
    v_compare_mask = mask & np.isfinite(v_fem[0]) & np.isfinite(v_fem[1])
    ids, counts = np.unique(geom.link_boundary_id, return_counts=True)
    count_by_id = {int(k): int(v) for k, v in zip(ids, counts)}

    results_dir = case_output_dir(args)
    summary: dict[str, Any] = {
        "case": "superellipse_shell_mixed_dirichlet_neumann_lbm_vs_fem",
        "case_name": case_name(args),
        "results_dir": str(results_dir),
        "mode": str(args.mode),
        "fem_mode": str(args.fem_mode),
        "n": n,
        "dx": float(geom.dx),
        "device": str(args.device),
        "fem_reference": str(fem_npz),
        "active_nodes": int(np.count_nonzero(solver.active_mask())),
        "compare_nodes": int(np.count_nonzero(mask)),
        "cut_links": int(geom.n_links),
        "outer_cut_links": count_by_id.get(OUTER_ID, 0),
        "inner_cut_links": count_by_id.get(INNER_ID, 0),
        "steps": int(solver.steps),
        "time": float(solver.time),
        "run_time": float(args.run_time),
        "traction": [float(args.traction_x), float(args.traction_y)],
        "lattice_speed": float(args.lattice_speed),
        "collision_omega": float(args.omega),
        "collision_model": str(args.collision_model),
        "ghost_omega": float(args.omega if args.ghost_omega is None else args.ghost_omega),
        "boundary_reconstruction": str(args.boundary_reconstruction),
        "damping_gamma": float(args.damping_gamma),
        "compatibility_projection": {"active": False, "implementation": "not available in newcode"},
        "local_method": {
            "displacement_interval": int(args.local_displacement_interval),
            "displacement_sweeps": int(args.local_displacement_sweeps),
            "displacement_relax": float(args.local_displacement_relax),
            "boundary_weight": float(args.local_displacement_boundary_weight),
            "F_repair_interval": int(args.local_compatibility_interval),
            "F_repair_blend": float(args.local_compatibility_blend),
            "boundary_stencil_radius": int(solver.boundary_stencil_radius),
            "kernel_stencil_radius": 2,
            "displacement_dependency_radius": int(args.local_displacement_sweeps),
            "composed_compatibility_radius": int(args.local_displacement_sweeps) + 2,
            "all_active": bool(not args.local_compatibility_interior_only),
        },
        "material": {
            "model": material.model,
            "mu": material.mu,
            "lambda": material.lam,
            "poisson": material.poisson,
        },
        "stage": stage_summary,
        "rel_l2_u": float(relative_l2(u_lbm - u_fem, u_fem, mask)),
        "rel_l2_F": float(relative_l2(F_err, F_fem, mask)),
        "rel_l2_F_minus_I": float(
            relative_l2(F_err, F_fem - np.eye(2, dtype=np.float64)[:, :, None, None], mask)
        ),
        "abs_l2_u": float(np.sqrt(np.mean(((u_lbm - u_fem)[:, mask]) ** 2))),
        "abs_l2_F": float(np.sqrt(np.mean((F_err[:, :, mask]) ** 2))),
        "max_abs_u_error": float(
            np.max(np.sqrt(np.sum((u_lbm[:, mask] - u_fem[:, mask]) ** 2, axis=0)))
        ),
        "max_abs_F_error": float(np.max(np.sqrt(np.sum(F_err[:, :, mask] ** 2, axis=(0, 1))))),
        "rms_v": float(np.sqrt(np.mean(v_mag[mask] ** 2))),
        "max_abs_v": float(np.max(v_mag[mask])),
        "min_J": float(np.min(J[mask])),
        "max_J": float(np.max(J[mask])),
        "finite": bool(not solver.invalid_state()),
        "outer": vars(outer_spec(args)),
        "inner": vars(inner_spec(args)),
    }
    if int(np.count_nonzero(v_compare_mask)) > 0:
        v_err = U_lbm[:2] - v_fem
        summary.update(
            {
                "velocity_compare_nodes": int(np.count_nonzero(v_compare_mask)),
                "rel_l2_v": float(relative_l2(v_err, v_fem, v_compare_mask)),
                "abs_l2_v": float(np.sqrt(np.mean((v_err[:, v_compare_mask]) ** 2))),
                "max_abs_v_error": float(
                    np.max(np.sqrt(np.sum(v_err[:, v_compare_mask] ** 2, axis=0)))
                ),
                "rms_v_fem": float(np.sqrt(np.mean(np.sum(v_fem[:, v_compare_mask] ** 2, axis=0)))),
                "rms_v_lbm": float(
                    np.sqrt(np.mean(np.sum(U_lbm[:2, v_compare_mask] ** 2, axis=0)))
                ),
            }
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    solver.save_npz(results_dir / f"superellipse_mixed_lbm_n{n}_final.npz")
    np.savez_compressed(
        results_dir / f"superellipse_mixed_comparison_n{n}.npz",
        x=np.load(fem_npz)["x"],
        y=np.load(fem_npz)["y"],
        mask=mask.astype(np.int32),
        u_lbm=u_lbm,
        F_lbm=F_lbm,
        u_fem=u_fem,
        F_fem=F_fem,
        v_lbm=U_lbm[:2],
        v_fem=v_fem,
        outer=np.asarray(
            [
                float(args.center_x),
                float(args.center_y),
                float(args.outer_a),
                float(args.outer_b),
                float(args.outer_theta),
                float(args.superellipse_power),
            ],
            dtype=np.float64,
        ),
        inner=np.asarray(
            [
                float(args.center_x),
                float(args.center_y),
                float(args.inner_a),
                float(args.inner_b),
                float(args.inner_theta),
                float(args.superellipse_power),
            ],
            dtype=np.float64,
        ),
        material=np.asarray([float(args.mu), float(args.poisson)], dtype=np.float64),
        summary=json.dumps(summary),
    )
    summary_path = results_dir / f"superellipse_mixed_summary_n{n}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not bool(args.no_plots):
        comparison_npz = results_dir / f"superellipse_mixed_comparison_n{n}.npz"
        plot_comparison(comparison_npz, results_dir / f"superellipse_mixed_comparison_n{n}.png")
        plot_stress_comparison(comparison_npz, results_dir / f"superellipse_mixed_stress_n{n}.png")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def plot_comparison(npz_path: Path, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    data = np.load(npz_path)
    X = data["x"]
    Y = data["y"]
    mask = data["mask"].astype(bool)
    u_lbm = data["u_lbm"]
    u_fem = data["u_fem"]
    F_lbm = data["F_lbm"]
    F_fem = data["F_fem"]
    outer_values = data["outer"]
    inner_values = data["inner"]
    outer = Superellipse(
        a=float(outer_values[2]),
        b=float(outer_values[3]),
        theta_deg=float(outer_values[4]),
        p=float(outer_values[5]),
        cx=float(outer_values[0]),
        cy=float(outer_values[1]),
    )
    inner = Superellipse(
        a=float(inner_values[2]),
        b=float(inner_values[3]),
        theta_deg=float(inner_values[4]),
        p=float(inner_values[5]),
        cx=float(inner_values[0]),
        cy=float(inner_values[1]),
    )

    fields = [
        (np.sqrt(np.sum(u_fem * u_fem, axis=0)), "|u| FEM", "viridis"),
        (np.sqrt(np.sum(u_lbm * u_lbm, axis=0)), "|u| LBM", "viridis"),
        (np.sqrt(np.sum((u_lbm - u_fem) ** 2, axis=0)), "|u_LBM-u_FEM|", "magma"),
        (np.sqrt(np.sum((F_lbm - F_fem) ** 2, axis=(0, 1))), "||F_LBM-F_FEM||", "magma"),
    ]

    boundary_t = np.linspace(0.0, 2.0 * math.pi, 900)
    outer_xy = np.asarray([outer.point(float(t)) for t in boundary_t])
    inner_xy = np.asarray([inner.point(float(t)) for t in boundary_t])
    xmin = min(float(np.min(outer_xy[:, 0])), float(np.min(inner_xy[:, 0]))) - 0.03
    xmax = max(float(np.max(outer_xy[:, 0])), float(np.max(inner_xy[:, 0]))) + 0.03
    ymin = min(float(np.min(outer_xy[:, 1])), float(np.min(inner_xy[:, 1]))) - 0.03
    ymax = max(float(np.max(outer_xy[:, 1])), float(np.max(inner_xy[:, 1]))) + 0.03
    gx = np.linspace(xmin, xmax, 720)
    gy = np.linspace(ymin, ymax, 720)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    smooth_mask = shell_phi(GX, GY, outer, inner) <= 0.0
    points = np.column_stack([X[mask].ravel(), Y[mask].ravel()])

    fig, axs = plt.subplots(2, 2, figsize=(10.8, 9.2), constrained_layout=True)
    for ax, (field, title, cmap) in zip(axs.ravel(), fields):
        values = field[mask].ravel()
        Z = griddata(points, values, (GX, GY), method="cubic")
        if np.isnan(Z[smooth_mask]).any():
            Z_linear = griddata(points, values, (GX, GY), method="linear")
            Z = np.where(np.isnan(Z), Z_linear, Z)
        if np.isnan(Z[smooth_mask]).any():
            Z_nearest = griddata(points, values, (GX, GY), method="nearest")
            Z = np.where(np.isnan(Z), Z_nearest, Z)
        z = np.ma.array(Z.T, mask=(~smooth_mask).T)
        finite = z.compressed()
        if finite.size:
            lo, hi = np.percentile(finite, [0.5, 99.5])
            if np.isclose(lo, hi):
                pad = max(abs(float(lo)), 1.0) * 0.02
                lo -= pad
                hi += pad
        else:
            lo, hi = 0.0, 1.0
        levels = np.linspace(float(lo), float(hi), 96)
        im = ax.contourf(GX.T, GY.T, z, levels=levels, cmap=cmap, extend="both", antialiased=True)
        ax.plot(outer_xy[:, 0], outer_xy[:, 1], color="black", lw=1.0)
        ax.plot(inner_xy[:, 0], inner_xy[:, 1], color="black", lw=1.0)
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def neo_hookean_cauchy_stress(
    F: np.ndarray, mu: float, poisson: float
) -> tuple[np.ndarray, np.ndarray]:
    lam = 2.0 * float(mu) * float(poisson) / (1.0 - 2.0 * float(poisson))
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    J_safe = np.where(np.abs(J) > 1.0e-14, J, np.nan)
    FinvT = np.empty_like(F)
    FinvT[0, 0] = F[1, 1] / J_safe
    FinvT[0, 1] = -F[1, 0] / J_safe
    FinvT[1, 0] = -F[0, 1] / J_safe
    FinvT[1, 1] = F[0, 0] / J_safe
    P = float(mu) * (F - FinvT) + 0.5 * lam * (J * J - 1.0) * FinvT
    sigma = np.einsum("ia...,ja...->ij...", P, F) / J_safe
    return sigma, P


def cauchy_von_mises_2d(sigma: np.ndarray) -> np.ndarray:
    sxx = sigma[0, 0]
    syy = sigma[1, 1]
    sxy = 0.5 * (sigma[0, 1] + sigma[1, 0])
    return np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))


def field_limits(values: np.ndarray, *, symmetric: bool) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric:
        vmax = float(np.percentile(np.abs(finite), 99.5))
        vmax = max(vmax, 1.0e-14)
        return -vmax, vmax
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if np.isclose(lo, hi):
        pad = max(abs(float(lo)), 1.0) * 0.02
        lo -= pad
        hi += pad
    return float(lo), float(hi)


def plot_stress_comparison(npz_path: Path, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    data = np.load(npz_path)
    X = data["x"]
    Y = data["y"]
    mask = data["mask"].astype(bool)
    F_lbm = data["F_lbm"]
    F_fem = data["F_fem"]
    outer_values = data["outer"]
    inner_values = data["inner"]
    material = data["material"] if "material" in data else np.asarray([1.0, 0.20], dtype=np.float64)
    mu = float(material[0])
    poisson = float(material[1])

    outer = Superellipse(
        a=float(outer_values[2]),
        b=float(outer_values[3]),
        theta_deg=float(outer_values[4]),
        p=float(outer_values[5]),
        cx=float(outer_values[0]),
        cy=float(outer_values[1]),
    )
    inner = Superellipse(
        a=float(inner_values[2]),
        b=float(inner_values[3]),
        theta_deg=float(inner_values[4]),
        p=float(inner_values[5]),
        cx=float(inner_values[0]),
        cy=float(inner_values[1]),
    )

    sigma_lbm, _P_lbm = neo_hookean_cauchy_stress(F_lbm, mu, poisson)
    sigma_fem, _P_fem = neo_hookean_cauchy_stress(F_fem, mu, poisson)
    vm_lbm = cauchy_von_mises_2d(sigma_lbm)
    vm_fem = cauchy_von_mises_2d(sigma_fem)
    sigma_yy_lbm = sigma_lbm[1, 1]
    sigma_yy_fem = sigma_fem[1, 1]
    vm_limits = field_limits(
        np.concatenate([vm_fem[mask].ravel(), vm_lbm[mask].ravel()]), symmetric=False
    )
    sigma_yy_limits = field_limits(
        np.concatenate([sigma_yy_fem[mask].ravel(), sigma_yy_lbm[mask].ravel()]), symmetric=True
    )
    vm_error_limits = field_limits(np.abs(vm_lbm[mask] - vm_fem[mask]), symmetric=False)
    sigma_yy_error_limits = field_limits(
        np.abs(sigma_yy_lbm[mask] - sigma_yy_fem[mask]), symmetric=False
    )

    fields = [
        (vm_fem, "Cauchy vm FEM", "viridis", vm_limits),
        (vm_lbm, "Cauchy vm LBM", "viridis", vm_limits),
        (np.abs(vm_lbm - vm_fem), "|vm_LBM-vm_FEM|", "magma", vm_error_limits),
        (sigma_yy_fem, "sigma_yy FEM", "coolwarm", sigma_yy_limits),
        (sigma_yy_lbm, "sigma_yy LBM", "coolwarm", sigma_yy_limits),
        (np.abs(sigma_yy_lbm - sigma_yy_fem), "|sigma_yy error|", "magma", sigma_yy_error_limits),
    ]

    boundary_t = np.linspace(0.0, 2.0 * math.pi, 900)
    outer_xy = np.asarray([outer.point(float(t)) for t in boundary_t])
    inner_xy = np.asarray([inner.point(float(t)) for t in boundary_t])
    xmin = min(float(np.min(outer_xy[:, 0])), float(np.min(inner_xy[:, 0]))) - 0.03
    xmax = max(float(np.max(outer_xy[:, 0])), float(np.max(inner_xy[:, 0]))) + 0.03
    ymin = min(float(np.min(outer_xy[:, 1])), float(np.min(inner_xy[:, 1]))) - 0.03
    ymax = max(float(np.max(outer_xy[:, 1])), float(np.max(inner_xy[:, 1]))) + 0.03
    gx = np.linspace(xmin, xmax, 720)
    gy = np.linspace(ymin, ymax, 720)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    smooth_mask = shell_phi(GX, GY, outer, inner) <= 0.0
    points = np.column_stack([X[mask].ravel(), Y[mask].ravel()])

    fig, axs = plt.subplots(2, 3, figsize=(15.0, 8.4), constrained_layout=True)
    for ax, (field, title, cmap, limits) in zip(axs.ravel(), fields):
        values = field[mask].ravel()
        Z = griddata(points, values, (GX, GY), method="cubic")
        if np.isnan(Z[smooth_mask]).any():
            Z_linear = griddata(points, values, (GX, GY), method="linear")
            Z = np.where(np.isnan(Z), Z_linear, Z)
        if np.isnan(Z[smooth_mask]).any():
            Z_nearest = griddata(points, values, (GX, GY), method="nearest")
            Z = np.where(np.isnan(Z), Z_nearest, Z)
        z = np.ma.array(Z.T, mask=(~smooth_mask).T)
        lo, hi = limits
        levels = np.linspace(float(lo), float(hi), 96)
        im = ax.contourf(GX.T, GY.T, z, levels=levels, cmap=cmap, extend="both", antialiased=True)
        ax.plot(outer_xy[:, 0], outer_xy[:, 1], color="black", lw=1.0)
        ax.plot(inner_xy[:, 0], inner_xy[:, 1], color="black", lw=1.0)
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=210)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fem-only", action="store_true", help="only generate the FEniCSx reference"
    )
    parser.add_argument(
        "--mode",
        choices=("stationary", "relax", "staged"),
        default="relax",
        help="stationary starts from FEM; relax starts from zero; staged runs two local boundary phases",
    )
    parser.add_argument("--fenicsx-python", default=str(DEFAULT_FENICSX_PYTHON))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n", type=int, default=384)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument(
        "--case-name",
        default="",
        help="subdirectory name under --results-dir; auto-generated when empty",
    )
    parser.add_argument("--force-fem", action="store_true")
    parser.add_argument("--no-plots", action="store_true")

    parser.add_argument("--center-x", type=float, default=0.5)
    parser.add_argument("--center-y", type=float, default=0.5)
    parser.add_argument("--outer-a", type=float, default=0.42)
    parser.add_argument("--outer-b", type=float, default=0.30)
    parser.add_argument("--outer-theta", type=float, default=20.0)
    parser.add_argument("--inner-a", type=float, default=0.16)
    parser.add_argument("--inner-b", type=float, default=0.10)
    parser.add_argument("--inner-theta", type=float, default=-15.0)
    parser.add_argument("--superellipse-power", type=float, default=4.0)

    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--poisson", type=float, default=0.20)
    parser.add_argument("--rho0", type=float, default=1.0)
    parser.add_argument("--traction-x", type=float, default=0.0)
    parser.add_argument("--traction-y", type=float, default=0.125)

    parser.add_argument("--run-time", type=float, default=20.0)
    parser.add_argument("--ramp-time", type=float, default=2.0)
    parser.add_argument("--stage1-time", type=float, default=20.0)
    parser.add_argument("--stage2-time", type=float, default=2.0)
    parser.add_argument(
        "--stage1-boundary-reconstruction",
        default="local_f",
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
            "local_char2",
            "local_f",
            "local_f_bfl",
        ),
    )
    parser.add_argument("--stage-transition", choices=("inplace", "restart"), default="inplace")
    parser.add_argument("--stage-restart-init-order", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--lattice-speed", type=float, default=10.0)
    parser.add_argument("--omega", type=float, default=1.8)
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
            "local_char2",
            "local_f",
            "local_f_bfl",
        ),
        default="local_f",
    )
    parser.add_argument("--damping-gamma", type=float, default=10.0)
    parser.add_argument("--local-compatibility-interval", type=int, default=1)
    parser.add_argument("--local-compatibility-blend", type=float, default=0.1)
    parser.add_argument(
        "--local-compatibility-interior-only",
        dest="local_compatibility_interior_only",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--local-compatibility-all-active",
        dest="local_compatibility_interior_only",
        action="store_false",
    )
    parser.add_argument("--local-displacement-interval", type=int, default=1)
    parser.add_argument("--local-displacement-sweeps", type=int, default=1)
    parser.add_argument("--local-displacement-relax", type=float, default=0.85)
    parser.add_argument("--local-displacement-boundary-weight", type=float, default=20.0)
    parser.add_argument("--init-order", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument(
        "--check-interval",
        type=int,
        default=0,
        help="optional CPU diagnostic interval; 0 keeps the time loop GPU-only",
    )

    parser.add_argument("--fem-degree", type=int, default=2)
    parser.add_argument("--fem-mode", choices=("static", "dynamic"), default="static")
    parser.add_argument(
        "--fem-dt", type=float, default=0.0, help="dynamic FEM time step; default follows LBM dt"
    )
    parser.add_argument("--fem-progress-every", type=int, default=2000)
    parser.add_argument("--geometry-order", type=int, default=2)
    parser.add_argument("--fem-mesh-size", type=float, default=0.018)
    parser.add_argument("--gmsh-outer-points", type=int, default=192)
    parser.add_argument("--gmsh-inner-points", type=int, default=128)
    parser.add_argument("--fem-load-steps", type=int, default=4)
    parser.add_argument("--fem-newton-rtol", type=float, default=1.0e-9)
    parser.add_argument("--fem-newton-atol", type=float, default=1.0e-10)
    parser.add_argument("--fem-newton-max-it", type=int, default=30)
    apply_defaults(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_output_dir(args).mkdir(parents=True, exist_ok=True)
    if args.fem_only:
        solve_fem_reference(args)
    else:
        run_lbm_comparison(args)


if __name__ == "__main__":
    main()
