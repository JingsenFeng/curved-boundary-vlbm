#!/usr/bin/env python3
"""Dynamic annulus traction benchmark against FEniCSx.

The domain is a smooth circular annulus.  The inner circle is clamped through a
Dirichlet velocity/displacement condition and the outer circle carries a ramped
nominal traction.  The default traction is radial, T=p n, so this case tests a
different smooth curved boundary from the superellipse shell while keeping the
same mixed Dirichlet/Neumann dynamic comparison.
"""

from __future__ import annotations

import argparse
from traction_settings import apply_defaults
import json
import math
from pathlib import Path
from _paths import ROOT, RESULTS, REFERENCES, FENICSX_PYTHON, install_reference
import subprocess
import sys
from typing import Any

import numpy as np

import superellipse_shell as base


SCRIPT = Path(__file__).resolve()
RECON_ROOT = ROOT / "src"
PACKAGE_ROOT = ROOT
NEWFIG_ROOT = RESULTS
DEFAULT_RESULTS = NEWFIG_ROOT / "fig2_3_annulus_radial" / "data"
DEFAULT_FENICSX_PYTHON = FENICSX_PYTHON

OUTER_ID = 1
INNER_ID = 2


def tag_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def fill_circle_aliases(args: argparse.Namespace) -> None:
    """Populate the superellipse-style fields reused by plotting helpers."""

    args.outer_a = float(args.outer_radius)
    args.outer_b = float(args.outer_radius)
    args.outer_theta = 0.0
    args.inner_a = float(args.inner_radius)
    args.inner_b = float(args.inner_radius)
    args.inner_theta = 0.0
    args.superellipse_power = 2.0


def case_name(args: argparse.Namespace) -> str:
    requested = str(getattr(args, "case_name", "")).strip()
    if requested:
        return requested
    traction_tag = (
        "radial"
        if str(args.traction_mode) == "radial"
        else f"tx{tag_float(args.traction_x)}_ty{tag_float(args.traction_y)}"
    )
    return (
        f"annulus_traction_{traction_tag}_{args.mode}_femdyn_n{int(args.n)}"
        f"_ri{tag_float(args.inner_radius)}_ro{tag_float(args.outer_radius)}"
        f"_bgk{tag_float(args.omega)}_local_T{tag_float(args.run_time)}"
    )


def case_output_dir(args: argparse.Namespace) -> Path:
    return Path(args.results_dir) / case_name(args)


def reference_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    dx = 1.0 / float(n)
    x = (np.arange(n, dtype=np.float64) + 0.5) * dx
    return np.meshgrid(x, x, indexing="ij")


def active_mask_for_grid(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Y = reference_grid(int(args.n))
    r = np.sqrt((X - float(args.center_x)) ** 2 + (Y - float(args.center_y)) ** 2)
    active = (r >= float(args.inner_radius)) & (r <= float(args.outer_radius))
    return X, Y, active


def build_lbm_geometry(args: argparse.Namespace):
    from curved_lbm.two_d.curved_boundary_warp import CurvedBoundaryGeometry

    return CurvedBoundaryGeometry.annulus(
        nx=int(args.n),
        ny=int(args.n),
        inner_radius=float(args.inner_radius),
        outer_radius=float(args.outer_radius),
        center_x=float(args.center_x),
        center_y=float(args.center_y),
    )


def build_gmsh_model(args: argparse.Namespace) -> None:
    import gmsh

    cx = float(args.center_x)
    cy = float(args.center_y)
    ri = float(args.inner_radius)
    ro = float(args.outer_radius)
    mesh_size = float(args.fem_mesh_size)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("annulus_traction_dynamic")

    def point(radius: float, angle: float) -> int:
        return gmsh.model.geo.addPoint(
            cx + radius * math.cos(angle), cy + radius * math.sin(angle), 0.0, mesh_size
        )

    center = gmsh.model.geo.addPoint(cx, cy, 0.0, mesh_size)
    outer_pts = [point(ro, a) for a in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)]
    inner_pts = [point(ri, a) for a in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)]

    outer_arcs = [
        gmsh.model.geo.addCircleArc(outer_pts[0], center, outer_pts[1]),
        gmsh.model.geo.addCircleArc(outer_pts[1], center, outer_pts[2]),
        gmsh.model.geo.addCircleArc(outer_pts[2], center, outer_pts[3]),
        gmsh.model.geo.addCircleArc(outer_pts[3], center, outer_pts[0]),
    ]
    inner_arcs = [
        gmsh.model.geo.addCircleArc(inner_pts[0], center, inner_pts[3]),
        gmsh.model.geo.addCircleArc(inner_pts[3], center, inner_pts[2]),
        gmsh.model.geo.addCircleArc(inner_pts[2], center, inner_pts[1]),
        gmsh.model.geo.addCircleArc(inner_pts[1], center, inner_pts[0]),
    ]
    outer_loop = gmsh.model.geo.addCurveLoop(outer_arcs)
    inner_loop = gmsh.model.geo.addCurveLoop(inner_arcs)
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [surface], 1)
    gmsh.model.setPhysicalName(2, 1, "material")
    gmsh.model.addPhysicalGroup(1, outer_arcs, OUTER_ID)
    gmsh.model.setPhysicalName(1, OUTER_ID, "outer_neumann")
    gmsh.model.addPhysicalGroup(1, inner_arcs, INNER_ID)
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


def fem_output_path(args: argparse.Namespace) -> Path:
    return case_output_dir(args) / f"annulus_traction_fem_n{int(args.n)}.npz"


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
    X, Y, active = active_mask_for_grid(args)
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

    fill_circle_aliases(args)
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
        traction=np.asarray(
            [float(args.traction_x), float(args.traction_y), float(args.traction_amplitude)],
            dtype=np.float64,
        ),
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

    fill_circle_aliases(args)
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
    load_amp = fem.Constant(domain, PETSc.ScalarType(0.0))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    if str(args.traction_mode) == "radial":
        normal = ufl.FacetNormal(domain)
        external_form = fem.form(load_amp * ufl.dot(normal, test) * ds(OUTER_ID))
    else:
        traction = ufl.as_vector((load_x, load_y))
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
        if str(args.traction_mode) == "radial":
            load_amp.value = PETSc.ScalarType(scale * float(args.traction_amplitude))
        else:
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
        "case": "annulus_traction_dynamic_fem",
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
        "traction_mode": str(args.traction_mode),
        "traction": [float(args.traction_x), float(args.traction_y)],
        "traction_amplitude": float(args.traction_amplitude),
        "run_time": t_end,
        "ramp_time": float(args.ramp_time),
        "dt": dt,
        "steps": nsteps,
        "damping_gamma": float(args.damping_gamma),
        "geometry": {
            "center": [float(args.center_x), float(args.center_y)],
            "inner_radius": float(args.inner_radius),
            "outer_radius": float(args.outer_radius),
        },
        "progress": force_history,
        "out": str(out_npz),
    }
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def ensure_fem_reference(args: argparse.Namespace) -> Path:
    out = fem_output_path(args)
    if not bool(args.force_fem):
        install_reference("annulus", args, out)
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
        "--inner-radius",
        str(float(args.inner_radius)),
        "--outer-radius",
        str(float(args.outer_radius)),
        "--traction-mode",
        str(args.traction_mode),
        "--traction-amplitude",
        str(float(args.traction_amplitude)),
        "--traction-x",
        str(float(args.traction_x)),
        "--traction-y",
        str(float(args.traction_y)),
        "--rho0",
        str(float(args.rho0)),
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
    ]
    subprocess.run(cmd, check=True)
    if not out.exists():
        raise FileNotFoundError(f"FEM reference was not written: {out}")
    return out


def run_lbm_comparison(args: argparse.Namespace) -> dict[str, Any]:
    from curved_lbm.two_d.curved_boundary_warp import (
        BOUNDARY_ID_INNER,
        BOUNDARY_ID_OUTER,
        CURVED_VALUE_RADIAL_TRACTION_SINE2_HOLD,
        CURVED_VALUE_ZERO,
        CurvedBoundarySpec,
        WarpCurvedBoundaryLBM2D,
        relative_l2,
    )
    from curved_lbm.two_d.vector_nonlinear_elastic_warp import (
        VALUE_SINE2_HOLD,
        WarpHyperelasticMaterial,
    )

    fill_circle_aliases(args)
    fem_npz = ensure_fem_reference(args)
    n = int(args.n)
    geom = build_lbm_geometry(args)
    material = WarpHyperelasticMaterial.from_poisson(
        "neo_hooke", float(args.poisson), mu=float(args.mu)
    )

    if str(args.traction_mode) == "radial":
        outer_mode = CURVED_VALUE_RADIAL_TRACTION_SINE2_HOLD
        outer_params = (float(args.traction_amplitude), float(args.ramp_time), 0.0)
    else:
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
    v_fem = (
        np.asarray(fem_data["v"], dtype=np.float64)
        if "v" in fem_data.files
        else np.full((2, n, n), np.nan, dtype=np.float64)
    )

    boundary = CurvedBoundarySpec(
        kind="neumann",
        value_mode=outer_mode,
        params=outer_params,
        id_kinds={BOUNDARY_ID_OUTER: "neumann", BOUNDARY_ID_INNER: "dirichlet"},
        id_value_modes={BOUNDARY_ID_OUTER: outer_mode, BOUNDARY_ID_INNER: CURVED_VALUE_ZERO},
        id_params={BOUNDARY_ID_OUTER: outer_params, BOUNDARY_ID_INNER: ()},
        label=f"annulus inner clamped Dirichlet, outer {args.traction_mode} ramped traction",
    )

    solver = WarpCurvedBoundaryLBM2D(
        geom,
        material=material,
        boundary=boundary,
        lattice_speed=float(args.lattice_speed),
        collision_omega=float(args.omega),
        collision_model=str(args.collision_model),
        ghost_omega=None if args.ghost_omega is None else float(args.ghost_omega),
        kinetic_filter_strength=float(args.kinetic_filter_strength),
        boundary_reconstruction=str(args.boundary_reconstruction),
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
    solver.initialize_from_numpy(U=U0, displacement=disp0, init_order=int(args.init_order))
    mask_for_history = solver.active_mask() & compare_mask
    lbm_history: list[dict[str, float | int | bool]] = []

    def record_lbm_history() -> None:
        solver.refresh_current()
        u_now = solver.displacement()
        F_now = solver.deformation_gradient()
        U_now = solver.system_state()
        J_now = solver.deformation_jacobian()
        v_now = np.sqrt(U_now[0] * U_now[0] + U_now[1] * U_now[1])
        if np.any(mask_for_history):
            F_now_err = F_now - F_fem
            lbm_history.append(
                {
                    "step": int(solver.steps),
                    "time": float(solver.time),
                    "max_u": float(
                        np.max(np.sqrt(np.sum(u_now[:, mask_for_history] ** 2, axis=0)))
                    ),
                    "max_v": float(np.max(v_now[mask_for_history])),
                    "rms_v": float(np.sqrt(np.mean(v_now[mask_for_history] ** 2))),
                    "rel_l2_u_to_final_fem": float(
                        relative_l2(u_now - u_fem, u_fem, mask_for_history)
                    ),
                    "rel_l2_F_to_final_fem": float(relative_l2(F_now_err, F_fem, mask_for_history)),
                    "rel_l2_F_minus_I_to_final_fem": float(
                        relative_l2(
                            F_now_err,
                            F_fem - np.eye(2, dtype=np.float64)[:, :, None, None],
                            mask_for_history,
                        )
                    ),
                    "min_J": float(np.min(J_now[mask_for_history])),
                    "max_J": float(np.max(J_now[mask_for_history])),
                    "finite": bool(
                        np.isfinite(u_now[:, mask_for_history]).all()
                        and np.isfinite(U_now[:, mask_for_history]).all()
                    ),
                }
            )

    progress_every = int(args.lbm_progress_every)
    if progress_every > 0:
        record_lbm_history()
        target_time = float(args.run_time)
        while solver.time < target_time - 0.5 * solver.dt:
            solver.step()
            if (
                int(args.check_interval) > 0
                and solver.steps % int(args.check_interval) == 0
                and solver.invalid_state()
            ):
                break
            if solver.steps % progress_every == 0 or solver.time >= target_time - 0.5 * solver.dt:
                record_lbm_history()
    else:
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
        "case": "annulus_traction_dynamic_lbm_vs_fem",
        "case_name": case_name(args),
        "results_dir": str(results_dir),
        "mode": str(args.mode),
        "fem_mode": "dynamic",
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
        "traction_mode": str(args.traction_mode),
        "traction": [float(args.traction_x), float(args.traction_y)],
        "traction_amplitude": float(args.traction_amplitude),
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
        "geometry": {
            "center": [float(args.center_x), float(args.center_y)],
            "inner_radius": float(args.inner_radius),
            "outer_radius": float(args.outer_radius),
        },
        "lbm_history": lbm_history,
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
    solver.save_npz(results_dir / f"annulus_traction_lbm_n{n}_final.npz")
    np.savez_compressed(
        results_dir / f"annulus_traction_comparison_n{n}.npz",
        x=fem_data["x"],
        y=fem_data["y"],
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
                float(args.outer_radius),
                float(args.outer_radius),
                0.0,
                2.0,
            ],
            dtype=np.float64,
        ),
        inner=np.asarray(
            [
                float(args.center_x),
                float(args.center_y),
                float(args.inner_radius),
                float(args.inner_radius),
                0.0,
                2.0,
            ],
            dtype=np.float64,
        ),
        material=np.asarray([float(args.mu), float(args.poisson)], dtype=np.float64),
        summary=json.dumps(summary),
    )
    summary_path = results_dir / f"annulus_traction_summary_n{n}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if lbm_history:
        history_path = results_dir / f"annulus_traction_lbm_history_n{n}.json"
        history_path.write_text(json.dumps(lbm_history, indent=2), encoding="utf-8")
        csv_path = results_dir / f"annulus_traction_lbm_history_n{n}.csv"
        columns = list(lbm_history[0].keys())
        csv_lines = [",".join(columns)]
        for row in lbm_history:
            csv_lines.append(",".join(str(row[col]) for col in columns))
        csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    if not bool(args.no_plots):
        comparison_npz = results_dir / f"annulus_traction_comparison_n{n}.npz"
        base.plot_comparison(comparison_npz, results_dir / f"annulus_traction_comparison_n{n}.png")
        base.plot_stress_comparison(
            comparison_npz, results_dir / f"annulus_traction_stress_n{n}.png"
        )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fem-only", action="store_true", help="only generate the FEniCSx dynamic reference"
    )
    parser.add_argument("--mode", choices=("relax",), default="relax", help="dynamic run from zero")
    parser.add_argument("--fenicsx-python", default=str(DEFAULT_FENICSX_PYTHON))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n", type=int, default=196)
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
    parser.add_argument("--inner-radius", type=float, default=0.18)
    parser.add_argument("--outer-radius", type=float, default=0.42)
    parser.add_argument("--traction-mode", choices=("radial", "cartesian"), default="radial")
    parser.add_argument(
        "--traction-amplitude",
        type=float,
        default=1.0,
        help="radial traction amplitude for traction-mode=radial",
    )
    parser.add_argument("--traction-x", type=float, default=0.0)
    parser.add_argument("--traction-y", type=float, default=0.02)

    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--poisson", type=float, default=0.20)
    parser.add_argument("--rho0", type=float, default=1.0)
    parser.add_argument("--run-time", type=float, default=5.0)
    parser.add_argument("--ramp-time", type=float, default=2.0)
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
    parser.add_argument("--local-displacement-boundary-weight", type=float, default=100.0)
    parser.add_argument("--init-order", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument(
        "--check-interval",
        type=int,
        default=0,
        help="optional CPU diagnostic interval; 0 keeps the time loop GPU-only",
    )
    parser.add_argument(
        "--lbm-progress-every",
        type=int,
        default=0,
        help="record LBM convergence history every N steps; 0 disables",
    )

    parser.add_argument(
        "--fem-dt", type=float, default=0.0, help="dynamic FEM time step; default follows LBM dt"
    )
    parser.add_argument("--fem-progress-every", type=int, default=2000)
    parser.add_argument("--fem-degree", type=int, default=2)
    parser.add_argument("--geometry-order", type=int, default=2)
    parser.add_argument("--fem-mesh-size", type=float, default=0.010)
    apply_defaults(parser)
    args = parser.parse_args()
    fill_circle_aliases(args)
    return args


def main() -> None:
    args = parse_args()
    case_output_dir(args).mkdir(parents=True, exist_ok=True)
    if bool(args.fem_only):
        solve_dynamic_fem_reference(args)
    else:
        run_lbm_comparison(args)


if __name__ == "__main__":
    main()
