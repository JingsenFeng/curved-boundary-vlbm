#!/usr/bin/env python3
"""Transient eccentric-hole disk under a smooth nominal-pressure load.

The outer circle is clamped and the inner circle carries time-dependent
nominal pressure. The reported interval covers the rising half of a
sine-squared pulse. The displacement history is compared with dynamic
FEM using zero velocity damping.
"""

from __future__ import annotations

import argparse
from traction_settings import apply_defaults
import csv
import json
import math
from pathlib import Path
from _paths import ROOT, RESULTS, REFERENCES, FENICSX_PYTHON, install_reference
import subprocess
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
NEWCODE_DIR = ROOT / "src"
PAPER3D_DIR = ROOT

CURVED_CODE_ROOT = NEWCODE_DIR
FIGURE_DIR = RESULTS / "figr14_eccentric_pulse"
DEFAULT_OUT_DIR = FIGURE_DIR / "data"
DEFAULT_FEM_REFERENCE = REFERENCES / "pulse" / "eccentric_pulse_fem_history_n128.npz"
DEFAULT_FENICSX_PYTHON = FENICSX_PYTHON

LENGTH_SCALE_CM = 10.0
BOUNDARY_ID_OUTER = 1
BOUNDARY_ID_INNER = 2


def tag_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def case_name(args: argparse.Namespace) -> str:
    requested = str(getattr(args, "case_name", "")).strip()
    if requested:
        return requested
    return (
        f"eccentric_pulse_n{int(args.n)}"
        f"_p{tag_float(args.traction_amplitude)}"
        f"_omega{tag_float(args.omega)}"
        f"_T{tag_float(args.run_time)}"
    )


def case_output_dir(args: argparse.Namespace) -> Path:
    return Path(args.results_dir) / case_name(args)


def reference_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    dx = 1.0 / float(n)
    x = (np.arange(n, dtype=np.float64) + 0.5) * dx
    return np.meshgrid(x, x, indexing="ij")


def shell_phi(x: np.ndarray | float, y: np.ndarray | float, args: argparse.Namespace) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    ro = np.sqrt((x_arr - float(args.outer_cx)) ** 2 + (y_arr - float(args.outer_cy)) ** 2)
    ri = np.sqrt((x_arr - float(args.inner_cx)) ** 2 + (y_arr - float(args.inner_cy)) ** 2)
    return np.maximum(ro - float(args.outer_radius), float(args.inner_radius) - ri)


def active_mask_for_grid(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Y = reference_grid(int(args.n))
    return X, Y, shell_phi(X, Y, args) <= 0.0


def geometry_normal(x: float, y: float, args: argparse.Namespace) -> tuple[float, float]:
    ro = math.hypot(x - float(args.outer_cx), y - float(args.outer_cy))
    ri = math.hypot(x - float(args.inner_cx), y - float(args.inner_cy))
    dist_outer = abs(ro - float(args.outer_radius))
    dist_inner = abs(float(args.inner_radius) - ri)
    if dist_inner <= dist_outer:
        rr = max(ri, 1.0e-30)
        return (float(args.inner_cx) - x) / rr, (float(args.inner_cy) - y) / rr
    rr = max(ro, 1.0e-30)
    return (x - float(args.outer_cx)) / rr, (y - float(args.outer_cy)) / rr


def geometry_boundary_id(x: float, y: float, args: argparse.Namespace) -> int:
    ro = math.hypot(x - float(args.outer_cx), y - float(args.outer_cy))
    ri = math.hypot(x - float(args.inner_cx), y - float(args.inner_cy))
    if abs(float(args.inner_radius) - ri) <= abs(ro - float(args.outer_radius)):
        return BOUNDARY_ID_INNER
    return BOUNDARY_ID_OUTER


def build_lbm_geometry(args: argparse.Namespace) -> Any:
    from curved_lbm.two_d.curved_boundary_warp import CurvedBoundaryGeometry

    return CurvedBoundaryGeometry.from_level_set(
        nx=int(args.n),
        ny=int(args.n),
        length_x=1.0,
        length_y=1.0,
        phi=lambda x, y: float(shell_phi(x, y, args)),
        normal=lambda x, y: geometry_normal(x, y, args),
        boundary_id=lambda x, y, _nx, _ny: geometry_boundary_id(x, y, args),
        label=(
            "eccentric_circle("
            f"Co=({args.outer_cx},{args.outer_cy}),Ro={args.outer_radius},"
            f"Ci=({args.inner_cx},{args.inner_cy}),Ri={args.inner_radius})"
        ),
    )


def build_gmsh_model(args: argparse.Namespace) -> None:
    import gmsh

    mesh_size = float(args.fem_mesh_size)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("eccentric_hole_pulse")

    def circle(cx: float, cy: float, radius: float, reverse: bool) -> list[int]:
        center = gmsh.model.geo.addPoint(cx, cy, 0.0, mesh_size)
        pts = [
            gmsh.model.geo.addPoint(cx + radius, cy, 0.0, mesh_size),
            gmsh.model.geo.addPoint(cx, cy + radius, 0.0, mesh_size),
            gmsh.model.geo.addPoint(cx - radius, cy, 0.0, mesh_size),
            gmsh.model.geo.addPoint(cx, cy - radius, 0.0, mesh_size),
        ]
        if reverse:
            order = [(0, 3), (3, 2), (2, 1), (1, 0)]
        else:
            order = [(0, 1), (1, 2), (2, 3), (3, 0)]
        return [gmsh.model.geo.addCircleArc(pts[a], center, pts[b]) for a, b in order]

    outer_arcs = circle(
        float(args.outer_cx), float(args.outer_cy), float(args.outer_radius), reverse=False
    )
    inner_arcs = circle(
        float(args.inner_cx), float(args.inner_cy), float(args.inner_radius), reverse=True
    )
    outer_loop = gmsh.model.geo.addCurveLoop(outer_arcs)
    inner_loop = gmsh.model.geo.addCurveLoop(inner_arcs)
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [surface], 1)
    gmsh.model.setPhysicalName(2, 1, "material")
    gmsh.model.addPhysicalGroup(1, outer_arcs, BOUNDARY_ID_OUTER)
    gmsh.model.setPhysicalName(1, BOUNDARY_ID_OUTER, "outer_dirichlet")
    gmsh.model.addPhysicalGroup(1, inner_arcs, BOUNDARY_ID_INNER)
    gmsh.model.setPhysicalName(1, BOUNDARY_ID_INNER, "inner_neumann")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.45 * mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    if int(args.geometry_order) > 1:
        gmsh.model.mesh.setOrder(int(args.geometry_order))
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)
        gmsh.model.mesh.optimize("HighOrder")


def pulse_scale(time: float, pulse_time: float) -> float:
    if float(pulse_time) <= 0.0 or float(time) >= float(pulse_time):
        return 0.0
    return math.sin(math.pi * float(time) / float(pulse_time)) ** 2


def neo_hookean_energy(F: np.ndarray, mu: float, poisson: float) -> np.ndarray:
    lam = 2.0 * float(mu) * float(poisson) / (1.0 - 2.0 * float(poisson))
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    J_safe = np.where(J > 1.0e-14, J, np.nan)
    norm2 = np.sum(F * F, axis=(0, 1))
    return (
        0.5 * float(mu) * (norm2 - 2.0)
        - (float(mu) + 0.5 * lam) * np.log(J_safe)
        + 0.25 * lam * (J_safe * J_safe - 1.0)
    )


def probe_average_radius(args: argparse.Namespace) -> float:
    if float(args.probe_average_radius) > 0.0:
        return float(args.probe_average_radius)
    return 2.5 / float(args.n)


def probe_cells(args: argparse.Namespace) -> dict[str, list[tuple[int, int, float, float]]]:
    X, Y, active = active_mask_for_grid(args)
    targets = {
        "left_ligament": (float(args.probe_left_x), float(args.probe_left_y)),
        "upper_ligament": (float(args.probe_upper_x), float(args.probe_upper_y)),
    }
    radius = probe_average_radius(args)
    out: dict[str, list[tuple[int, int, float, float]]] = {}
    for name, (px, py) in targets.items():
        dist2 = (X - px) ** 2 + (Y - py) ** 2
        window = active & (dist2 <= radius * radius)
        if not np.any(window):
            dist2 = np.where(active, dist2, np.inf)
            idx = int(np.argmin(dist2))
            window[:] = False
            window[np.unravel_index(idx, active.shape)] = True
        cells: list[tuple[int, int, float, float]] = []
        for i, j in zip(*np.nonzero(window)):
            cells.append((int(i), int(j), float(X[i, j]), float(Y[i, j])))
        out[name] = cells
    return out


def fem_history_path(args: argparse.Namespace) -> Path:
    return case_output_dir(args) / f"eccentric_pulse_fem_history_n{int(args.n)}.npz"


def lbm_history_path(args: argparse.Namespace) -> Path:
    return case_output_dir(args) / f"eccentric_pulse_lbm_history_n{int(args.n)}.npz"


def find_cells_for_points(domain: Any, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from dolfinx import geometry

    points = np.zeros((points_xy.shape[0], 3), dtype=np.float64)
    points[:, :2] = points_xy
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidate_cells = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, candidate_cells, points)
    cells: list[int] = []
    valid: list[int] = []
    for k in range(points.shape[0]):
        links = colliding_cells.links(k)
        if len(links) > 0:
            valid.append(k)
            cells.append(int(links[0]))
    if len(valid) != points.shape[0]:
        missing = sorted(set(range(points.shape[0])) - set(valid))
        raise RuntimeError(f"could not locate FEM cells for probe points {missing}")
    return points, np.asarray(cells, dtype=np.int32)


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

    outer_facets = facet_tags.find(BOUNDARY_ID_OUTER)
    outer_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, outer_facets)

    mu = float(args.mu)
    poisson = float(args.poisson)
    lam = 2.0 * mu * poisson / (1.0 - 2.0 * poisson)
    rho0 = float(args.rho0)

    I = ufl.Identity(2)
    F = I + ufl.grad(u)
    J = ufl.det(F)
    FinvT = ufl.inv(F).T
    P = mu * (F - FinvT) + 0.5 * lam * (J * J - 1.0) * FinvT
    W = (
        0.5 * mu * (ufl.inner(F, F) - 2.0)
        - (mu + 0.5 * lam) * ufl.ln(J)
        + 0.25 * lam * (J * J - 1.0)
    )
    internal_form = fem.form(ufl.inner(P, ufl.grad(test)) * ufl.dx)
    internal_energy_form = fem.form(W * ufl.dx)

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

    load_amp = fem.Constant(domain, PETSc.ScalarType(0.0))
    normal = ufl.FacetNormal(domain)
    traction = -load_amp * normal
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    external_form = fem.form(ufl.dot(traction, test) * ds(BOUNDARY_ID_INNER))

    u_values = np.asarray(u.x.array, dtype=np.float64).reshape((-1, 2))
    v_values = np.asarray(v_fun.x.array, dtype=np.float64).reshape((-1, 2))
    u_values[:, :] = 0.0
    v_values[:, :] = 0.0
    u_values[outer_dofs, :] = 0.0
    v_values[outer_dofs, :] = 0.0
    u.x.scatter_forward()
    v_fun.x.scatter_forward()

    probes = probe_cells(args)
    probe_names = list(probes.keys())
    probe_counts = [len(probes[name]) for name in probe_names]
    probe_offsets = np.cumsum([0, *probe_counts])
    probe_xy = np.asarray(
        [[cell[2], cell[3]] for name in probe_names for cell in probes[name]], dtype=np.float64
    )
    probe_centers = np.asarray(
        [
            [
                float(np.mean([cell[2] for cell in probes[name]])),
                float(np.mean([cell[3] for cell in probes[name]])),
            ]
            for name in probe_names
        ],
        dtype=np.float64,
    )
    eval_points, eval_cells = find_cells_for_points(domain, probe_xy)

    def acceleration(time: float) -> np.ndarray:
        u.x.scatter_forward()
        fint_vec = petsc.assemble_vector(internal_form)
        fint = np.asarray(fint_vec.array, dtype=np.float64).reshape((-1, 2)).copy()
        fint_vec.destroy()

        load_amp.value = PETSc.ScalarType(
            float(args.traction_amplitude) * pulse_scale(time, float(args.pulse_time))
        )
        fext_vec = petsc.assemble_vector(external_form)
        fext = np.asarray(fext_vec.array, dtype=np.float64).reshape((-1, 2)).copy()
        fext_vec.destroy()

        acc = (fext - fint) / mass - float(args.damping_gamma) * v_values
        acc[outer_dofs, :] = 0.0
        return acc

    time_values: list[float] = []
    u_probe: list[np.ndarray] = []
    v_probe: list[np.ndarray] = []
    kinetic_values: list[float] = []
    internal_values: list[float] = []
    load_values: list[float] = []
    max_u_values: list[float] = []
    max_v_values: list[float] = []

    def record(time: float) -> None:
        u.x.scatter_forward()
        v_fun.x.scatter_forward()
        up_all = np.asarray(u.eval(eval_points, eval_cells), dtype=np.float64)
        vp_all = np.asarray(v_fun.eval(eval_points, eval_cells), dtype=np.float64)
        up = np.asarray(
            [
                np.mean(up_all[probe_offsets[k] : probe_offsets[k + 1]], axis=0)
                for k in range(len(probe_names))
            ],
            dtype=np.float64,
        )
        vp = np.asarray(
            [
                np.mean(vp_all[probe_offsets[k] : probe_offsets[k + 1]], axis=0)
                for k in range(len(probe_names))
            ],
            dtype=np.float64,
        )
        kinetic = 0.5 * float(np.sum(mass * v_values * v_values))
        internal = float(fem.assemble_scalar(internal_energy_form))
        time_values.append(float(time))
        u_probe.append(up)
        v_probe.append(vp)
        kinetic_values.append(kinetic)
        internal_values.append(internal)
        load_values.append(
            float(args.traction_amplitude) * pulse_scale(time, float(args.pulse_time))
        )
        max_u_values.append(float(np.max(np.sqrt(np.sum(u_values * u_values, axis=1)))))
        max_v_values.append(float(np.max(np.sqrt(np.sum(v_values * v_values, axis=1)))))

    t_end = float(args.run_time)
    dt_requested = float(args.fem_dt)
    if dt_requested <= 0.0:
        dt_requested = 0.5 * (1.0 / float(args.n)) / float(args.lattice_speed)
    nsteps = max(1, int(math.ceil(t_end / dt_requested)))
    dt = t_end / float(nsteps)
    record(0.0)
    acc = acceleration(0.0)
    history_every = max(1, int(args.history_every))
    progress_every = max(1, int(args.fem_progress_every))

    for step in range(nsteps):
        t_next = (step + 1) * dt
        u_values[:, :] += dt * v_values + 0.5 * dt * dt * acc
        u_values[outer_dofs, :] = 0.0
        u.x.scatter_forward()
        acc_next = acceleration(t_next)
        v_values[:, :] += 0.5 * dt * (acc + acc_next)
        v_values[outer_dofs, :] = 0.0
        v_fun.x.scatter_forward()
        acc = acc_next
        if not (np.isfinite(u_values).all() and np.isfinite(v_values).all()):
            raise RuntimeError(
                f"dynamic FEM produced non-finite values at step {step + 1}, time {t_next}"
            )
        if (step + 1) % history_every == 0 or step + 1 == nsteps:
            record(t_next)
        if progress_every and ((step + 1) % progress_every == 0 or step + 1 == nsteps):
            print(
                json.dumps(
                    {
                        "fem_dynamic_step": step + 1,
                        "steps": nsteps,
                        "time": t_next,
                        "max_u": float(np.max(np.sqrt(np.sum(u_values * u_values, axis=1)))),
                        "max_v": float(np.max(np.sqrt(np.sum(v_values * v_values, axis=1)))),
                    }
                ),
                flush=True,
            )

    out_npz = fem_history_path(args)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        time=np.asarray(time_values, dtype=np.float64),
        u_probe=np.asarray(u_probe, dtype=np.float64),
        v_probe=np.asarray(v_probe, dtype=np.float64),
        kinetic=np.asarray(kinetic_values, dtype=np.float64),
        internal=np.asarray(internal_values, dtype=np.float64),
        total=np.asarray(kinetic_values, dtype=np.float64)
        + np.asarray(internal_values, dtype=np.float64),
        load=np.asarray(load_values, dtype=np.float64),
        max_u=np.asarray(max_u_values, dtype=np.float64),
        max_v=np.asarray(max_v_values, dtype=np.float64),
        probe_names=np.asarray(probe_names),
        probe_xy=probe_centers,
        probe_counts=np.asarray(probe_counts, dtype=np.int32),
        probe_average_radius=np.asarray(probe_average_radius(args), dtype=np.float64),
    )
    meta = {
        "case": "eccentric_hole_pressure_pulse_dynamic_fem",
        "case_name": case_name(args),
        "dolfinx_version": dolfinx.__version__,
        "n": int(args.n),
        "dt": dt,
        "steps": nsteps,
        "history_every": history_every,
        "fem_degree": degree,
        "geometry_order": int(args.geometry_order),
        "fem_mesh_size": float(args.fem_mesh_size),
        "mass_method": mass_method,
        "material": {"mu": mu, "lambda": lam, "poisson": poisson, "rho0": rho0},
        "traction_amplitude": float(args.traction_amplitude),
        "pulse_time": float(args.pulse_time),
        "run_time": t_end,
        "damping_gamma": float(args.damping_gamma),
        "probe_names": probe_names,
        "probe_xy": probe_centers.tolist(),
        "probe_counts": probe_counts,
        "probe_average_radius": probe_average_radius(args),
        "out": str(out_npz),
    }
    out_npz.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def fem_command_args(args: argparse.Namespace) -> list[str]:
    keys = [
        "n",
        "results_dir",
        "case_name",
        "outer_cx",
        "outer_cy",
        "outer_radius",
        "inner_cx",
        "inner_cy",
        "inner_radius",
        "probe_left_x",
        "probe_left_y",
        "probe_upper_x",
        "probe_upper_y",
        "probe_average_radius",
        "mu",
        "poisson",
        "rho0",
        "traction_amplitude",
        "pulse_time",
        "run_time",
        "lattice_speed",
        "damping_gamma",
        "history_every",
        "fem_dt",
        "fem_progress_every",
        "fem_degree",
        "geometry_order",
        "fem_mesh_size",
    ]
    cmd = [str(Path(__file__).resolve()), "--fem-only"]
    for key in keys:
        cmd.extend([f"--{key.replace('_', '-')}", str(getattr(args, key))])
    return cmd


def ensure_fem_reference(args: argparse.Namespace) -> Path | None:
    if bool(args.no_fem):
        return None
    supplied = str(args.fem_reference).strip()
    if supplied and not bool(args.force_fem):
        reference = Path(supplied).resolve()
        if reference == DEFAULT_FEM_REFERENCE.resolve():
            install_reference("pulse", args, reference, validate_only=True)
        if not reference.exists():
            raise FileNotFoundError(f"supplied FEM reference does not exist: {reference}")
        return reference
    out = fem_history_path(args)
    if out.exists() and not bool(args.force_fem):
        return out
    cmd = [str(args.fenicsx_python), *fem_command_args(args)]
    print("Running FEM reference:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    if not out.exists():
        raise FileNotFoundError(f"FEM reference was not written: {out}")
    return out


def make_solver(args: argparse.Namespace) -> Any:
    from curved_lbm.two_d.curved_boundary_warp import (
        CURVED_VALUE_RADIAL_TRACTION_SINE2_PULSE,
        CURVED_VALUE_ZERO,
        CurvedBoundarySpec,
        WarpCurvedBoundaryLBM2D,
    )
    from curved_lbm.two_d.vector_nonlinear_elastic_warp import WarpHyperelasticMaterial

    geom = build_lbm_geometry(args)
    material = WarpHyperelasticMaterial.from_poisson(
        "neo_hooke", float(args.poisson), mu=float(args.mu)
    )
    inner_params = (-float(args.traction_amplitude), float(args.pulse_time), 0.0)
    boundary = CurvedBoundarySpec(
        kind="neumann",
        value_mode=CURVED_VALUE_RADIAL_TRACTION_SINE2_PULSE,
        params=inner_params,
        id_kinds={BOUNDARY_ID_OUTER: "dirichlet", BOUNDARY_ID_INNER: "neumann"},
        id_value_modes={
            BOUNDARY_ID_OUTER: CURVED_VALUE_ZERO,
            BOUNDARY_ID_INNER: CURVED_VALUE_RADIAL_TRACTION_SINE2_PULSE,
        },
        id_params={BOUNDARY_ID_OUTER: (), BOUNDARY_ID_INNER: inner_params},
        label="outer clamped Dirichlet velocity, inner pressure-pulse nominal traction",
    )
    return WarpCurvedBoundaryLBM2D(
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
        local_displacement_interval=int(args.local_displacement_interval),
        local_displacement_sweeps=int(args.local_displacement_sweeps),
        local_displacement_relax=float(args.local_displacement_relax),
        local_displacement_boundary_weight=float(args.local_displacement_boundary_weight),
        local_displacement_boundary_id=BOUNDARY_ID_OUTER,
        local_compatibility_interval=int(args.local_compatibility_interval),
        local_compatibility_blend=float(args.local_compatibility_blend),
        local_compatibility_interior_only=bool(args.local_compatibility_interior_only),
        device=str(args.device),
    )


def record_lbm_state(
    solver: Any,
    args: argparse.Namespace,
    probes: dict[str, list[tuple[int, int, float, float]]],
) -> dict[str, Any]:
    solver.refresh_current()
    active = solver.active_mask()
    U = solver.system_state()
    F = solver.deformation_gradient()
    u = solver.displacement()
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    vmag2 = U[0] * U[0] + U[1] * U[1]
    W = neo_hookean_energy(F, float(args.mu), float(args.poisson))
    area = float(solver.dx) * float(solver.dy)
    kinetic = 0.5 * float(args.rho0) * float(np.nansum(vmag2[active])) * area
    internal = float(np.nansum(W[active])) * area
    row: dict[str, Any] = {
        "step": int(solver.steps),
        "time": float(solver.time),
        "load": float(args.traction_amplitude)
        * pulse_scale(float(solver.time), float(args.pulse_time)),
        "kinetic": kinetic,
        "internal": internal,
        "total": kinetic + internal,
        "max_u": float(np.nanmax(np.sqrt(np.sum(u[:, active] * u[:, active], axis=0))))
        if np.any(active)
        else 0.0,
        "max_v": float(np.nanmax(np.sqrt(vmag2[active]))) if np.any(active) else 0.0,
        "rms_v": float(np.sqrt(np.nanmean(vmag2[active]))) if np.any(active) else 0.0,
        "min_J": float(np.nanmin(J[active])) if np.any(active) else float("nan"),
        "max_J": float(np.nanmax(J[active])) if np.any(active) else float("nan"),
        "finite": bool(
            np.isfinite(u[:, active]).all()
            and np.isfinite(U[:, active]).all()
            and np.all(J[active] > 0.0)
        )
        if np.any(active)
        else False,
    }
    for name, cells in probes.items():
        ii = np.asarray([cell[0] for cell in cells], dtype=np.int64)
        jj = np.asarray([cell[1] for cell in cells], dtype=np.int64)
        row[f"{name}_ux"] = float(np.mean(u[0, ii, jj]))
        row[f"{name}_uy"] = float(np.mean(u[1, ii, jj]))
        row[f"{name}_vx"] = float(np.mean(U[0, ii, jj]))
        row[f"{name}_vy"] = float(np.mean(U[1, ii, jj]))
    return row


def write_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def compare_lbm_fem(lbm_npz: Path, fem_npz: Path, args: argparse.Namespace) -> dict[str, float]:
    lbm = np.load(lbm_npz)
    fem = np.load(fem_npz)
    t_lbm = np.asarray(lbm["time"], dtype=np.float64)
    t_fem = np.asarray(fem["time"], dtype=np.float64)
    probe_names = [str(v) for v in lbm["probe_names"]]
    metrics: dict[str, float] = {}
    err_values: list[float] = []
    ref_values: list[float] = []
    plotted_err_values: list[float] = []
    plotted_ref_values: list[float] = []
    plotted_components = {"left_ligament": "ux", "upper_ligament": "uy"}
    for pidx, name in enumerate(probe_names):
        for comp, suffix in enumerate(("ux", "uy")):
            y_lbm = np.asarray(lbm[f"{name}_{suffix}"], dtype=np.float64)
            y_fem = np.interp(
                t_lbm, t_fem, np.asarray(fem["u_probe"], dtype=np.float64)[:, pidx, comp]
            )
            err_values.append(float(np.mean((y_lbm - y_fem) ** 2)))
            ref_values.append(float(np.mean(y_fem * y_fem)))
            denom = math.sqrt(max(ref_values[-1], 1.0e-30))
            metrics[f"rms_rel_{name}_{suffix}"] = math.sqrt(err_values[-1]) / denom
            if plotted_components.get(name) == suffix:
                plotted_err_values.append(err_values[-1])
                plotted_ref_values.append(ref_values[-1])
    metrics["rms_rel_all_probe_components"] = math.sqrt(sum(err_values)) / math.sqrt(
        max(sum(ref_values), 1.0e-30)
    )
    metrics["rms_rel_probe_displacement"] = math.sqrt(sum(plotted_err_values)) / math.sqrt(
        max(sum(plotted_ref_values), 1.0e-30)
    )
    if "max_u" in fem.files:
        max_u_lbm = np.asarray(lbm["max_u"], dtype=np.float64)
        max_u_fem = np.interp(t_lbm, t_fem, np.asarray(fem["max_u"], dtype=np.float64))
        metrics["peak_max_u_lbm"] = float(np.nanmax(max_u_lbm))
        metrics["peak_max_u_fem"] = float(np.nanmax(max_u_fem))
        metrics["rms_rel_max_u"] = float(
            np.sqrt(np.nanmean((max_u_lbm - max_u_fem) ** 2))
            / max(np.sqrt(np.nanmean(max_u_fem * max_u_fem)), 1.0e-30)
        )
    return metrics


def run_lbm(args: argparse.Namespace) -> dict[str, Any]:
    fem_npz = ensure_fem_reference(args)
    solver = make_solver(args)
    n = int(args.n)
    U0 = np.zeros((6, n, n), dtype=np.float64)
    U0[2] = 1.0
    U0[5] = 1.0
    solver.initialize_from_numpy(
        U=U0, displacement=np.zeros((2, n, n), dtype=np.float64), init_order=int(args.init_order)
    )

    out_dir = case_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    probes = probe_cells(args)
    probe_names = list(probes.keys())
    snapshot_times = sorted(float(v) for v in args.snapshot_times)
    pending_snapshots = list(snapshot_times)
    snapshot_paths: list[Path] = []
    rows: list[dict[str, Any]] = [record_lbm_state(solver, args, probes)]
    history_every = max(1, int(args.history_every))

    def maybe_save_snapshot() -> None:
        while pending_snapshots and solver.time >= pending_snapshots[0] - 0.5 * solver.dt:
            label = f"t{pending_snapshots.pop(0):.2f}".replace(".", "p")
            path = out_dir / f"eccentric_pulse_lbm_{label}.npz"
            solver.save_npz(path)
            snapshot_paths.append(path)

    target_time = float(args.run_time)
    while solver.time < target_time - 0.5 * solver.dt:
        solver.step()
        maybe_save_snapshot()
        if solver.steps % history_every == 0 or solver.time >= target_time - 0.5 * solver.dt:
            rows.append(record_lbm_state(solver, args, probes))
        if (
            int(args.check_interval) > 0
            and solver.steps % int(args.check_interval) == 0
            and solver.invalid_state()
        ):
            if rows[-1]["step"] != solver.steps:
                rows.append(record_lbm_state(solver, args, probes))
            break
    if rows[-1]["step"] != solver.steps:
        rows.append(record_lbm_state(solver, args, probes))

    final_path = out_dir / f"eccentric_pulse_lbm_final_n{n}.npz"
    solver.save_npz(final_path)
    if not snapshot_paths or snapshot_paths[-1] != final_path:
        snapshot_paths.append(final_path)

    hist_npz = lbm_history_path(args)
    arrays: dict[str, Any] = {
        "time": np.asarray([row["time"] for row in rows], dtype=np.float64),
        "load": np.asarray([row["load"] for row in rows], dtype=np.float64),
        "kinetic": np.asarray([row["kinetic"] for row in rows], dtype=np.float64),
        "internal": np.asarray([row["internal"] for row in rows], dtype=np.float64),
        "total": np.asarray([row["total"] for row in rows], dtype=np.float64),
        "max_u": np.asarray([row["max_u"] for row in rows], dtype=np.float64),
        "max_v": np.asarray([row["max_v"] for row in rows], dtype=np.float64),
        "rms_v": np.asarray([row["rms_v"] for row in rows], dtype=np.float64),
        "min_J": np.asarray([row["min_J"] for row in rows], dtype=np.float64),
        "max_J": np.asarray([row["max_J"] for row in rows], dtype=np.float64),
        "finite": np.asarray([row["finite"] for row in rows], dtype=np.int32),
        "probe_names": np.asarray(probe_names),
        "probe_xy": np.asarray(
            [
                [
                    float(np.mean([cell[2] for cell in probes[name]])),
                    float(np.mean([cell[3] for cell in probes[name]])),
                ]
                for name in probe_names
            ],
            dtype=np.float64,
        ),
        "probe_counts": np.asarray([len(probes[name]) for name in probe_names], dtype=np.int32),
        "probe_average_radius": np.asarray(probe_average_radius(args), dtype=np.float64),
    }
    for name in probe_names:
        for suffix in ("ux", "uy", "vx", "vy"):
            arrays[f"{name}_{suffix}"] = np.asarray(
                [row[f"{name}_{suffix}"] for row in rows], dtype=np.float64
            )
    np.savez_compressed(hist_npz, **arrays)
    write_history_csv(out_dir / f"eccentric_pulse_lbm_history_n{n}.csv", rows)

    ids, counts = np.unique(solver.geometry.link_boundary_id, return_counts=True)
    count_by_id = {int(k): int(v) for k, v in zip(ids, counts)}
    summary: dict[str, Any] = {
        "case": "eccentric_hole_pressure_pulse_lbm",
        "case_name": case_name(args),
        "results_dir": str(out_dir),
        "n": n,
        "dx": float(solver.dx),
        "dt": float(solver.dt),
        "steps": int(solver.steps),
        "time": float(solver.time),
        "active_nodes": int(np.count_nonzero(solver.active_mask())),
        "cut_links": int(solver.geometry.n_links),
        "outer_cut_links": count_by_id.get(BOUNDARY_ID_OUTER, 0),
        "inner_cut_links": count_by_id.get(BOUNDARY_ID_INNER, 0),
        "traction_amplitude": float(args.traction_amplitude),
        "pulse_time": float(args.pulse_time),
        "run_time": float(args.run_time),
        "damping_gamma": float(args.damping_gamma),
        "lattice_speed": float(args.lattice_speed),
        "collision_omega": float(args.omega),
        "collision_model": str(args.collision_model),
        "boundary_reconstruction": str(args.boundary_reconstruction),
        "compatibility_projection": {"active": False, "implementation": "not available in newcode"},
        "local_compatibility": {
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
        "time_stepping": "GPU; CPU transfers are restricted to requested history, snapshots, and final output",
        "material": {
            "model": solver.material.model,
            "mu": solver.material.mu,
            "lambda": solver.material.lam,
            "poisson": solver.material.poisson,
        },
        "geometry": {
            "outer_center": [float(args.outer_cx), float(args.outer_cy)],
            "outer_radius": float(args.outer_radius),
            "inner_center": [float(args.inner_cx), float(args.inner_cy)],
            "inner_radius": float(args.inner_radius),
        },
        "probes": {
            name: {
                "count": len(probes[name]),
                "x": float(np.mean([cell[2] for cell in probes[name]])),
                "y": float(np.mean([cell[3] for cell in probes[name]])),
            }
            for name in probe_names
        },
        "probe_average_radius": probe_average_radius(args),
        "final": rows[-1],
        "finite": bool(rows[-1]["finite"]),
        "fem_reference": None if fem_npz is None else str(fem_npz),
        "lbm_history": str(hist_npz),
        "final_npz": str(final_path),
        "snapshot_npz": [str(path) for path in snapshot_paths],
    }
    if fem_npz is not None:
        summary.update(compare_lbm_fem(hist_npz, fem_npz, args))
    ligament_thickness = (
        float(args.outer_radius)
        - math.hypot(
            float(args.inner_cx) - float(args.outer_cx), float(args.inner_cy) - float(args.outer_cy)
        )
        - float(args.inner_radius)
    )
    summary["minimum_ligament_thickness"] = ligament_thickness
    summary["peak_max_u_over_min_ligament"] = float(
        np.nanmax(arrays["max_u"]) / max(ligament_thickness, 1.0e-30)
    )
    summary["peak_max_u_over_outer_radius"] = float(
        np.nanmax(arrays["max_u"]) / max(float(args.outer_radius), 1.0e-30)
    )
    summary["peak_max_u_over_outer_diameter"] = float(
        np.nanmax(arrays["max_u"]) / max(2.0 * float(args.outer_radius), 1.0e-30)
    )

    summary_path = out_dir / f"eccentric_pulse_summary_n{n}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not bool(args.no_plots):
        plot_case(args)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def circle_points(
    cx: float, cy: float, r: float, count: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * math.pi, count)
    return cx + r * np.cos(theta), cy + r * np.sin(theta)


def configure_plot_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.65,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def compact_sci(value: float, _position: int | None = None) -> str:
    if not np.isfinite(value) or abs(value) < 1.0e-14:
        return "0"
    if 1.0e-3 <= abs(value) < 1.0e4:
        return f"{value:.3g}"
    text = f"{value:.1e}"
    mantissa, exponent = text.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent):+d}"


def panel_label(ax: Any, text: str) -> None:
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )


def plot_snapshot(
    ax: Any,
    path: Path,
    args: argparse.Namespace,
    *,
    title: str,
    vmax_cm: float,
) -> Any:
    data = np.load(path)
    X = np.asarray(data["x"], dtype=np.float64)
    Y = np.asarray(data["y"], dtype=np.float64)
    active = np.asarray(data["active"]).astype(bool)
    u = np.asarray(data["u"], dtype=np.float64)
    umag_cm = LENGTH_SCALE_CM * np.sqrt(u[0] * u[0] + u[1] * u[1])
    cmap = "rainbow"
    sc = ax.scatter(
        LENGTH_SCALE_CM * X[active],
        LENGTH_SCALE_CM * Y[active],
        c=umag_cm[active],
        s=1.5,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax_cm,
        linewidths=0.0,
        rasterized=True,
    )
    ox, oy = circle_points(float(args.outer_cx), float(args.outer_cy), float(args.outer_radius))
    ix, iy = circle_points(float(args.inner_cx), float(args.inner_cy), float(args.inner_radius))
    ax.plot(LENGTH_SCALE_CM * ox, LENGTH_SCALE_CM * oy, color="black", lw=0.7)
    ax.plot(LENGTH_SCALE_CM * ix, LENGTH_SCALE_CM * iy, color="black", lw=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(
        LENGTH_SCALE_CM * (float(args.outer_cx) - float(args.outer_radius) - 0.02),
        LENGTH_SCALE_CM * (float(args.outer_cx) + float(args.outer_radius) + 0.02),
    )
    ax.set_ylim(
        LENGTH_SCALE_CM * (float(args.outer_cy) - float(args.outer_radius) - 0.02),
        LENGTH_SCALE_CM * (float(args.outer_cy) + float(args.outer_radius) + 0.02),
    )
    ax.set_title(title, pad=2.0)
    ax.set_xlabel(r"$X_1$ [cm]", labelpad=1.0)
    ax.set_ylabel(r"$X_2$ [cm]", labelpad=1.0)
    ax.tick_params(top=True, right=True)
    return sc


def plot_case(args: argparse.Namespace) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    configure_plot_style(plt)

    out_dir = case_output_dir(args)
    lbm_npz = lbm_history_path(args)
    if not lbm_npz.exists():
        raise FileNotFoundError(lbm_npz)
    lbm = np.load(lbm_npz)
    summary_path = out_dir / f"eccentric_pulse_summary_n{int(args.n)}.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    recorded_fem = summary.get("fem_reference")
    fem_path = out_dir / Path(recorded_fem).name if recorded_fem else fem_history_path(args)
    if not fem_path.exists():
        fem_path = Path(args.fem_reference) if args.fem_reference else fem_path
    fem = np.load(fem_path) if fem_path.exists() else None
    snapshot_paths = [
        out_dir / Path(v).name
        for v in summary.get("snapshot_npz", [])
        if (out_dir / Path(v).name).exists()
    ]
    selected = snapshot_paths[:3]
    if len(selected) < 3 and snapshot_paths:
        selected = snapshot_paths

    time_lbm = np.asarray(lbm["time"], dtype=np.float64)
    max_u_lbm_cm = LENGTH_SCALE_CM * np.asarray(lbm["max_u"], dtype=np.float64)
    max_u_fem_cm = np.asarray([], dtype=np.float64)
    if fem is not None and "max_u" in fem.files:
        max_u_fem_cm = LENGTH_SCALE_CM * np.asarray(fem["max_u"], dtype=np.float64)
    displacement_scale_cm = max(
        float(np.nanmax(max_u_lbm_cm)) if max_u_lbm_cm.size else 0.0,
        float(np.nanmax(max_u_fem_cm)) if max_u_fem_cm.size else 0.0,
        1.0e-12,
    )
    line_ylim = 1.06 * displacement_scale_cm
    colorbar_vmax = displacement_scale_cm

    fig, axs = plt.subplots(2, 3, figsize=(7.25, 4.25), constrained_layout=False)
    fig.subplots_adjust(left=0.070, right=0.985, bottom=0.105, top=0.950, wspace=0.30, hspace=0.34)
    snapshot_axes = list(axs[0, :])
    cbar_formatter = FuncFormatter(compact_sci)
    for idx, ax in enumerate(snapshot_axes):
        if idx < len(selected):
            t = float(np.load(selected[idx])["time"])
            sc = plot_snapshot(
                ax, selected[idx], args, title=rf"$t={t:.1f}\ \mathrm{{ms}}$", vmax_cm=colorbar_vmax
            )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4.0%", pad=0.028)
            cbar = fig.colorbar(sc, cax=cax, format=cbar_formatter)
            cbar.set_ticks(np.linspace(0.0, colorbar_vmax, 5))
            cbar.update_ticks()
            cbar.set_label(r"$|\mathbf{u}_{\mathrm{LBM}}|$ [cm]", labelpad=0.8)
            cbar.ax.tick_params(direction="in", length=2.5, width=0.55, labelsize=7.0, pad=1.2)
            if idx:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
        else:
            ax.axis("off")
        panel_label(ax, ("(a)", "(b)", "(c)")[idx])

    ax_thin, ax_upper, ax_peak = axs[1, :]

    probe_names = [str(v) for v in lbm["probe_names"]]
    lbm_color = "#1f77b4"
    fem_color = "black"
    if "left_ligament" in probe_names:
        y = LENGTH_SCALE_CM * np.asarray(lbm["left_ligament_ux"], dtype=np.float64)
        ax_thin.plot(time_lbm, y, label="LBM", color=lbm_color, lw=1.05)
        if fem is not None:
            idx = probe_names.index("left_ligament")
            ax_thin.plot(
                np.asarray(fem["time"], dtype=np.float64),
                LENGTH_SCALE_CM * np.asarray(fem["u_probe"], dtype=np.float64)[:, idx, 0],
                label="FEM",
                color=fem_color,
                lw=0.9,
                ls="--",
            )
    peak_time = 0.5 * float(args.pulse_time)
    if time_lbm.size and time_lbm[0] < peak_time < time_lbm[-1] - 1.0e-9:
        ax_thin.axvline(peak_time, color="0.55", lw=0.8, ls=":")
    ax_thin.set_xlabel(r"$t$ [ms]", labelpad=1.0)
    ax_thin.set_ylabel(r"$u_1$ [cm]", labelpad=1.0)

    if "upper_ligament" in probe_names:
        y = LENGTH_SCALE_CM * np.asarray(lbm["upper_ligament_uy"], dtype=np.float64)
        ax_upper.plot(time_lbm, y, label="LBM", color=lbm_color, lw=1.05)
        if fem is not None:
            idx = probe_names.index("upper_ligament")
            ax_upper.plot(
                np.asarray(fem["time"], dtype=np.float64),
                LENGTH_SCALE_CM * np.asarray(fem["u_probe"], dtype=np.float64)[:, idx, 1],
                label="FEM",
                color=fem_color,
                lw=0.9,
                ls="--",
            )
    if time_lbm.size and time_lbm[0] < peak_time < time_lbm[-1] - 1.0e-9:
        ax_upper.axvline(peak_time, color="0.55", lw=0.8, ls=":")
    ax_upper.set_xlabel(r"$t$ [ms]", labelpad=1.0)
    ax_upper.set_ylabel(r"$u_2$ [cm]", labelpad=1.0)

    ax_peak.plot(time_lbm, max_u_lbm_cm, label="LBM", color=lbm_color, lw=1.05)
    if fem is not None:
        ax_peak.plot(
            np.asarray(fem["time"], dtype=np.float64),
            max_u_fem_cm,
            label="FEM",
            color=fem_color,
            lw=0.9,
            ls="--",
        )
    ax_peak.axhline(
        0.2 * LENGTH_SCALE_CM * float(args.outer_radius),
        color="0.45",
        lw=0.75,
        ls=":",
        label=r"$0.2R_o$",
    )
    ax_peak.set_xlabel(r"$t$ [ms]", labelpad=1.0)
    ax_peak.set_ylabel(r"$\max|\mathbf{u}|$ [cm]", labelpad=1.0)

    for ax, letter in zip((ax_thin, ax_upper, ax_peak), ("(d)", "(e)", "(f)")):
        ax.set_ylim(-line_ylim, line_ylim)
        ax.set_xlim(float(np.nanmin(time_lbm)), float(np.nanmax(time_lbm)))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.axhline(0.0, color="0.70", lw=0.45, zorder=0)
        ax.grid(True, color="0.88", lw=0.45)
        ax.tick_params(top=True, right=True)
        panel_label(ax, letter)

    ax_peak.legend(loc="lower right", frameon=False, handlelength=1.6, borderaxespad=0.2)

    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = figure_dir / "figr14_eccentric_pulse.pdf"
    out_png = figure_dir / "figr14_eccentric_pulse.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fem-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-fem", action="store_true")
    parser.add_argument("--force-fem", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--fenicsx-python", default=str(DEFAULT_FENICSX_PYTHON))
    parser.add_argument("--fem-reference", default=str(DEFAULT_FEM_REFERENCE))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--results-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--figure-dir", default=str(FIGURE_DIR))
    parser.add_argument("--case-name", default="")
    parser.add_argument("--n", type=int, default=128)

    parser.add_argument("--outer-cx", type=float, default=0.5)
    parser.add_argument("--outer-cy", type=float, default=0.5)
    parser.add_argument("--outer-radius", type=float, default=0.42)
    parser.add_argument("--inner-cx", type=float, default=0.60)
    parser.add_argument("--inner-cy", type=float, default=0.50)
    parser.add_argument("--inner-radius", type=float, default=0.16)
    parser.add_argument(
        "--probe-left-x", "--probe-thin-x", dest="probe_left_x", type=float, default=0.26
    )
    parser.add_argument(
        "--probe-left-y", "--probe-thin-y", dest="probe_left_y", type=float, default=0.50
    )
    parser.add_argument("--probe-upper-x", type=float, default=0.50)
    parser.add_argument("--probe-upper-y", type=float, default=0.82)
    parser.add_argument(
        "--probe-average-radius",
        type=float,
        default=0.05,
        help="radius of the local cell-averaging window",
    )

    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--poisson", type=float, default=0.20)
    parser.add_argument("--rho0", type=float, default=1.0)
    parser.add_argument("--traction-amplitude", type=float, default=1.5)
    parser.add_argument("--pulse-time", type=float, default=16.0)
    parser.add_argument("--run-time", type=float, default=8.0)
    parser.add_argument("--damping-gamma", type=float, default=0.0)

    parser.add_argument("--lattice-speed", type=float, default=10.0)
    parser.add_argument("--omega", type=float, default=1.9900497512437811)
    parser.add_argument("--collision-model", choices=("bgk", "mrt_ghost"), default="bgk")
    parser.add_argument("--ghost-omega", type=float, default=None)
    parser.add_argument("--kinetic-filter-strength", type=float, default=0.0)
    parser.add_argument(
        "--boundary-reconstruction", choices=("local_f", "local_f_bfl"), default="local_f"
    )
    parser.add_argument("--local-displacement-interval", type=int, default=1)
    parser.add_argument("--local-displacement-sweeps", type=int, default=1)
    parser.add_argument("--local-displacement-relax", type=float, default=0.85)
    parser.add_argument("--local-displacement-boundary-weight", type=float, default=20.0)
    parser.add_argument("--local-compatibility-interval", type=int, default=1)
    parser.add_argument("--local-compatibility-blend", type=float, default=0.1)
    parser.add_argument("--local-compatibility-interior-only", action="store_true", default=False)
    parser.add_argument("--init-order", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--check-interval", type=int, default=0)
    parser.add_argument("--history-every", type=int, default=100)
    parser.add_argument("--snapshot-times", type=float, nargs="*", default=[2.0, 4.0, 8.0])

    parser.add_argument(
        "--fem-dt",
        type=float,
        default=0.0,
        help="dynamic FEM time step; non-positive uses half the LBM time step",
    )
    parser.add_argument("--fem-progress-every", type=int, default=2000)
    parser.add_argument("--fem-degree", type=int, default=2)
    parser.add_argument("--geometry-order", type=int, default=2)
    parser.add_argument("--fem-mesh-size", type=float, default=1.0 / 128.0)
    apply_defaults(parser, dynamic=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_output_dir(args).mkdir(parents=True, exist_ok=True)
    if bool(args.fem_only):
        solve_dynamic_fem_reference(args)
    elif bool(args.plot_only):
        plot_case(args)
    else:
        run_lbm(args)


if __name__ == "__main__":
    main()
