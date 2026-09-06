"""Two-dimensional curved-boundary D2Q4x6 hyperelastic solver.

The reference material occupies phi(X) <= 0. Velocity Dirichlet cut links
use the vectorial population-pair identities and BFL interpolation.
The local_f traction reconstruction predicts tangential deformation,
solves P(F_b)n = T for the normal image, and recovers incoming populations
using local characteristic derivatives with the actual BGK relaxation
and source coefficients. Local displacement and deformation-gradient
updates use centered or second-order one-sided gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

from .vector_nonlinear_elastic_warp import (
    Q,
    NCOMP,
    FQ,
    BC_DIRICHLET,
    BC_NEUMANN,
    VALUE_ZERO,
    VALUE_CONSTANT,
    VALUE_AFFINE_VELOCITY,
    SOURCE_NONE,
    SOURCE_CONSTANT,
    SOURCE_DAMPING,
    MATERIAL_SVK,
    MATERIAL_NEO_HOOKE,
    WarpHyperelasticMaterial,
    MaterialName,
    default_device,
    _CX,
    _CY,
    CX_I,
    CY_I,
    PI,
    _as_warp,
    _equilibrium_np,
    _second_order_populations_np,
    _second_order_populations_masked_np,
    _source_np,
    _piola_vec,
    _state_comp,
    _flux_x_comp,
    _flux_y_comp,
    _equilibrium_comp,
    _source_comp,
    _reset_invalid_kernel,
    _clear_kernel,
    _clamp_j,
    _boundary_value_comp,
    _first_piola_np,
)

# Curved-boundary value modes.  The first modes intentionally reuse the base
# modes where the semantic meaning is identical.
CURVED_VALUE_ZERO = VALUE_ZERO
CURVED_VALUE_CONSTANT_VECTOR = VALUE_CONSTANT
CURVED_VALUE_AFFINE_VELOCITY = VALUE_AFFINE_VELOCITY
CURVED_VALUE_PIOLA_NORMAL = 101  # traction T=P0 n, params=(P11,P12,P21,P22)
CURVED_VALUE_RADIAL_TRACTION = 102  # traction T=amp n
CURVED_VALUE_RADIAL_TRACTION_SINE2_HOLD = 103  # params=(amp,ramp_time,denom), traction T=amp*s(t)*n
CURVED_VALUE_RADIAL_TRACTION_SINE2_PULSE = (
    104  # params=(amp,pulse_time,unused), traction T=amp*sin^2(pi*t/Tp)*n for t<Tp, else 0
)

BOUNDARY_RECON_BFL = 0
BOUNDARY_RECON_QBFL = 1
BOUNDARY_RECON_LSQ_BFL = 2
BOUNDARY_RECON_REG_BFL = 3
BOUNDARY_RECON_RNEE = 4
BOUNDARY_RECON_PAIR_NEE = 5
BOUNDARY_RECON_SAME_NEE = 6
BOUNDARY_RECON_POLY = 7
BOUNDARY_RECON_LOWQ_REG_BFL = 8
BOUNDARY_RECON_COMPAT_BFL = 9
BOUNDARY_RECON_COMPAT_NEE = 10
BOUNDARY_RECON_LOCAL_CHAR2 = 11
BOUNDARY_RECON_LOCAL_F = 12
BOUNDARY_RECON_LOCAL_F_BFL = 13

NORMAL_JACOBIAN_FINITE_DIFFERENCE = 0
NORMAL_JACOBIAN_ACOUSTIC = 1
# Internal dispatch value; the public Jacobian option still describes Newton.
NORMAL_SOLVER_ENERGY = 2


def neumann_solver_name(name: str | None, material: WarpHyperelasticMaterial) -> str:
    value = os.environ.get("XLB_NEUMANN_SOLVER", "newton") if name is None else str(name)
    key = value.strip().lower()
    if key not in ("newton", "energy"):
        raise ValueError(f"unsupported Neumann solver: {value!r}")
    if key == "energy" and (
        material.id != MATERIAL_NEO_HOOKE
        or not np.isfinite(material.mu)
        or not np.isfinite(material.lam)
        or material.mu <= 0.0
        or material.lam < 0.0
    ):
        raise ValueError(
            "energy Neumann solver requires quadratic-volumetric neo_hooke, mu > 0, lambda >= 0"
        )
    return key


def normal_jacobian_id(name: str | None = None) -> int:
    value = os.environ.get("XLB_NORMAL_JACOBIAN", "acoustic") if name is None else str(name)
    key = value.strip().lower().replace("-", "_")
    if key in ("finite_difference", "finite", "fd"):
        return NORMAL_JACOBIAN_FINITE_DIFFERENCE
    if key in ("acoustic", "acoustic_tensor", "analytic", "analytical"):
        return NORMAL_JACOBIAN_ACOUSTIC
    raise ValueError(f"unsupported normal Jacobian mode: {value!r}")


def normal_jacobian_name(mode: int) -> str:
    if int(mode) == NORMAL_JACOBIAN_FINITE_DIFFERENCE:
        return "finite_difference"
    if int(mode) == NORMAL_JACOBIAN_ACOUSTIC:
        return "acoustic"
    return f"unknown({mode})"


MAX_LSQ_STENCIL = 24


def boundary_reconstruction_id(name: str | int) -> int:
    if isinstance(name, int):
        return int(name)
    name = str(name).strip().lower()
    if name == "bfl":
        return BOUNDARY_RECON_BFL
    if name == "qbfl":
        return BOUNDARY_RECON_QBFL
    if name == "lsq_bfl" or name == "normal_probe_bfl":
        return BOUNDARY_RECON_LSQ_BFL
    if name == "regularized_bfl":
        return BOUNDARY_RECON_REG_BFL
    if name == "regularized_nee":
        return BOUNDARY_RECON_RNEE
    if name == "pair_nee":
        return BOUNDARY_RECON_PAIR_NEE
    if name == "same_nee":
        return BOUNDARY_RECON_SAME_NEE
    if name == "poly_ibb":
        return BOUNDARY_RECON_POLY
    if name == "lowq_reg_bfl":
        return BOUNDARY_RECON_LOWQ_REG_BFL
    if name == "compat_bfl":
        return BOUNDARY_RECON_COMPAT_BFL
    if name == "compat_nee":
        return BOUNDARY_RECON_COMPAT_NEE
    if name in ("local_char2", "char2", "characteristic2", "second_order", "strict2"):
        return BOUNDARY_RECON_LOCAL_CHAR2
    # ``local_f`` is the single production definition: the robust compact
    # F-characteristic closure used by every 2-D mixed-boundary case.
    if name in ("local_f", "fully_local"):
        return BOUNDARY_RECON_LOCAL_F
    # The cheaper one-chain BFL closure is retained only for explicit
    # diagnostics; it is not a production default because it fails the strong
    # transient pulse case.
    if name in ("local_f_bfl", "single_history"):
        return BOUNDARY_RECON_LOCAL_F_BFL
    raise ValueError(f"unsupported boundary_reconstruction: {name}")


def boundary_reconstruction_name(rid: int) -> str:
    if rid == BOUNDARY_RECON_BFL:
        return "bfl"
    if rid == BOUNDARY_RECON_QBFL:
        return "qbfl"
    if rid == BOUNDARY_RECON_LSQ_BFL:
        return "lsq_bfl"
    if rid == BOUNDARY_RECON_REG_BFL:
        return "regularized_bfl"
    if rid == BOUNDARY_RECON_RNEE:
        return "regularized_nee"
    if rid == BOUNDARY_RECON_PAIR_NEE:
        return "pair_nee"
    if rid == BOUNDARY_RECON_SAME_NEE:
        return "same_nee"
    if rid == BOUNDARY_RECON_POLY:
        return "poly_ibb"
    if rid == BOUNDARY_RECON_LOWQ_REG_BFL:
        return "lowq_reg_bfl"
    if rid == BOUNDARY_RECON_COMPAT_BFL:
        return "compat_bfl"
    if rid == BOUNDARY_RECON_COMPAT_NEE:
        return "compat_nee"
    if rid == BOUNDARY_RECON_LOCAL_CHAR2:
        return "local_char2"
    if rid == BOUNDARY_RECON_LOCAL_F:
        return "local_f"
    if rid == BOUNDARY_RECON_LOCAL_F_BFL:
        return "local_f_bfl"
    return f"unknown_{rid}"


def boundary_reconstruction_stencil_radius(rid: int) -> int:
    """Maximum grid-cell radius read by one boundary-reconstruction kernel."""
    if rid in (
        BOUNDARY_RECON_LSQ_BFL,
        BOUNDARY_RECON_COMPAT_BFL,
        BOUNDARY_RECON_COMPAT_NEE,
        BOUNDARY_RECON_LOCAL_CHAR2,
        BOUNDARY_RECON_LOCAL_F,
    ):
        return 3
    if rid in (BOUNDARY_RECON_QBFL, BOUNDARY_RECON_POLY):
        return 2
    return 1


MAX_BOUNDARY_IDS = 8
BOUNDARY_ID_OUTER = 1
BOUNDARY_ID_INNER = 2

BoundaryKind = Literal["dirichlet", "neumann"]


@dataclass(frozen=True)
class CurvedBoundarySpec:
    """Physical boundary data for all cut links of an implicit boundary.

    Parameters
    ----------
    kind:
        ``"dirichlet"`` prescribes velocity.  ``"neumann"`` prescribes nominal
        traction per reference length/area.
    value_mode:
        Integer mode interpreted inside the Warp kernel.  Supported built-ins:
        ``CURVED_VALUE_ZERO``, ``CURVED_VALUE_CONSTANT_VECTOR``,
        ``CURVED_VALUE_AFFINE_VELOCITY``, ``CURVED_VALUE_PIOLA_NORMAL``, and
        ``CURVED_VALUE_RADIAL_TRACTION``.
    params:
        Up to eight scalar parameters.  Unused slots are zero-padded.
    id_kinds, id_value_modes, id_params:
        Optional per-boundary-id overrides.  Boundary id 0 is reserved for the
        default rule.  Embedded geometries such as annuli use ids 1 and 2 for
        outer and inner boundary components.
    label:
        Optional description stored in output metadata.
    """

    kind: BoundaryKind = "dirichlet"
    value_mode: int = CURVED_VALUE_ZERO
    params: tuple[float, ...] = ()
    id_kinds: dict[int, BoundaryKind] | None = None
    id_value_modes: dict[int, int] | None = None
    id_params: dict[int, tuple[float, ...]] | None = None
    label: str = ""

    @property
    def kind_id(self) -> int:
        if self.kind == "dirichlet":
            return BC_DIRICHLET
        if self.kind == "neumann":
            return BC_NEUMANN
        raise ValueError(f"unsupported curved boundary kind: {self.kind}")

    @property
    def params8(self) -> np.ndarray:
        out = np.zeros(8, dtype=np.float64)
        for i, value in enumerate(self.params[:8]):
            out[i] = float(value)
        return out

    def tables(self, max_ids: int = MAX_BOUNDARY_IDS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return per-boundary-id kind, value-mode, and parameter tables."""

        kind = np.full(max_ids, self.kind_id, dtype=np.int32)
        mode = np.full(max_ids, int(self.value_mode), dtype=np.int32)
        params = np.tile(self.params8[None, :], (max_ids, 1))

        if self.id_kinds is not None:
            for bid, value in self.id_kinds.items():
                if 0 <= int(bid) < max_ids:
                    if value == "dirichlet":
                        kind[int(bid)] = BC_DIRICHLET
                    elif value == "neumann":
                        kind[int(bid)] = BC_NEUMANN
                    else:
                        raise ValueError(f"unsupported boundary kind for id {bid}: {value}")
        if self.id_value_modes is not None:
            for bid, value in self.id_value_modes.items():
                if 0 <= int(bid) < max_ids:
                    mode[int(bid)] = int(value)
        if self.id_params is not None:
            for bid, values in self.id_params.items():
                if 0 <= int(bid) < max_ids:
                    params[int(bid), :] = 0.0
                    for i, value in enumerate(tuple(values)[:8]):
                        params[int(bid), i] = float(value)
        return kind, mode, np.ascontiguousarray(params)


@dataclass(frozen=True)
class CircleDomain:
    """Circle level-set helper for validation and examples."""

    center_x: float = 0.5
    center_y: float = 0.5
    radius: float = 0.38

    def phi(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
        return (
            np.sqrt((np.asarray(x) - self.center_x) ** 2 + (np.asarray(y) - self.center_y) ** 2)
            - self.radius
        )

    def normal(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        rx = np.asarray(x) - self.center_x
        ry = np.asarray(y) - self.center_y
        r = np.sqrt(rx * rx + ry * ry)
        r = np.maximum(r, np.finfo(float).eps)
        return rx / r, ry / r


def _build_boundary_lsq_weights(
    *,
    active: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    link_i: np.ndarray,
    link_j: np.ndarray,
    link_xb: np.ndarray,
    link_yb: np.ndarray,
    link_nx: np.ndarray,
    link_ny: np.ndarray,
    dx: float,
    max_stencil: int = MAX_LSQ_STENCIL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build compact normal-probe and surface-tangent reconstruction weights.

    For each cut link, two off-boundary probe points are placed along the
    inward normal, ``X1=Xb-h1*n`` and ``X2=Xb-h2*n``.  Bilinear interpolation at
    these probes is then linearly extrapolated back to the physical cut point.
    The construction is more geometrically aligned than one-link extrapolation
    and less extrapolatory than a full quadratic MLS stencil.

    The returned weights evaluate

        U_b^- ≈ (h2 U(X1) - h1 U(X2))/(h2-h1),

    with fallbacks to a single probe or to the legacy link predictor if the
    bilinear support leaves the active set.
    """
    n_links = int(link_i.size)
    nx, ny = active.shape
    out_i = np.full((n_links, max_stencil), -1, dtype=np.int32)
    out_j = np.full((n_links, max_stencil), -1, dtype=np.int32)
    out_w = np.zeros((n_links, max_stencil), dtype=np.float64)
    out_wt = np.zeros((n_links, max_stencil), dtype=np.float64)
    out_wx = np.zeros((n_links, max_stencil), dtype=np.float64)
    out_wy = np.zeros((n_links, max_stencil), dtype=np.float64)
    valid = np.zeros(n_links, dtype=np.int32)
    dx = float(dx)
    x0_grid = float(x_coords[0]) if len(x_coords) else 0.5 * dx
    y0_grid = float(y_coords[0]) if len(y_coords) else 0.5 * dx

    def bilinear_weights(xp: float, yp: float):
        rx = (xp - x0_grid) / dx
        ry = (yp - y0_grid) / dx
        i0 = math.floor(rx)
        j0 = math.floor(ry)
        sx = rx - i0
        sy = ry - j0
        if i0 < 0 or j0 < 0 or i0 + 1 >= nx or j0 + 1 >= ny:
            return None
        nodes = [
            (i0, j0, (1.0 - sx) * (1.0 - sy)),
            (i0 + 1, j0, sx * (1.0 - sy)),
            (i0, j0 + 1, (1.0 - sx) * sy),
            (i0 + 1, j0 + 1, sx * sy),
        ]
        for ii, jj, _w in nodes:
            if active[ii, jj] == 0:
                return None
        return nodes

    # Ordered from most accurate to most robust.  h values are in cell units.
    probe_pairs = ((0.75, 1.75), (1.0, 2.0), (1.25, 2.5), (1.5, 3.0))
    single_probes = (0.75, 1.0, 1.25, 1.5, 2.0)
    eye6 = np.eye(6, dtype=np.float64)

    def tangent_derivative_weights(ell: int, xb: float, yb: float, nx_b: float, ny_b: float):
        tx = -ny_b
        ty = nx_b
        i0 = int(link_i[ell])
        j0 = int(link_j[ell])
        cand: list[tuple[float, int, int, float, float]] = []
        radius_cells = 3
        for ii in range(max(0, i0 - radius_cells), min(nx, i0 + radius_cells + 1)):
            for jj in range(max(0, j0 - radius_cells), min(ny, j0 + radius_cells + 1)):
                if active[ii, jj] == 0:
                    continue
                rx = (float(x_coords[ii]) - xb) / dx
                ry = (float(y_coords[jj]) - yb) / dx
                r2 = rx * rx + ry * ry
                if r2 <= 9.5:
                    cand.append((r2, ii, jj, rx, ry))
        cand.sort(key=lambda item: item[0])
        if len(cand) < 6:
            return None
        cand = cand[:max_stencil]
        k = len(cand)
        A = np.empty((k, 6), dtype=np.float64)
        wd = np.empty(k, dtype=np.float64)
        for m, (r2, _ii, _jj, rx, ry) in enumerate(cand):
            A[m] = (1.0, rx, ry, 0.5 * rx * rx, rx * ry, 0.5 * ry * ry)
            wd[m] = math.exp(-0.5 * r2 / (1.5 * 1.5)) + 1.0e-4
        if np.linalg.matrix_rank(A, tol=1.0e-10) < 6:
            return None
        Aw = A * wd[:, None]
        M = A.T @ Aw
        cond = np.linalg.cond(M)
        if not np.isfinite(cond) or cond > 1.0e8:
            return None
        try:
            # derivative target with physical dimensions; coordinates were normalized by dx
            target = np.array([0.0, tx / dx, ty / dx, 0.0, 0.0, 0.0], dtype=np.float64)
            coeff_to_values = np.linalg.solve(M + 1.0e-12 * eye6, A.T * wd[None, :])
            wt = target @ coeff_to_values
        except np.linalg.LinAlgError:
            return None
        if (
            not np.all(np.isfinite(wt))
            or abs(np.sum(wt)) * dx > 1.0e-6
            or np.sum(np.abs(wt)) * dx > 8.0
        ):
            return None
        return cand, wt

    def gradient_weights(ell: int, xb: float, yb: float):
        i0 = int(link_i[ell])
        j0 = int(link_j[ell])
        cand: list[tuple[float, int, int, float, float]] = []
        radius_cells = 3
        for ii in range(max(0, i0 - radius_cells), min(nx, i0 + radius_cells + 1)):
            for jj in range(max(0, j0 - radius_cells), min(ny, j0 + radius_cells + 1)):
                if active[ii, jj] == 0:
                    continue
                rx = (float(x_coords[ii]) - xb) / dx
                ry = (float(y_coords[jj]) - yb) / dx
                r2 = rx * rx + ry * ry
                if r2 <= 9.5:
                    cand.append((r2, ii, jj, rx, ry))
        cand.sort(key=lambda item: item[0])
        if len(cand) < 6:
            return None
        cand = cand[:max_stencil]
        k = len(cand)
        A = np.empty((k, 6), dtype=np.float64)
        wd = np.empty(k, dtype=np.float64)
        for m, (r2, _ii, _jj, rx, ry) in enumerate(cand):
            A[m] = (1.0, rx, ry, 0.5 * rx * rx, rx * ry, 0.5 * ry * ry)
            wd[m] = math.exp(-0.5 * r2 / (1.5 * 1.5)) + 1.0e-4
        if np.linalg.matrix_rank(A, tol=1.0e-10) < 6:
            return None
        Aw = A * wd[:, None]
        M = A.T @ Aw
        cond = np.linalg.cond(M)
        if not np.isfinite(cond) or cond > 1.0e8:
            return None
        try:
            coeff_to_values = np.linalg.solve(M + 1.0e-12 * eye6, A.T * wd[None, :])
            wx = np.array([0.0, 1.0 / dx, 0.0, 0.0, 0.0, 0.0], dtype=np.float64) @ coeff_to_values
            wy = np.array([0.0, 0.0, 1.0 / dx, 0.0, 0.0, 0.0], dtype=np.float64) @ coeff_to_values
        except np.linalg.LinAlgError:
            return None
        if (
            not np.all(np.isfinite(wx))
            or not np.all(np.isfinite(wy))
            or abs(np.sum(wx)) * dx > 1.0e-6
            or abs(np.sum(wy)) * dx > 1.0e-6
            or np.sum(np.abs(wx)) * dx > 12.0
            or np.sum(np.abs(wy)) * dx > 12.0
        ):
            return None
        return cand, wx, wy

    for ell in range(n_links):
        xb = float(link_xb[ell])
        yb = float(link_yb[ell])
        nx_b = float(link_nx[ell])
        ny_b = float(link_ny[ell])
        accum: dict[tuple[int, int], float] = {}
        ok = False
        for h1, h2 in probe_pairs:
            w1 = bilinear_weights(xb - h1 * dx * nx_b, yb - h1 * dx * ny_b)
            w2 = bilinear_weights(xb - h2 * dx * nx_b, yb - h2 * dx * ny_b)
            if w1 is None or w2 is None:
                continue
            accum.clear()
            c1 = h2 / (h2 - h1)
            c2 = -h1 / (h2 - h1)
            for ii, jj, ww in w1:
                accum[(int(ii), int(jj))] = accum.get((int(ii), int(jj)), 0.0) + c1 * float(ww)
            for ii, jj, ww in w2:
                accum[(int(ii), int(jj))] = accum.get((int(ii), int(jj)), 0.0) + c2 * float(ww)
            ok = True
            break
        if not ok:
            for h in single_probes:
                w1 = bilinear_weights(xb - h * dx * nx_b, yb - h * dx * ny_b)
                if w1 is None:
                    continue
                accum.clear()
                for ii, jj, ww in w1:
                    accum[(int(ii), int(jj))] = accum.get((int(ii), int(jj)), 0.0) + float(ww)
                ok = True
                break
        if not ok:
            continue
        items = [(ij, w) for ij, w in accum.items() if abs(w) > 1.0e-14]
        items.sort(
            key=lambda item: (item[0][0] - int(link_i[ell])) ** 2
            + (item[0][1] - int(link_j[ell])) ** 2
        )
        if len(items) > max_stencil:
            continue
        if (
            not items
            or not np.isfinite([w for _ij, w in items]).all()
            or sum(abs(w) for _ij, w in items) > 3.5
        ):
            continue
        for m, ((ii, jj), ww) in enumerate(items):
            out_i[ell, m] = int(ii)
            out_j[ell, m] = int(jj)
            out_w[ell, m] = float(ww)
        td = tangent_derivative_weights(ell, xb, yb, nx_b, ny_b)
        if td is not None:
            cand_td, wt = td
            # Use the same index slots for the tangent derivative.  If a node is
            # not already present for the value extrapolation, append it.
            slot_of: dict[tuple[int, int], int] = {}
            for m in range(max_stencil):
                if out_i[ell, m] >= 0 and out_j[ell, m] >= 0:
                    slot_of[(int(out_i[ell, m]), int(out_j[ell, m]))] = m
            ok_td = True
            next_slot = len(slot_of)
            for m, (_r2, ii, jj, _rx, _ry) in enumerate(cand_td):
                key = (int(ii), int(jj))
                slot = slot_of.get(key)
                if slot is None:
                    if next_slot >= max_stencil:
                        ok_td = False
                        break
                    slot = next_slot
                    next_slot += 1
                    slot_of[key] = slot
                    out_i[ell, slot] = int(ii)
                    out_j[ell, slot] = int(jj)
                out_wt[ell, slot] = float(wt[m])
            if not ok_td:
                out_wt[ell, :] = 0.0
        gd = gradient_weights(ell, xb, yb)
        if gd is not None:
            cand_g, wx, wy = gd
            slot_of: dict[tuple[int, int], int] = {}
            for m in range(max_stencil):
                if out_i[ell, m] >= 0 and out_j[ell, m] >= 0:
                    slot_of[(int(out_i[ell, m]), int(out_j[ell, m]))] = m
            ok_g = True
            next_slot = len(slot_of)
            for m, (_r2, ii, jj, _rx, _ry) in enumerate(cand_g):
                key = (int(ii), int(jj))
                slot = slot_of.get(key)
                if slot is None:
                    if next_slot >= max_stencil:
                        ok_g = False
                        break
                    slot = next_slot
                    next_slot += 1
                    slot_of[key] = slot
                    out_i[ell, slot] = int(ii)
                    out_j[ell, slot] = int(jj)
                out_wx[ell, slot] = float(wx[m])
                out_wy[ell, slot] = float(wy[m])
            if not ok_g:
                out_wx[ell, :] = 0.0
                out_wy[ell, :] = 0.0
        valid[ell] = 1
    return (
        np.ascontiguousarray(out_i),
        np.ascontiguousarray(out_j),
        np.ascontiguousarray(out_w),
        np.ascontiguousarray(out_wt),
        np.ascontiguousarray(out_wx),
        np.ascontiguousarray(out_wy),
        np.ascontiguousarray(valid),
    )


@dataclass
class CurvedBoundaryGeometry:
    """Precomputed embedded-domain topology and cut-link geometry."""

    nx: int
    ny: int
    length_x: float
    length_y: float
    active: np.ndarray
    link_i: np.ndarray
    link_j: np.ndarray
    link_q: np.ndarray
    link_boundary_id: np.ndarray
    link_eta: np.ndarray
    link_xb: np.ndarray
    link_yb: np.ndarray
    link_nx: np.ndarray
    link_ny: np.ndarray
    label: str = "implicit"
    lsq_i: np.ndarray | None = None
    lsq_j: np.ndarray | None = None
    lsq_wb: np.ndarray | None = None
    lsq_wt: np.ndarray | None = None
    lsq_wx: np.ndarray | None = None
    lsq_wy: np.ndarray | None = None
    lsq_valid: np.ndarray | None = None

    @property
    def dx(self) -> float:
        return self.length_x / float(self.nx)

    @property
    def dy(self) -> float:
        return self.length_y / float(self.ny)

    @property
    def n_active(self) -> int:
        return int(np.count_nonzero(self.active))

    @property
    def n_links(self) -> int:
        return int(self.link_i.size)

    def as_metadata(self) -> dict[str, float | int | str]:
        return {
            "label": self.label,
            "nx": self.nx,
            "ny": self.ny,
            "length_x": self.length_x,
            "length_y": self.length_y,
            "dx": self.dx,
            "dy": self.dy,
            "n_active": self.n_active,
            "n_cut_links": self.n_links,
            "boundary_ids": sorted(int(v) for v in np.unique(self.link_boundary_id))
            if self.n_links
            else [],
        }

    @classmethod
    def from_level_set(
        cls,
        *,
        nx: int,
        ny: int,
        length_x: float,
        length_y: float,
        phi: Callable[[float, float], float],
        normal: Callable[[float, float], tuple[float, float]] | None = None,
        boundary_id: Callable[[float, float, float, float], int] | None = None,
        label: str = "level_set",
        root_iterations: int = 40,
    ) -> "CurvedBoundaryGeometry":
        """Build active mask and cut-link data from a scalar level-set function.

        ``phi<=0`` defines the material/reference domain.  Link cut fractions are
        found by bisection, so ``phi`` does not need to be an exact signed
        distance; only the sign is required.  If a normal function is not
        supplied, a central finite-difference gradient of ``phi`` is used at the
        cut point.
        """

        if nx < 4 or ny < 4:
            raise ValueError("nx and ny must be at least four")
        dx = float(length_x) / float(nx)
        dy = float(length_y) / float(ny)
        if abs(dx - dy) > 1.0e-14:
            raise ValueError("D2Q4 curved-boundary implementation requires dx=dy")
        x = (np.arange(nx, dtype=np.float64) + 0.5) * dx
        y = (np.arange(ny, dtype=np.float64) + 0.5) * dy
        xx, yy = np.meshgrid(x, y, indexing="ij")
        phi_grid = np.empty((nx, ny), dtype=np.float64)
        for i in range(nx):
            for j in range(ny):
                phi_grid[i, j] = float(phi(float(xx[i, j]), float(yy[i, j])))
        active = phi_grid <= 0.0
        if not np.any(active):
            raise ValueError("level-set contains no active material nodes")

        link_i: list[int] = []
        link_j: list[int] = []
        link_q: list[int] = []
        link_boundary_id: list[int] = []
        link_eta: list[float] = []
        link_xb: list[float] = []
        link_yb: list[float] = []
        link_nx: list[float] = []
        link_ny: list[float] = []

        def phi_safe(xp: float, yp: float) -> float:
            return float(phi(float(xp), float(yp)))

        def normal_safe(xp: float, yp: float) -> tuple[float, float]:
            if normal is not None:
                nxv, nyv = normal(float(xp), float(yp))
                nxv = float(nxv)
                nyv = float(nyv)
            else:
                h = 0.5 * min(dx, dy)
                gx = (phi_safe(xp + h, yp) - phi_safe(xp - h, yp)) / (2.0 * h)
                gy = (phi_safe(xp, yp + h) - phi_safe(xp, yp - h)) / (2.0 * h)
                nxv, nyv = gx, gy
            norm = math.hypot(nxv, nyv)
            if norm <= 1.0e-30:
                raise ValueError("zero level-set normal encountered")
            return nxv / norm, nyv / norm

        for i in range(nx):
            for j in range(ny):
                if not active[i, j]:
                    continue
                x0 = float(xx[i, j])
                y0 = float(yy[i, j])
                p0 = float(phi_grid[i, j])
                for q in range(Q):
                    ii = i + int(_CX[q])
                    jj = j + int(_CY[q])
                    neighbor_active = 0 <= ii < nx and 0 <= jj < ny and bool(active[ii, jj])
                    if neighbor_active:
                        continue
                    x1 = x0 + float(_CX[q]) * dx
                    y1 = y0 + float(_CY[q]) * dy
                    p1 = phi_safe(x1, y1)
                    # If the neighbor is outside the array, p1 is evaluated at
                    # the ghost node coordinate.  A valid cut link requires the
                    # segment to leave the implicit material region.
                    if not (p0 <= 0.0 and p1 >= 0.0):
                        # For under-resolved or non-monotone level sets, still
                        # try to locate the first sign change by sampling.  If
                        # none is found, use a conservative half-way fallback.
                        samples = np.linspace(0.0, 1.0, 17)
                        vals = [phi_safe(x0 + s * (x1 - x0), y0 + s * (y1 - y0)) for s in samples]
                        idx = None
                        for m in range(len(samples) - 1):
                            if vals[m] <= 0.0 and vals[m + 1] >= 0.0:
                                idx = m
                                break
                        if idx is None:
                            eta = 0.5
                            xb = x0 + eta * (x1 - x0)
                            yb = y0 + eta * (y1 - y0)
                            nxv, nyv = normal_safe(xb, yb)
                        else:
                            lo = float(samples[idx])
                            hi = float(samples[idx + 1])
                            for _ in range(root_iterations):
                                mid = 0.5 * (lo + hi)
                                pm = phi_safe(x0 + mid * (x1 - x0), y0 + mid * (y1 - y0))
                                if pm <= 0.0:
                                    lo = mid
                                else:
                                    hi = mid
                            eta = 0.5 * (lo + hi)
                            xb = x0 + eta * (x1 - x0)
                            yb = y0 + eta * (y1 - y0)
                            nxv, nyv = normal_safe(xb, yb)
                    else:
                        lo = 0.0
                        hi = 1.0
                        for _ in range(root_iterations):
                            mid = 0.5 * (lo + hi)
                            pm = phi_safe(x0 + mid * (x1 - x0), y0 + mid * (y1 - y0))
                            if pm <= 0.0:
                                lo = mid
                            else:
                                hi = mid
                        eta = 0.5 * (lo + hi)
                        xb = x0 + eta * (x1 - x0)
                        yb = y0 + eta * (y1 - y0)
                        nxv, nyv = normal_safe(xb, yb)
                    eta = float(np.clip(eta, 1.0e-6, 1.0 - 1.0e-6))
                    link_i.append(i)
                    link_j.append(j)
                    link_q.append(q)
                    bid = 1 if boundary_id is None else int(boundary_id(xb, yb, nxv, nyv))
                    link_boundary_id.append(max(0, min(MAX_BOUNDARY_IDS - 1, bid)))
                    link_eta.append(eta)
                    link_xb.append(xb)
                    link_yb.append(yb)
                    link_nx.append(nxv)
                    link_ny.append(nyv)

        link_i_arr = np.ascontiguousarray(np.asarray(link_i, dtype=np.int32))
        link_j_arr = np.ascontiguousarray(np.asarray(link_j, dtype=np.int32))
        link_q_arr = np.ascontiguousarray(np.asarray(link_q, dtype=np.int32))
        link_boundary_id_arr = np.ascontiguousarray(np.asarray(link_boundary_id, dtype=np.int32))
        link_eta_arr = np.ascontiguousarray(np.asarray(link_eta, dtype=np.float64))
        link_xb_arr = np.ascontiguousarray(np.asarray(link_xb, dtype=np.float64))
        link_yb_arr = np.ascontiguousarray(np.asarray(link_yb, dtype=np.float64))
        link_nx_arr = np.ascontiguousarray(np.asarray(link_nx, dtype=np.float64))
        link_ny_arr = np.ascontiguousarray(np.asarray(link_ny, dtype=np.float64))
        lsq_i_arr, lsq_j_arr, lsq_wb_arr, lsq_wt_arr, lsq_wx_arr, lsq_wy_arr, lsq_valid_arr = (
            _build_boundary_lsq_weights(
                active=active.astype(np.int32),
                x_coords=x,
                y_coords=y,
                link_i=link_i_arr,
                link_j=link_j_arr,
                link_xb=link_xb_arr,
                link_yb=link_yb_arr,
                link_nx=link_nx_arr,
                link_ny=link_ny_arr,
                dx=dx,
            )
        )

        return cls(
            nx=int(nx),
            ny=int(ny),
            length_x=float(length_x),
            length_y=float(length_y),
            active=np.ascontiguousarray(active.astype(np.int32)),
            link_i=link_i_arr,
            link_j=link_j_arr,
            link_q=link_q_arr,
            link_boundary_id=link_boundary_id_arr,
            link_eta=link_eta_arr,
            link_xb=link_xb_arr,
            link_yb=link_yb_arr,
            link_nx=link_nx_arr,
            link_ny=link_ny_arr,
            label=label,
            lsq_i=lsq_i_arr,
            lsq_j=lsq_j_arr,
            lsq_wb=lsq_wb_arr,
            lsq_wt=lsq_wt_arr,
            lsq_wx=lsq_wx_arr,
            lsq_wy=lsq_wy_arr,
            lsq_valid=lsq_valid_arr,
        )

    @classmethod
    def circle(
        cls,
        *,
        nx: int,
        ny: int,
        length_x: float = 1.0,
        length_y: float = 1.0,
        center_x: float = 0.5,
        center_y: float = 0.5,
        radius: float = 0.38,
    ) -> "CurvedBoundaryGeometry":
        circle = CircleDomain(center_x=center_x, center_y=center_y, radius=radius)
        return cls.from_level_set(
            nx=nx,
            ny=ny,
            length_x=length_x,
            length_y=length_y,
            phi=lambda x, y: float(math.hypot(x - center_x, y - center_y) - radius),
            normal=lambda x, y: (
                (x - center_x) / max(math.hypot(x - center_x, y - center_y), 1.0e-30),
                (y - center_y) / max(math.hypot(x - center_x, y - center_y), 1.0e-30),
            ),
            label=f"circle(cx={center_x},cy={center_y},r={radius})",
        )

    @classmethod
    def annulus(
        cls,
        *,
        nx: int,
        ny: int,
        length_x: float = 1.0,
        length_y: float = 1.0,
        center_x: float = 0.5,
        center_y: float = 0.5,
        inner_radius: float = 0.18,
        outer_radius: float = 0.42,
    ) -> "CurvedBoundaryGeometry":
        if not (0.0 < inner_radius < outer_radius):
            raise ValueError("annulus requires 0 < inner_radius < outer_radius")

        def radius(x: float, y: float) -> float:
            return math.hypot(x - center_x, y - center_y)

        def phi(x: float, y: float) -> float:
            r = radius(x, y)
            return max(r - outer_radius, inner_radius - r)

        def normal(x: float, y: float) -> tuple[float, float]:
            r = max(radius(x, y), 1.0e-30)
            ex = (x - center_x) / r
            ey = (y - center_y) / r
            if abs(r - inner_radius) <= abs(r - outer_radius):
                return -ex, -ey
            return ex, ey

        def boundary_id(x: float, y: float, _nx: float, _ny: float) -> int:
            r = radius(x, y)
            if abs(r - inner_radius) <= abs(r - outer_radius):
                return BOUNDARY_ID_INNER
            return BOUNDARY_ID_OUTER

        return cls.from_level_set(
            nx=nx,
            ny=ny,
            length_x=length_x,
            length_y=length_y,
            phi=phi,
            normal=normal,
            boundary_id=boundary_id,
            label=f"annulus(cx={center_x},cy={center_y},ri={inner_radius},ro={outer_radius})",
        )


@wp.func
def _curved_boundary_value_comp(
    comp: int,
    mode: int,
    p0: wp.float64,
    p1: wp.float64,
    p2: wp.float64,
    p3: wp.float64,
    p4: wp.float64,
    p5: wp.float64,
    p6: wp.float64,
    p7: wp.float64,
    x: wp.float64,
    y: wp.float64,
    nx_b: wp.float64,
    ny_b: wp.float64,
    t: wp.float64,
) -> wp.float64:
    if mode == CURVED_VALUE_PIOLA_NORMAL:
        # params: [P11, P12, P21, P22].  Output is T_i=P_iA n_A.
        if comp == 0:
            return p0 * nx_b + p1 * ny_b
        return p2 * nx_b + p3 * ny_b
    if mode == CURVED_VALUE_RADIAL_TRACTION:
        if comp == 0:
            return p0 * nx_b
        return p0 * ny_b
    if mode == CURVED_VALUE_RADIAL_TRACTION_SINE2_HOLD:
        active_time = p1
        sine_denominator = p2
        scale = wp.float64(1.0)
        if active_time > wp.float64(0.0) and t < active_time:
            denom = sine_denominator
            if denom <= wp.float64(0.0):
                denom = wp.float64(2.0) * active_time
            s = wp.sin(PI * t / denom)
            scale = s * s
        if comp == 0:
            return p0 * scale * nx_b
        return p0 * scale * ny_b
    if mode == CURVED_VALUE_RADIAL_TRACTION_SINE2_PULSE:
        pulse_time = p1
        scale = wp.float64(0.0)
        if pulse_time > wp.float64(0.0) and t < pulse_time:
            s = wp.sin(PI * t / pulse_time)
            scale = s * s
        if comp == 0:
            return p0 * scale * nx_b
        return p0 * scale * ny_b
    return _boundary_value_comp(comp, mode, p0, p1, p2, p3, p4, p5, p6, p7, x, y, t)


@wp.func
def _normal_traction_residual_arbitrary(
    hn0: wp.float64,
    hn1: wp.float64,
    gt0: wp.float64,
    gt1: wp.float64,
    nx_b: wp.float64,
    ny_b: wp.float64,
    tx: wp.float64,
    ty: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
) -> wp.vec2d:
    # t = (-ny, nx); F = (F n) \otimes n + (F t) \otimes t.
    tx1 = -ny_b
    tx2 = nx_b
    F11 = hn0 * nx_b + gt0 * tx1
    F12 = hn0 * ny_b + gt0 * tx2
    F21 = hn1 * nx_b + gt1 * tx1
    F22 = hn1 * ny_b + gt1 * tx2
    P = _piola_vec(F11, F12, F21, F22, model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor)
    return wp.vec2d(P[0] * nx_b + P[1] * ny_b - tx, P[2] * nx_b + P[3] * ny_b - ty)


@wp.func
def _neo_hooke_acoustic_normal_2d(
    F11: wp.float64,
    F12: wp.float64,
    F21: wp.float64,
    F22: wp.float64,
    nx_b: wp.float64,
    ny_b: wp.float64,
    lam: wp.float64,
    mu: wp.float64,
    j_floor: wp.float64,
) -> wp.vec4d:
    J = _clamp_j(F11 * F22 - F12 * F21, j_floor)
    H00 = F22 / J
    H01 = -F21 / J
    H10 = -F12 / J
    H11 = F11 / J
    beta = wp.float64(0.5) * lam * (J * J - wp.float64(1.0)) - mu
    coeff = lam * J * J - beta
    n2 = nx_b * nx_b + ny_b * ny_b
    v0 = H00 * nx_b + H01 * ny_b
    v1 = H10 * nx_b + H11 * ny_b
    return wp.vec4d(
        mu * n2 + coeff * v0 * v0,
        coeff * v0 * v1,
        coeff * v1 * v0,
        mu * n2 + coeff * v1 * v1,
    )


@wp.func
def _solve_normal_image_arbitrary(
    F11_guess: wp.float64,
    F12_guess: wp.float64,
    F21_guess: wp.float64,
    F22_guess: wp.float64,
    nx_b: wp.float64,
    ny_b: wp.float64,
    tx: wp.float64,
    ty: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
    normal_jacobian: int,
) -> wp.vec4d:
    if normal_jacobian == NORMAL_SOLVER_ENERGY and model == MATERIAL_NEO_HOOKE:
        # With the tangential image fixed, a = cof(F) n is independent of
        # h = F n and J = a.h.  Minimise W(F) - T.h on J > 0.  Writing
        # e=a/|a|, s=h.e gives A*s*s - (T.e)*s - B = 0, with A,B > 0.
        # The positive root is the unique minimiser.  No inverse or iteration.
        a0 = F22_guess * nx_b - F21_guess * ny_b
        a1 = -F12_guess * nx_b + F11_guess * ny_b
        a2 = a0 * a0 + a1 * a1
        if a2 > wp.float64(1.0e-28):
            area = wp.sqrt(a2)
            e0 = a0 / area
            e1 = a1 / area
            tau = tx * e0 + ty * e1
            aa = mu + wp.float64(0.5) * lam * a2
            bb = mu + wp.float64(0.5) * lam
            disc = wp.sqrt(tau * tau + wp.float64(4.0) * aa * bb)
            s = (tau + disc) / (wp.float64(2.0) * aa)
            if tau < wp.float64(0.0):
                s = wp.float64(2.0) * bb / (disc - tau)
            if area * s > j_floor:
                h0 = (tx - tau * e0) / mu + s * e0
                h1 = (ty - tau * e1) / mu + s * e1
                d0 = h0 - (F11_guess * nx_b + F12_guess * ny_b)
                d1 = h1 - (F21_guess * nx_b + F22_guess * ny_b)
                return wp.vec4d(
                    F11_guess + d0 * nx_b,
                    F12_guess + d0 * ny_b,
                    F21_guess + d1 * nx_b,
                    F22_guess + d1 * ny_b,
                )
        # Degenerate surface images / states below the existing J floor use
        # the legacy bounded Newton path.  Such states are not certified by
        # the closed-form energy argument.
    # Decompose the predicted boundary gradient into normal and tangential
    # images.  Tangential image is fixed; normal image is solved from P n=T.
    tanx = -ny_b
    tany = nx_b
    hn0 = F11_guess * nx_b + F12_guess * ny_b
    hn1 = F21_guess * nx_b + F22_guess * ny_b
    gt0 = F11_guess * tanx + F12_guess * tany
    gt1 = F21_guess * tanx + F22_guess * tany
    eps = wp.float64(1.0e-6)

    for _it in range(16):
        r = _normal_traction_residual_arbitrary(
            hn0, hn1, gt0, gt1, nx_b, ny_b, tx, ty, model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor
        )
        rnorm = wp.sqrt(r[0] * r[0] + r[1] * r[1])
        if rnorm < wp.float64(1.0e-11):
            break
        a = wp.float64(0.0)
        b = wp.float64(0.0)
        c = wp.float64(0.0)
        d = wp.float64(0.0)
        if normal_jacobian != NORMAL_JACOBIAN_FINITE_DIFFERENCE and model == MATERIAL_NEO_HOOKE:
            F11 = hn0 * nx_b + gt0 * tanx
            F12 = hn0 * ny_b + gt0 * tany
            F21 = hn1 * nx_b + gt1 * tanx
            F22 = hn1 * ny_b + gt1 * tany
            Qn = _neo_hooke_acoustic_normal_2d(F11, F12, F21, F22, nx_b, ny_b, lam, mu, j_floor)
            a = Qn[0]
            b = Qn[1]
            c = Qn[2]
            d = Qn[3]
        else:
            rp0 = _normal_traction_residual_arbitrary(
                hn0 + eps,
                hn1,
                gt0,
                gt1,
                nx_b,
                ny_b,
                tx,
                ty,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
            )
            rp1 = _normal_traction_residual_arbitrary(
                hn0,
                hn1 + eps,
                gt0,
                gt1,
                nx_b,
                ny_b,
                tx,
                ty,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
            )
            a = (rp0[0] - r[0]) / eps
            c = (rp0[1] - r[1]) / eps
            b = (rp1[0] - r[0]) / eps
            d = (rp1[1] - r[1]) / eps
        det = a * d - b * c
        if wp.abs(det) < wp.float64(1.0e-14):
            break
        delta0 = (-r[0] * d + b * r[1]) / det
        delta1 = (c * r[0] - a * r[1]) / det
        step = wp.float64(1.0)
        for _ls in range(12):
            chn0 = hn0 + step * delta0
            chn1 = hn1 + step * delta1
            F11 = chn0 * nx_b + gt0 * tanx
            F12 = chn0 * ny_b + gt0 * tany
            F21 = chn1 * nx_b + gt1 * tanx
            F22 = chn1 * ny_b + gt1 * tany
            J = F11 * F22 - F12 * F21
            if J > j_floor:
                hn0 = chn0
                hn1 = chn1
                break
            step = wp.float64(0.5) * step

    F11_out = hn0 * nx_b + gt0 * tanx
    F12_out = hn0 * ny_b + gt0 * tany
    F21_out = hn1 * nx_b + gt1 * tanx
    F22_out = hn1 * ny_b + gt1 * tany
    return wp.vec4d(F11_out, F12_out, F21_out, F22_out)


@wp.func
def _state_from_U_component(
    U: wp.array4d(dtype=wp.float64), comp: int, i: int, j: int
) -> wp.float64:
    return U[comp, i, j, 0]


@wp.func
def _lsq_U_at_boundary(
    U: wp.array4d(dtype=wp.float64),
    lsq_i: wp.array2d(dtype=wp.int32),
    lsq_j: wp.array2d(dtype=wp.int32),
    lsq_wb: wp.array2d(dtype=wp.float64),
    lsq_valid: wp.array(dtype=wp.int32),
    ell: int,
    comp: int,
    fallback: wp.float64,
) -> wp.float64:
    if lsq_valid[ell] == 0:
        return fallback
    out = wp.float64(0.0)
    for m in range(MAX_LSQ_STENCIL):
        ii = lsq_i[ell, m]
        jj = lsq_j[ell, m]
        if ii >= 0 and jj >= 0:
            out += lsq_wb[ell, m] * U[comp, ii, jj, 0]
    return out


@wp.func
def _jvp_eps6(
    w0: wp.float64,
    w1: wp.float64,
    w2: wp.float64,
    w3: wp.float64,
    w4: wp.float64,
    w5: wp.float64,
) -> wp.float64:
    scale = wp.abs(w0)
    if wp.abs(w1) > scale:
        scale = wp.abs(w1)
    if wp.abs(w2) > scale:
        scale = wp.abs(w2)
    if wp.abs(w3) > scale:
        scale = wp.abs(w3)
    if wp.abs(w4) > scale:
        scale = wp.abs(w4)
    if wp.abs(w5) > scale:
        scale = wp.abs(w5)
    eps = wp.float64(1.0e-6)
    if scale > wp.float64(1.0):
        eps = eps / scale
    return eps


@wp.func
def _flux_x_jvp_comp(
    a: int,
    vx: wp.float64,
    vy: wp.float64,
    F11: wp.float64,
    F12: wp.float64,
    F21: wp.float64,
    F22: wp.float64,
    w0: wp.float64,
    w1: wp.float64,
    w2: wp.float64,
    w3: wp.float64,
    w4: wp.float64,
    w5: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
) -> wp.float64:
    eps = _jvp_eps6(w0, w1, w2, w3, w4, w5)
    fp = _flux_x_comp(
        a,
        vx + eps * w0,
        vy + eps * w1,
        F11 + eps * w2,
        F12 + eps * w3,
        F21 + eps * w4,
        F22 + eps * w5,
        model,
        lam,
        mu,
        mat_p0,
        mat_p1,
        mat_p2,
        j_floor,
    )
    fm = _flux_x_comp(
        a,
        vx - eps * w0,
        vy - eps * w1,
        F11 - eps * w2,
        F12 - eps * w3,
        F21 - eps * w4,
        F22 - eps * w5,
        model,
        lam,
        mu,
        mat_p0,
        mat_p1,
        mat_p2,
        j_floor,
    )
    return (fp - fm) / (wp.float64(2.0) * eps)


@wp.func
def _flux_y_jvp_comp(
    a: int,
    vx: wp.float64,
    vy: wp.float64,
    F11: wp.float64,
    F12: wp.float64,
    F21: wp.float64,
    F22: wp.float64,
    w0: wp.float64,
    w1: wp.float64,
    w2: wp.float64,
    w3: wp.float64,
    w4: wp.float64,
    w5: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
) -> wp.float64:
    eps = _jvp_eps6(w0, w1, w2, w3, w4, w5)
    fp = _flux_y_comp(
        a,
        vx + eps * w0,
        vy + eps * w1,
        F11 + eps * w2,
        F12 + eps * w3,
        F21 + eps * w4,
        F22 + eps * w5,
        model,
        lam,
        mu,
        mat_p0,
        mat_p1,
        mat_p2,
        j_floor,
    )
    fm = _flux_y_comp(
        a,
        vx - eps * w0,
        vy - eps * w1,
        F11 - eps * w2,
        F12 - eps * w3,
        F21 - eps * w4,
        F22 - eps * w5,
        model,
        lam,
        mu,
        mat_p0,
        mat_p1,
        mat_p2,
        j_floor,
    )
    return (fp - fm) / (wp.float64(2.0) * eps)


@wp.func
def _equilibrium_jvp_comp(
    q: int,
    a: int,
    vx: wp.float64,
    vy: wp.float64,
    F11: wp.float64,
    F12: wp.float64,
    F21: wp.float64,
    F22: wp.float64,
    w0: wp.float64,
    w1: wp.float64,
    w2: wp.float64,
    w3: wp.float64,
    w4: wp.float64,
    w5: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    lattice_speed: wp.float64,
    j_floor: wp.float64,
) -> wp.float64:
    AxW = _flux_x_jvp_comp(
        a,
        vx,
        vy,
        F11,
        F12,
        F21,
        F22,
        w0,
        w1,
        w2,
        w3,
        w4,
        w5,
        model,
        lam,
        mu,
        mat_p0,
        mat_p1,
        mat_p2,
        j_floor,
    )
    AyW = _flux_y_jvp_comp(
        a,
        vx,
        vy,
        F11,
        F12,
        F21,
        F22,
        w0,
        w1,
        w2,
        w3,
        w4,
        w5,
        model,
        lam,
        mu,
        mat_p0,
        mat_p1,
        mat_p2,
        j_floor,
    )
    Wa = _state_comp(a, w0, w1, w2, w3, w4, w5)
    dxq = wp.float64(CX_I[q])
    dyq = wp.float64(CY_I[q])
    return wp.float64(0.25) * (Wa + (wp.float64(2.0) / lattice_speed) * (dxq * AxW + dyq * AyW))


@wp.func
def _interp_U_link(
    U: wp.array4d(dtype=wp.float64),
    active: wp.array3d(dtype=wp.int32),
    comp: int,
    i: int,
    j: int,
    q_out: int,
    s: wp.float64,
    nx: int,
    ny: int,
) -> wp.float64:
    """Quadratic one-sided reconstruction along the inward link coordinate.

    The active cut node is at coordinate 0, and the two interior upstream
    nodes, when present, are at -1 and -2.  The evaluation coordinate s is
    typically eta (boundary point) or 2*eta-1 (mirror point).  If insufficient
    upstream nodes are active, the formula degrades to linear or constant
    extrapolation.
    """
    v0 = U[comp, i, j, 0]
    i1 = i - CX_I[q_out]
    j1 = j - CY_I[q_out]
    if i1 >= 0 and i1 < nx and j1 >= 0 and j1 < ny:
        if active[i1, j1, 0] != 0:
            v1 = U[comp, i1, j1, 0]
            i2 = i - wp.int32(2) * CX_I[q_out]
            j2 = j - wp.int32(2) * CY_I[q_out]
            if i2 >= 0 and i2 < nx and j2 >= 0 and j2 < ny:
                if active[i2, j2, 0] != 0:
                    v2 = U[comp, i2, j2, 0]
                    l0 = wp.float64(0.5) * (s + wp.float64(1.0)) * (s + wp.float64(2.0))
                    l1 = -s * (s + wp.float64(2.0))
                    l2 = wp.float64(0.5) * s * (s + wp.float64(1.0))
                    return l0 * v0 + l1 * v1 + l2 * v2
            # Linear through coordinates 0 and -1.
            return (s + wp.float64(1.0)) * v0 - s * v1
    return v0


@wp.func
def _interp_fpost_link(
    fpost: wp.array4d(dtype=wp.float64),
    active: wp.array3d(dtype=wp.int32),
    q: int,
    comp: int,
    i: int,
    j: int,
    q_out: int,
    s: wp.float64,
    nx: int,
    ny: int,
) -> wp.float64:
    """Quadratic one-sided interpolation of a post-collision population."""
    v0 = fpost[q * 6 + comp, i, j, 0]
    i1 = i - CX_I[q_out]
    j1 = j - CY_I[q_out]
    if i1 >= 0 and i1 < nx and j1 >= 0 and j1 < ny:
        if active[i1, j1, 0] != 0:
            v1 = fpost[q * 6 + comp, i1, j1, 0]
            i2 = i - wp.int32(2) * CX_I[q_out]
            j2 = j - wp.int32(2) * CY_I[q_out]
            if i2 >= 0 and i2 < nx and j2 >= 0 and j2 < ny:
                if active[i2, j2, 0] != 0:
                    v2 = fpost[q * 6 + comp, i2, j2, 0]
                    l0 = wp.float64(0.5) * (s + wp.float64(1.0)) * (s + wp.float64(2.0))
                    l1 = -s * (s + wp.float64(2.0))
                    l2 = wp.float64(0.5) * s * (s + wp.float64(1.0))
                    return l0 * v0 + l1 * v1 + l2 * v2
            return (s + wp.float64(1.0)) * v0 - s * v1
    return v0


@wp.func
def _regularized_post_neq_comp(
    fpost: wp.array4d(dtype=wp.float64),
    q: int,
    comp: int,
    i: int,
    j: int,
    vx: wp.float64,
    vy: wp.float64,
    F11: wp.float64,
    F12: wp.float64,
    F21: wp.float64,
    F22: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    lattice_speed: wp.float64,
    j_floor: wp.float64,
) -> wp.float64:
    """Post-collision nonequilibrium with the D2Q4 ghost moment removed.

    The removal leaves the conserved moment and the two physical flux moments
    unchanged and sets m3=hE-hN+hW-hS to zero.
    """
    hE = fpost[0 * 6 + comp, i, j, 0] - _equilibrium_comp(
        0,
        comp,
        vx,
        vy,
        F11,
        F12,
        F21,
        F22,
        model,
        lam,
        mu,
        mat_p0,
        mat_p1,
        mat_p2,
        lattice_speed,
        j_floor,
    )
    hN = fpost[1 * 6 + comp, i, j, 0] - _equilibrium_comp(
        1,
        comp,
        vx,
        vy,
        F11,
        F12,
        F21,
        F22,
        model,
        lam,
        mu,
        mat_p0,
        mat_p1,
        mat_p2,
        lattice_speed,
        j_floor,
    )
    hW = fpost[2 * 6 + comp, i, j, 0] - _equilibrium_comp(
        2,
        comp,
        vx,
        vy,
        F11,
        F12,
        F21,
        F22,
        model,
        lam,
        mu,
        mat_p0,
        mat_p1,
        mat_p2,
        lattice_speed,
        j_floor,
    )
    hS = fpost[3 * 6 + comp, i, j, 0] - _equilibrium_comp(
        3,
        comp,
        vx,
        vy,
        F11,
        F12,
        F21,
        F22,
        model,
        lam,
        mu,
        mat_p0,
        mat_p1,
        mat_p2,
        lattice_speed,
        j_floor,
    )
    m3 = hE - hN + hW - hS
    corr = wp.float64(0.25) * m3
    if q == 0:
        return hE - corr
    if q == 1:
        return hN + corr
    if q == 2:
        return hW - corr
    return hS + corr


@wp.func
def _regularized_fpost_comp(
    fpost: wp.array4d(dtype=wp.float64),
    q: int,
    comp: int,
    i: int,
    j: int,
    U: wp.array4d(dtype=wp.float64),
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    lattice_speed: wp.float64,
    j_floor: wp.float64,
) -> wp.float64:
    vx = U[0, i, j, 0]
    vy = U[1, i, j, 0]
    F11 = U[2, i, j, 0]
    F12 = U[3, i, j, 0]
    F21 = U[4, i, j, 0]
    F22 = U[5, i, j, 0]
    feq = _equilibrium_comp(
        q,
        comp,
        vx,
        vy,
        F11,
        F12,
        F21,
        F22,
        model,
        lam,
        mu,
        mat_p0,
        mat_p1,
        mat_p2,
        lattice_speed,
        j_floor,
    )
    hreg = _regularized_post_neq_comp(
        fpost,
        q,
        comp,
        i,
        j,
        vx,
        vy,
        F11,
        F12,
        F21,
        F22,
        model,
        lam,
        mu,
        mat_p0,
        mat_p1,
        mat_p2,
        lattice_speed,
        j_floor,
    )
    return feq + hreg


@wp.kernel
def _collide_masked_kernel(
    f0: wp.array4d(dtype=wp.float64),
    fpost: wp.array4d(dtype=wp.float64),
    u: wp.array4d(dtype=wp.float64),
    u_star: wp.array4d(dtype=wp.float64),
    U: wp.array4d(dtype=wp.float64),
    P_field: wp.array4d(dtype=wp.float64),
    sigma: wp.array4d(dtype=wp.float64),
    active: wp.array3d(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
    dt: wp.float64,
    omega: wp.float64,
    lattice_speed: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
    source_mode: int,
    source_p0: wp.float64,
    source_p1: wp.float64,
    source_p2: wp.float64,
    source_p3: wp.float64,
    source_p4: wp.float64,
    source_p5: wp.float64,
):
    i, j, k = wp.tid()
    if active[i, j, 0] == 0:
        for q in range(4):
            for a in range(6):
                fpost[q * 6 + a, i, j, k] = wp.float64(0.0)
        for a in range(6):
            U[a, i, j, k] = wp.float64(0.0)
        u[0, i, j, k] = wp.float64(0.0)
        u[1, i, j, k] = wp.float64(0.0)
        return

    vx = wp.float64(0.0)
    vy = wp.float64(0.0)
    F11 = wp.float64(0.0)
    F12 = wp.float64(0.0)
    F21 = wp.float64(0.0)
    F22 = wp.float64(0.0)
    for q in range(4):
        vx += f0[q * 6 + 0, i, j, k]
        vy += f0[q * 6 + 1, i, j, k]
        F11 += f0[q * 6 + 2, i, j, k]
        F12 += f0[q * 6 + 3, i, j, k]
        F21 += f0[q * 6 + 4, i, j, k]
        F22 += f0[q * 6 + 5, i, j, k]

    half_dt = wp.float64(0.5) * dt
    if source_mode == SOURCE_CONSTANT:
        vx += half_dt * source_p0
        vy += half_dt * source_p1
        F11 += half_dt * source_p2
        F12 += half_dt * source_p3
        F21 += half_dt * source_p4
        F22 += half_dt * source_p5
    elif source_mode == SOURCE_DAMPING:
        denom = wp.float64(1.0) + half_dt * source_p0
        vx = vx / denom
        vy = vy / denom

    b0 = _source_comp(
        0, source_mode, source_p0, source_p1, source_p2, source_p3, source_p4, source_p5, vx, vy
    )
    b1 = _source_comp(
        1, source_mode, source_p0, source_p1, source_p2, source_p3, source_p4, source_p5, vx, vy
    )
    b2 = _source_comp(
        2, source_mode, source_p0, source_p1, source_p2, source_p3, source_p4, source_p5, vx, vy
    )
    b3 = _source_comp(
        3, source_mode, source_p0, source_p1, source_p2, source_p3, source_p4, source_p5, vx, vy
    )
    b4 = _source_comp(
        4, source_mode, source_p0, source_p1, source_p2, source_p3, source_p4, source_p5, vx, vy
    )
    b5 = _source_comp(
        5, source_mode, source_p0, source_p1, source_p2, source_p3, source_p4, source_p5, vx, vy
    )

    U[0, i, j, k] = vx
    U[1, i, j, k] = vy
    U[2, i, j, k] = F11
    U[3, i, j, k] = F12
    U[4, i, j, k] = F21
    U[5, i, j, k] = F22

    ux = u_star[0, i, j, k] + wp.float64(0.5) * dt * vx
    uy = u_star[1, i, j, k] + wp.float64(0.5) * dt * vy
    u[0, i, j, k] = ux
    u[1, i, j, k] = uy
    u_star[0, i, j, k] = ux + wp.float64(0.5) * dt * vx
    u_star[1, i, j, k] = uy + wp.float64(0.5) * dt * vy

    J = F11 * F22 - F12 * F21
    if J <= j_floor:
        invalid[0] = wp.int32(1)
    J_safe = _clamp_j(J, j_floor)
    P = _piola_vec(F11, F12, F21, F22, model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor)
    P_field[0, i, j, k] = P[0]
    P_field[1, i, j, k] = P[1]
    P_field[2, i, j, k] = P[2]
    P_field[3, i, j, k] = P[3]
    sigma[0, i, j, k] = (P[0] * F11 + P[1] * F12) / J_safe
    sigma[1, i, j, k] = (P[0] * F21 + P[1] * F22) / J_safe
    sigma[2, i, j, k] = (P[2] * F11 + P[3] * F12) / J_safe
    sigma[3, i, j, k] = (P[2] * F21 + P[3] * F22) / J_safe

    for q in range(4):
        for a in range(6):
            feq = _equilibrium_comp(
                q,
                a,
                vx,
                vy,
                F11,
                F12,
                F21,
                F22,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                lattice_speed,
                j_floor,
            )
            old = f0[q * 6 + a, i, j, k]
            source_term = wp.float64(0.0)
            if a == 0:
                source_term = b0
            elif a == 1:
                source_term = b1
            elif a == 2:
                source_term = b2
            elif a == 3:
                source_term = b3
            elif a == 4:
                source_term = b4
            else:
                source_term = b5
            fpost[q * 6 + a, i, j, k] = (
                old
                + omega * (feq - old)
                + dt * (wp.float64(2.0) - omega) * wp.float64(0.125) * source_term
            )


@wp.kernel
def _collide_masked_mrt_ghost_kernel(
    f0: wp.array4d(dtype=wp.float64),
    fpost: wp.array4d(dtype=wp.float64),
    u: wp.array4d(dtype=wp.float64),
    u_star: wp.array4d(dtype=wp.float64),
    U: wp.array4d(dtype=wp.float64),
    P_field: wp.array4d(dtype=wp.float64),
    sigma: wp.array4d(dtype=wp.float64),
    active: wp.array3d(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
    dt: wp.float64,
    omega_flux: wp.float64,
    omega_ghost: wp.float64,
    lattice_speed: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
    source_mode: int,
    source_p0: wp.float64,
    source_p1: wp.float64,
    source_p2: wp.float64,
    source_p3: wp.float64,
    source_p4: wp.float64,
    source_p5: wp.float64,
):
    """Moment-space collision with low-dissipation flux relaxation and ghost damping.

    For each six-component vector population this kernel uses moments

        m0 = fE + fN + fW + fS,
        m1 = fE - fW,
        m2 = fN - fS,
        m3 = fE - fN + fW - fS.

    The physical flux moments m1,m2 are relaxed with omega_flux, while the
    non-hydrodynamic horizontal/vertical ghost m3 is relaxed with omega_ghost.
    Setting omega_ghost=omega_flux recovers the BGK collision exactly.
    """
    i, j, k = wp.tid()
    if active[i, j, 0] == 0:
        for q in range(4):
            for a in range(6):
                fpost[q * 6 + a, i, j, k] = wp.float64(0.0)
        for a in range(6):
            U[a, i, j, k] = wp.float64(0.0)
        u[0, i, j, k] = wp.float64(0.0)
        u[1, i, j, k] = wp.float64(0.0)
        return

    vx = wp.float64(0.0)
    vy = wp.float64(0.0)
    F11 = wp.float64(0.0)
    F12 = wp.float64(0.0)
    F21 = wp.float64(0.0)
    F22 = wp.float64(0.0)
    for q in range(4):
        vx += f0[q * 6 + 0, i, j, k]
        vy += f0[q * 6 + 1, i, j, k]
        F11 += f0[q * 6 + 2, i, j, k]
        F12 += f0[q * 6 + 3, i, j, k]
        F21 += f0[q * 6 + 4, i, j, k]
        F22 += f0[q * 6 + 5, i, j, k]

    half_dt = wp.float64(0.5) * dt
    if source_mode == SOURCE_CONSTANT:
        vx += half_dt * source_p0
        vy += half_dt * source_p1
        F11 += half_dt * source_p2
        F12 += half_dt * source_p3
        F21 += half_dt * source_p4
        F22 += half_dt * source_p5
    elif source_mode == SOURCE_DAMPING:
        denom = wp.float64(1.0) + half_dt * source_p0
        vx = vx / denom
        vy = vy / denom

    b0 = _source_comp(
        0, source_mode, source_p0, source_p1, source_p2, source_p3, source_p4, source_p5, vx, vy
    )
    b1 = _source_comp(
        1, source_mode, source_p0, source_p1, source_p2, source_p3, source_p4, source_p5, vx, vy
    )
    b2 = _source_comp(
        2, source_mode, source_p0, source_p1, source_p2, source_p3, source_p4, source_p5, vx, vy
    )
    b3 = _source_comp(
        3, source_mode, source_p0, source_p1, source_p2, source_p3, source_p4, source_p5, vx, vy
    )
    b4 = _source_comp(
        4, source_mode, source_p0, source_p1, source_p2, source_p3, source_p4, source_p5, vx, vy
    )
    b5 = _source_comp(
        5, source_mode, source_p0, source_p1, source_p2, source_p3, source_p4, source_p5, vx, vy
    )

    U[0, i, j, k] = vx
    U[1, i, j, k] = vy
    U[2, i, j, k] = F11
    U[3, i, j, k] = F12
    U[4, i, j, k] = F21
    U[5, i, j, k] = F22

    ux = u_star[0, i, j, k] + wp.float64(0.5) * dt * vx
    uy = u_star[1, i, j, k] + wp.float64(0.5) * dt * vy
    u[0, i, j, k] = ux
    u[1, i, j, k] = uy
    u_star[0, i, j, k] = ux + wp.float64(0.5) * dt * vx
    u_star[1, i, j, k] = uy + wp.float64(0.5) * dt * vy

    J = F11 * F22 - F12 * F21
    if J <= j_floor:
        invalid[0] = wp.int32(1)
    J_safe = _clamp_j(J, j_floor)
    P = _piola_vec(F11, F12, F21, F22, model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor)
    P_field[0, i, j, k] = P[0]
    P_field[1, i, j, k] = P[1]
    P_field[2, i, j, k] = P[2]
    P_field[3, i, j, k] = P[3]
    sigma[0, i, j, k] = (P[0] * F11 + P[1] * F12) / J_safe
    sigma[1, i, j, k] = (P[0] * F21 + P[1] * F22) / J_safe
    sigma[2, i, j, k] = (P[2] * F11 + P[3] * F12) / J_safe
    sigma[3, i, j, k] = (P[2] * F21 + P[3] * F22) / J_safe

    for a in range(6):
        fE = f0[0 * 6 + a, i, j, k]
        fN = f0[1 * 6 + a, i, j, k]
        fW = f0[2 * 6 + a, i, j, k]
        fS = f0[3 * 6 + a, i, j, k]

        m0 = fE + fN + fW + fS
        m1 = fE - fW
        m2 = fN - fS
        m3 = fE - fN + fW - fS

        ua = wp.float64(0.0)
        bx = wp.float64(0.0)
        phix = wp.float64(0.0)
        phiy = wp.float64(0.0)
        if a == 0:
            ua = vx
            bx = b0
            phix = -P[0]
            phiy = -P[1]
        elif a == 1:
            ua = vy
            bx = b1
            phix = -P[2]
            phiy = -P[3]
        elif a == 2:
            ua = F11
            bx = b2
            phix = -vx
            phiy = wp.float64(0.0)
        elif a == 3:
            ua = F12
            bx = b3
            phix = wp.float64(0.0)
            phiy = -vx
        elif a == 4:
            ua = F21
            bx = b4
            phix = -vy
            phiy = wp.float64(0.0)
        else:
            ua = F22
            bx = b5
            phix = wp.float64(0.0)
            phiy = -vy

        m0_post = (
            m0 + omega_flux * (ua - m0) + dt * (wp.float64(2.0) - omega_flux) * wp.float64(0.5) * bx
        )
        m1_eq = phix / lattice_speed
        m2_eq = phiy / lattice_speed
        m1_post = m1 + omega_flux * (m1_eq - m1)
        m2_post = m2 + omega_flux * (m2_eq - m2)
        m3_post = (wp.float64(1.0) - omega_ghost) * m3

        fpost[0 * 6 + a, i, j, k] = wp.float64(0.25) * (
            m0_post + m3_post + wp.float64(2.0) * m1_post
        )
        fpost[1 * 6 + a, i, j, k] = wp.float64(0.25) * (
            m0_post - m3_post + wp.float64(2.0) * m2_post
        )
        fpost[2 * 6 + a, i, j, k] = wp.float64(0.25) * (
            m0_post + m3_post - wp.float64(2.0) * m1_post
        )
        fpost[3 * 6 + a, i, j, k] = wp.float64(0.25) * (
            m0_post - m3_post - wp.float64(2.0) * m2_post
        )


@wp.kernel
def _stream_masked_kernel(
    fpost: wp.array4d(dtype=wp.float64),
    f1: wp.array4d(dtype=wp.float64),
    active: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
):
    i, j, k = wp.tid()
    if active[i, j, 0] == 0:
        return
    for q in range(4):
        ii = i + CX_I[q]
        jj = j + CY_I[q]
        if ii >= 0 and ii < nx and jj >= 0 and jj < ny:
            if active[ii, jj, 0] != 0:
                for a in range(6):
                    f1[q * 6 + a, ii, jj, k] = fpost[q * 6 + a, i, j, k]


@wp.kernel
def _apply_curved_boundary_links_kernel(
    fpost: wp.array4d(dtype=wp.float64),
    f1: wp.array4d(dtype=wp.float64),
    U: wp.array4d(dtype=wp.float64),
    u: wp.array4d(dtype=wp.float64),
    active: wp.array3d(dtype=wp.int32),
    link_i: wp.array(dtype=wp.int32),
    link_j: wp.array(dtype=wp.int32),
    link_q: wp.array(dtype=wp.int32),
    link_boundary_id: wp.array(dtype=wp.int32),
    link_eta: wp.array(dtype=wp.float64),
    link_xb: wp.array(dtype=wp.float64),
    link_yb: wp.array(dtype=wp.float64),
    link_nx: wp.array(dtype=wp.float64),
    link_ny: wp.array(dtype=wp.float64),
    lsq_i: wp.array2d(dtype=wp.int32),
    lsq_j: wp.array2d(dtype=wp.int32),
    lsq_wb: wp.array2d(dtype=wp.float64),
    lsq_wt: wp.array2d(dtype=wp.float64),
    lsq_wx: wp.array2d(dtype=wp.float64),
    lsq_wy: wp.array2d(dtype=wp.float64),
    lsq_valid: wp.array(dtype=wp.int32),
    n_links: int,
    nx: int,
    ny: int,
    boundary_kind_by_id: wp.array(dtype=wp.int32),
    boundary_mode_by_id: wp.array(dtype=wp.int32),
    boundary_params_by_id: wp.array2d(dtype=wp.float64),
    link_traction_x: wp.array(dtype=wp.float64),
    link_traction_y: wp.array(dtype=wp.float64),
    use_dynamic_link_traction: int,
    time_mid: wp.float64,
    dt: wp.float64,
    collision_omega: wp.float64,
    lattice_speed: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
    source_mode: int,
    source_p0: wp.float64,
    source_p1: wp.float64,
    source_p2: wp.float64,
    source_p3: wp.float64,
    source_p4: wp.float64,
    source_p5: wp.float64,
    reconstruction_id: int,
    normal_jacobian: int,
):
    ell = wp.tid()
    if ell >= n_links:
        return
    i = link_i[ell]
    j = link_j[ell]
    q_out = link_q[ell]
    bid = link_boundary_id[ell]
    if bid < 0 or bid >= MAX_BOUNDARY_IDS:
        bid = 0
    boundary_kind = boundary_kind_by_id[bid]
    boundary_mode = boundary_mode_by_id[bid]
    p0 = boundary_params_by_id[bid, 0]
    p1 = boundary_params_by_id[bid, 1]
    p2 = boundary_params_by_id[bid, 2]
    p3 = boundary_params_by_id[bid, 3]
    p4 = boundary_params_by_id[bid, 4]
    p5 = boundary_params_by_id[bid, 5]
    p6 = boundary_params_by_id[bid, 6]
    p7 = boundary_params_by_id[bid, 7]
    q_in = q_out + 2
    if q_in >= 4:
        q_in = q_in - 4
    eta = link_eta[ell]
    xb = link_xb[ell]
    yb = link_yb[ell]
    nx_b = link_nx[ell]
    ny_b = link_ny[ell]

    # Direction d is the missing incoming direction, from outside to the active
    # node.  e is the outgoing direction from active node to the cut boundary.
    dx_in = wp.float64(CX_I[q_in])
    dy_in = wp.float64(CY_I[q_in])

    # Boundary data.  For Dirichlet these are prescribed velocity components;
    # for Neumann they are nominal traction components.
    bv0 = _curved_boundary_value_comp(
        0, boundary_mode, p0, p1, p2, p3, p4, p5, p6, p7, xb, yb, nx_b, ny_b, time_mid
    )
    bv1 = _curved_boundary_value_comp(
        1, boundary_mode, p0, p1, p2, p3, p4, p5, p6, p7, xb, yb, nx_b, ny_b, time_mid
    )
    if boundary_kind == BC_NEUMANN and use_dynamic_link_traction != 0:
        # Fully local FSI extension: each cut link reads only its own nominal
        # traction, populated by the GPU-resident interface-transfer kernel.
        bv0 = link_traction_x[ell]
        bv1 = link_traction_y[ell]

    # Boundary deformation-gradient predictor.  For the legacy BFL branch this
    # reproduces the original one-link extrapolation.  The newer branches use a
    # quadratic one-sided reconstruction at the actual cut point.
    F11g = U[2, i, j, 0]
    F12g = U[3, i, j, 0]
    F21g = U[4, i, j, 0]
    F22g = U[5, i, j, 0]
    iu = i - CX_I[q_out]
    ju = j - CY_I[q_out]
    upstream_ok = False
    if iu >= 0 and iu < nx and ju >= 0 and ju < ny:
        if active[iu, ju, 0] != 0:
            upstream_ok = True
    if (
        reconstruction_id == BOUNDARY_RECON_BFL
        or reconstruction_id == BOUNDARY_RECON_REG_BFL
        or reconstruction_id == BOUNDARY_RECON_LSQ_BFL
        or reconstruction_id == BOUNDARY_RECON_LOCAL_F_BFL
    ):
        if upstream_ok:
            F11g = U[2, i, j, 0] + eta * (U[2, i, j, 0] - U[2, iu, ju, 0])
            F12g = U[3, i, j, 0] + eta * (U[3, i, j, 0] - U[3, iu, ju, 0])
            F21g = U[4, i, j, 0] + eta * (U[4, i, j, 0] - U[4, iu, ju, 0])
            F22g = U[5, i, j, 0] + eta * (U[5, i, j, 0] - U[5, iu, ju, 0])
        if reconstruction_id == BOUNDARY_RECON_LSQ_BFL:
            F11g = _lsq_U_at_boundary(U, lsq_i, lsq_j, lsq_wb, lsq_valid, ell, 2, F11g)
            F12g = _lsq_U_at_boundary(U, lsq_i, lsq_j, lsq_wb, lsq_valid, ell, 3, F12g)
            F21g = _lsq_U_at_boundary(U, lsq_i, lsq_j, lsq_wb, lsq_valid, ell, 4, F21g)
            F22g = _lsq_U_at_boundary(U, lsq_i, lsq_j, lsq_wb, lsq_valid, ell, 5, F22g)
    else:
        F11g = _interp_U_link(U, active, 2, i, j, q_out, eta, nx, ny)
        F12g = _interp_U_link(U, active, 3, i, j, q_out, eta, nx, ny)
        F21g = _interp_U_link(U, active, 4, i, j, q_out, eta, nx, ny)
        F22g = _interp_U_link(U, active, 5, i, j, q_out, eta, nx, ny)
    if (
        reconstruction_id == BOUNDARY_RECON_LOCAL_CHAR2
        or reconstruction_id == BOUNDARY_RECON_LOCAL_F
    ):
        F11g = _lsq_U_at_boundary(U, lsq_i, lsq_j, lsq_wb, lsq_valid, ell, 2, F11g)
        F12g = _lsq_U_at_boundary(U, lsq_i, lsq_j, lsq_wb, lsq_valid, ell, 3, F12g)
        F21g = _lsq_U_at_boundary(U, lsq_i, lsq_j, lsq_wb, lsq_valid, ell, 4, F21g)
        F22g = _lsq_U_at_boundary(U, lsq_i, lsq_j, lsq_wb, lsq_valid, ell, 5, F22g)

    if (
        reconstruction_id == BOUNDARY_RECON_COMPAT_BFL
        or reconstruction_id == BOUNDARY_RECON_COMPAT_NEE
        or reconstruction_id == BOUNDARY_RECON_LOCAL_CHAR2
    ) and lsq_valid[ell] != 0:
        # Geometry-compatible boundary F.  The normal image F n is extrapolated
        # from the interior predictor.  The tangential image F t is supplied by
        # the boundary displacement: for the clamped inner boundary u_b=0,
        # F_b t = t; for Neumann boundaries, use the current projected
        # displacement tangent derivative, F_b t = t + partial_t u.
        tanx = -ny_b
        tany = nx_b
        dut0 = wp.float64(0.0)
        dut1 = wp.float64(0.0)
        wt_abs = wp.float64(0.0)
        if boundary_kind == BC_NEUMANN:
            for m in range(MAX_LSQ_STENCIL):
                ii_w = lsq_i[ell, m]
                jj_w = lsq_j[ell, m]
                if ii_w >= 0 and jj_w >= 0:
                    ww = lsq_wt[ell, m]
                    dut0 += ww * u[0, ii_w, jj_w, 0]
                    dut1 += ww * u[1, ii_w, jj_w, 0]
                    wt_abs += wp.abs(ww)
        else:
            # Zero clamps have partial_t u_b=0.  For affine velocity Dirichlet
            # data starting from zero displacement, integrate the constant
            # affine velocity in time and use its tangential derivative.
            if boundary_mode == CURVED_VALUE_AFFINE_VELOCITY:
                active_time = time_mid
                if p6 > wp.float64(0.0) and active_time > p6:
                    active_time = p6
                dut0 = active_time * (p0 * tanx + p1 * tany)
                dut1 = active_time * (p3 * tanx + p4 * tany)
            wt_abs = wp.float64(1.0)
        if wt_abs > wp.float64(0.0):
            hn0_tmp = F11g * nx_b + F12g * ny_b
            hn1_tmp = F21g * nx_b + F22g * ny_b
            gt0_tmp = tanx + dut0
            gt1_tmp = tany + dut1
            F11g = hn0_tmp * nx_b + gt0_tmp * tanx
            F12g = hn0_tmp * ny_b + gt0_tmp * tany
            F21g = hn1_tmp * nx_b + gt1_tmp * tanx
            F22g = hn1_tmp * ny_b + gt1_tmp * tany

    Fb = wp.vec4d(F11g, F12g, F21g, F22g)
    if boundary_kind == BC_NEUMANN:
        Fb = _solve_normal_image_arbitrary(
            F11g,
            F12g,
            F21g,
            F22g,
            nx_b,
            ny_b,
            bv0,
            bv1,
            model,
            lam,
            mu,
            mat_p0,
            mat_p1,
            mat_p2,
            j_floor,
            normal_jacobian,
        )

    # Regularized non-equilibrium extrapolation (RNEE) branch.  A ghost state is
    # constructed by reflecting a reconstructed interior mirror state through the
    # boundary datum; the post-collision non-equilibrium part is extrapolated
    # from the adjacent active node after removing the pure D2Q4 ghost moment.
    if reconstruction_id == BOUNDARY_RECON_RNEE:
        sm = wp.float64(2.0) * eta - wp.float64(1.0)
        vm0 = _interp_U_link(U, active, 0, i, j, q_out, sm, nx, ny)
        vm1 = _interp_U_link(U, active, 1, i, j, q_out, sm, nx, ny)
        Fm11 = _interp_U_link(U, active, 2, i, j, q_out, sm, nx, ny)
        Fm12 = _interp_U_link(U, active, 3, i, j, q_out, sm, nx, ny)
        Fm21 = _interp_U_link(U, active, 4, i, j, q_out, sm, nx, ny)
        Fm22 = _interp_U_link(U, active, 5, i, j, q_out, sm, nx, ny)

        vxg = wp.float64(0.0)
        vyg = wp.float64(0.0)
        if boundary_kind == BC_DIRICHLET:
            vxg = wp.float64(2.0) * bv0 - vm0
            vyg = wp.float64(2.0) * bv1 - vm1
        else:
            vb0 = _interp_U_link(U, active, 0, i, j, q_out, eta, nx, ny)
            vb1 = _interp_U_link(U, active, 1, i, j, q_out, eta, nx, ny)
            vxg = wp.float64(2.0) * vb0 - vm0
            vyg = wp.float64(2.0) * vb1 - vm1

        Fg11 = wp.float64(2.0) * Fb[0] - Fm11
        Fg12 = wp.float64(2.0) * Fb[1] - Fm12
        Fg21 = wp.float64(2.0) * Fb[2] - Fm21
        Fg22 = wp.float64(2.0) * Fb[3] - Fm22
        Jg = Fg11 * Fg22 - Fg12 * Fg21
        if Jg <= j_floor:
            # Positivity guard for highly stretched or severely under-resolved
            # cut links.  This preserves the imposed boundary F rather than
            # allowing a ghost extrapolation to create an invalid material state.
            Fg11 = Fb[0]
            Fg12 = Fb[1]
            Fg21 = Fb[2]
            Fg22 = Fb[3]

        vxf = U[0, i, j, 0]
        vyf = U[1, i, j, 0]
        Ff11 = U[2, i, j, 0]
        Ff12 = U[3, i, j, 0]
        Ff21 = U[4, i, j, 0]
        Ff22 = U[5, i, j, 0]
        for a in range(6):
            feq_g = _equilibrium_comp(
                q_in,
                a,
                vxg,
                vyg,
                Fg11,
                Fg12,
                Fg21,
                Fg22,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                lattice_speed,
                j_floor,
            )
            hreg = _regularized_post_neq_comp(
                fpost,
                q_in,
                a,
                i,
                j,
                vxf,
                vyf,
                Ff11,
                Ff12,
                Ff21,
                Ff22,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                lattice_speed,
                j_floor,
            )
            f1[q_in * 6 + a, i, j, 0] = feq_g + hreg
        return

    # Regularized non-equilibrium extrapolation at the cut point.  This branch
    # uses the boundary equilibrium reconstructed from physical data plus a
    # ghost-projected kinetic correction from the adjacent active node.  The
    # pair form applies the same even/odd parity D as the half-way identities;
    # the same-direction form is kept as a diagnostic option.
    if (
        reconstruction_id == BOUNDARY_RECON_PAIR_NEE
        or reconstruction_id == BOUNDARY_RECON_SAME_NEE
        or reconstruction_id == BOUNDARY_RECON_COMPAT_NEE
    ):
        vb0 = bv0
        vb1 = bv1
        if boundary_kind == BC_NEUMANN:
            vb0 = _interp_U_link(U, active, 0, i, j, q_out, eta, nx, ny)
            vb1 = _interp_U_link(U, active, 1, i, j, q_out, eta, nx, ny)
        vxf = U[0, i, j, 0]
        vyf = U[1, i, j, 0]
        Ff11 = U[2, i, j, 0]
        Ff12 = U[3, i, j, 0]
        Ff21 = U[4, i, j, 0]
        Ff22 = U[5, i, j, 0]
        for a in range(6):
            Dp = wp.float64(1.0)
            if boundary_kind == BC_DIRICHLET:
                if a == 0 or a == 1:
                    Dp = wp.float64(-1.0)
            else:
                if a >= 2:
                    Dp = wp.float64(-1.0)
            feq_b = _equilibrium_comp(
                q_in,
                a,
                vb0,
                vb1,
                Fb[0],
                Fb[1],
                Fb[2],
                Fb[3],
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                lattice_speed,
                j_floor,
            )
            hreg = wp.float64(0.0)
            if reconstruction_id == BOUNDARY_RECON_SAME_NEE:
                hreg = _regularized_post_neq_comp(
                    fpost,
                    q_in,
                    a,
                    i,
                    j,
                    vxf,
                    vyf,
                    Ff11,
                    Ff12,
                    Ff21,
                    Ff22,
                    model,
                    lam,
                    mu,
                    mat_p0,
                    mat_p1,
                    mat_p2,
                    lattice_speed,
                    j_floor,
                )
            else:
                hreg = Dp * _regularized_post_neq_comp(
                    fpost,
                    q_out,
                    a,
                    i,
                    j,
                    vxf,
                    vyf,
                    Ff11,
                    Ff12,
                    Ff21,
                    Ff22,
                    model,
                    lam,
                    mu,
                    mat_p0,
                    mat_p1,
                    mat_p2,
                    lattice_speed,
                    j_floor,
                )
            f1[q_in * 6 + a, i, j, 0] = feq_b + hreg
        return

    Pb = _piola_vec(Fb[0], Fb[1], Fb[2], Fb[3], model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor)

    use_char_reg = (
        (
            reconstruction_id == BOUNDARY_RECON_LOCAL_CHAR2
            or reconstruction_id == BOUNDARY_RECON_LOCAL_F
        )
        and boundary_kind == BC_NEUMANN
        and lsq_valid[ell] != 0
    )
    ux0 = wp.float64(0.0)
    ux1 = wp.float64(0.0)
    ux2 = wp.float64(0.0)
    ux3 = wp.float64(0.0)
    ux4 = wp.float64(0.0)
    ux5 = wp.float64(0.0)
    uy0 = wp.float64(0.0)
    uy1 = wp.float64(0.0)
    uy2 = wp.float64(0.0)
    uy3 = wp.float64(0.0)
    uy4 = wp.float64(0.0)
    uy5 = wp.float64(0.0)
    cv0 = U[0, i, j, 0]
    cv1 = U[1, i, j, 0]
    if boundary_kind == BC_DIRICHLET:
        cv0 = bv0
        cv1 = bv1
    cv2 = Fb[0]
    cv3 = Fb[1]
    cv4 = Fb[2]
    cv5 = Fb[3]
    cw0 = wp.float64(0.0)
    cw1 = wp.float64(0.0)
    cw2 = wp.float64(0.0)
    cw3 = wp.float64(0.0)
    cw4 = wp.float64(0.0)
    cw5 = wp.float64(0.0)
    if use_char_reg:
        ux0 = _lsq_U_at_boundary(U, lsq_i, lsq_j, lsq_wx, lsq_valid, ell, 0, wp.float64(0.0))
        ux1 = _lsq_U_at_boundary(U, lsq_i, lsq_j, lsq_wx, lsq_valid, ell, 1, wp.float64(0.0))
        ux2 = _lsq_U_at_boundary(U, lsq_i, lsq_j, lsq_wx, lsq_valid, ell, 2, wp.float64(0.0))
        ux3 = _lsq_U_at_boundary(U, lsq_i, lsq_j, lsq_wx, lsq_valid, ell, 3, wp.float64(0.0))
        ux4 = _lsq_U_at_boundary(U, lsq_i, lsq_j, lsq_wx, lsq_valid, ell, 4, wp.float64(0.0))
        ux5 = _lsq_U_at_boundary(U, lsq_i, lsq_j, lsq_wx, lsq_valid, ell, 5, wp.float64(0.0))
        uy0 = _lsq_U_at_boundary(U, lsq_i, lsq_j, lsq_wy, lsq_valid, ell, 0, wp.float64(0.0))
        uy1 = _lsq_U_at_boundary(U, lsq_i, lsq_j, lsq_wy, lsq_valid, ell, 1, wp.float64(0.0))
        uy2 = _lsq_U_at_boundary(U, lsq_i, lsq_j, lsq_wy, lsq_valid, ell, 2, wp.float64(0.0))
        uy3 = _lsq_U_at_boundary(U, lsq_i, lsq_j, lsq_wy, lsq_valid, ell, 3, wp.float64(0.0))
        uy4 = _lsq_U_at_boundary(U, lsq_i, lsq_j, lsq_wy, lsq_valid, ell, 4, wp.float64(0.0))
        uy5 = _lsq_U_at_boundary(U, lsq_i, lsq_j, lsq_wy, lsq_valid, ell, 5, wp.float64(0.0))
        shift_x = -eta * dt * lattice_speed * wp.float64(CX_I[q_out])
        shift_y = -eta * dt * lattice_speed * wp.float64(CY_I[q_out])
        dv0 = shift_x * ux0 + shift_y * uy0
        dv1 = shift_x * ux1 + shift_y * uy1
        dF11 = shift_x * ux2 + shift_y * uy2
        dF12 = shift_x * ux3 + shift_y * uy3
        dF21 = shift_x * ux4 + shift_y * uy4
        dF22 = shift_x * ux5 + shift_y * uy5
        scale_mech = wp.float64(1.0)
        for _mguard in range(10):
            tF11 = cv2 + scale_mech * dF11
            tF12 = cv3 + scale_mech * dF12
            tF21 = cv4 + scale_mech * dF21
            tF22 = cv5 + scale_mech * dF22
            if tF11 * tF22 - tF12 * tF21 > wp.float64(10.0) * j_floor:
                break
            scale_mech = wp.float64(0.5) * scale_mech
        cv0 = cv0 + scale_mech * dv0
        cv1 = cv1 + scale_mech * dv1
        cv2 = cv2 + scale_mech * dF11
        cv3 = cv3 + scale_mech * dF12
        cv4 = cv4 + scale_mech * dF21
        cv5 = cv5 + scale_mech * dF22
        b0 = _source_comp(
            0,
            source_mode,
            source_p0,
            source_p1,
            source_p2,
            source_p3,
            source_p4,
            source_p5,
            cv0,
            cv1,
        )
        b1 = _source_comp(
            1,
            source_mode,
            source_p0,
            source_p1,
            source_p2,
            source_p3,
            source_p4,
            source_p5,
            cv0,
            cv1,
        )
        b2 = _source_comp(
            2,
            source_mode,
            source_p0,
            source_p1,
            source_p2,
            source_p3,
            source_p4,
            source_p5,
            cv0,
            cv1,
        )
        b3 = _source_comp(
            3,
            source_mode,
            source_p0,
            source_p1,
            source_p2,
            source_p3,
            source_p4,
            source_p5,
            cv0,
            cv1,
        )
        b4 = _source_comp(
            4,
            source_mode,
            source_p0,
            source_p1,
            source_p2,
            source_p3,
            source_p4,
            source_p5,
            cv0,
            cv1,
        )
        b5 = _source_comp(
            5,
            source_mode,
            source_p0,
            source_p1,
            source_p2,
            source_p3,
            source_p4,
            source_p5,
            cv0,
            cv1,
        )
        cw0 = (
            b0
            - _flux_x_jvp_comp(
                0,
                cv0,
                cv1,
                cv2,
                cv3,
                cv4,
                cv5,
                ux0,
                ux1,
                ux2,
                ux3,
                ux4,
                ux5,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
            )
            - _flux_y_jvp_comp(
                0,
                cv0,
                cv1,
                cv2,
                cv3,
                cv4,
                cv5,
                uy0,
                uy1,
                uy2,
                uy3,
                uy4,
                uy5,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
            )
            + lattice_speed * (dx_in * ux0 + dy_in * uy0)
        )
        cw1 = (
            b1
            - _flux_x_jvp_comp(
                1,
                cv0,
                cv1,
                cv2,
                cv3,
                cv4,
                cv5,
                ux0,
                ux1,
                ux2,
                ux3,
                ux4,
                ux5,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
            )
            - _flux_y_jvp_comp(
                1,
                cv0,
                cv1,
                cv2,
                cv3,
                cv4,
                cv5,
                uy0,
                uy1,
                uy2,
                uy3,
                uy4,
                uy5,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
            )
            + lattice_speed * (dx_in * ux1 + dy_in * uy1)
        )
        cw2 = (
            b2
            - _flux_x_jvp_comp(
                2,
                cv0,
                cv1,
                cv2,
                cv3,
                cv4,
                cv5,
                ux0,
                ux1,
                ux2,
                ux3,
                ux4,
                ux5,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
            )
            - _flux_y_jvp_comp(
                2,
                cv0,
                cv1,
                cv2,
                cv3,
                cv4,
                cv5,
                uy0,
                uy1,
                uy2,
                uy3,
                uy4,
                uy5,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
            )
            + lattice_speed * (dx_in * ux2 + dy_in * uy2)
        )
        cw3 = (
            b3
            - _flux_x_jvp_comp(
                3,
                cv0,
                cv1,
                cv2,
                cv3,
                cv4,
                cv5,
                ux0,
                ux1,
                ux2,
                ux3,
                ux4,
                ux5,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
            )
            - _flux_y_jvp_comp(
                3,
                cv0,
                cv1,
                cv2,
                cv3,
                cv4,
                cv5,
                uy0,
                uy1,
                uy2,
                uy3,
                uy4,
                uy5,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
            )
            + lattice_speed * (dx_in * ux3 + dy_in * uy3)
        )
        cw4 = (
            b4
            - _flux_x_jvp_comp(
                4,
                cv0,
                cv1,
                cv2,
                cv3,
                cv4,
                cv5,
                ux0,
                ux1,
                ux2,
                ux3,
                ux4,
                ux5,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
            )
            - _flux_y_jvp_comp(
                4,
                cv0,
                cv1,
                cv2,
                cv3,
                cv4,
                cv5,
                uy0,
                uy1,
                uy2,
                uy3,
                uy4,
                uy5,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
            )
            + lattice_speed * (dx_in * ux4 + dy_in * uy4)
        )
        cw5 = (
            b5
            - _flux_x_jvp_comp(
                5,
                cv0,
                cv1,
                cv2,
                cv3,
                cv4,
                cv5,
                ux0,
                ux1,
                ux2,
                ux3,
                ux4,
                ux5,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
            )
            - _flux_y_jvp_comp(
                5,
                cv0,
                cv1,
                cv2,
                cv3,
                cv4,
                cv5,
                uy0,
                uy1,
                uy2,
                uy3,
                uy4,
                uy5,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
            )
            + lattice_speed * (dx_in * ux5 + dy_in * uy5)
        )

    for a in range(6):
        D = wp.float64(1.0)
        S = wp.float64(0.0)
        if boundary_kind == BC_DIRICHLET:
            if a == 0:
                D = wp.float64(-1.0)
                S = wp.float64(0.5) * bv0
            elif a == 1:
                D = wp.float64(-1.0)
                S = wp.float64(0.5) * bv1
            elif a == 2:
                # F11: d_A Phi_A = -d_X v_x
                D = wp.float64(1.0)
                S = -dx_in * bv0 / lattice_speed
            elif a == 3:
                # F12: d_A Phi_A = -d_Y v_x
                D = wp.float64(1.0)
                S = -dy_in * bv0 / lattice_speed
            elif a == 4:
                D = wp.float64(1.0)
                S = -dx_in * bv1 / lattice_speed
            else:
                D = wp.float64(1.0)
                S = -dy_in * bv1 / lattice_speed
        else:
            # Neumann: velocity rows are flux components along d; F rows impose
            # the boundary state by anti-bounce-back.
            if a == 0:
                D = wp.float64(1.0)
                S = -(dx_in * Pb[0] + dy_in * Pb[1]) / lattice_speed
            elif a == 1:
                D = wp.float64(1.0)
                S = -(dx_in * Pb[2] + dy_in * Pb[3]) / lattice_speed
            elif a == 2:
                D = wp.float64(-1.0)
                S = wp.float64(0.5) * Fb[0]
            elif a == 3:
                D = wp.float64(-1.0)
                S = wp.float64(0.5) * Fb[1]
            elif a == 4:
                D = wp.float64(-1.0)
                S = wp.float64(0.5) * Fb[2]
            else:
                D = wp.float64(-1.0)
                S = wp.float64(0.5) * Fb[3]

        val = wp.float64(0.0)
        f_qout_here = fpost[q_out * 6 + a, i, j, 0]
        f_qin_here = fpost[q_in * 6 + a, i, j, 0]
        use_reg_bfl = False
        if boundary_kind == BC_NEUMANN:
            if reconstruction_id == BOUNDARY_RECON_REG_BFL:
                use_reg_bfl = True
            elif reconstruction_id == BOUNDARY_RECON_LOWQ_REG_BFL and eta < wp.float64(0.5):
                use_reg_bfl = True
        if use_reg_bfl:
            f_qout_here = _regularized_fpost_comp(
                fpost,
                q_out,
                a,
                i,
                j,
                U,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                lattice_speed,
                j_floor,
            )
            f_qin_here = _regularized_fpost_comp(
                fpost,
                q_in,
                a,
                i,
                j,
                U,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                lattice_speed,
                j_floor,
            )

        if (
            reconstruction_id == BOUNDARY_RECON_POLY
            and upstream_ok
            and eta > wp.float64(0.25)
            and eta < wp.float64(0.95)
        ):
            # Quadratic interpolated bounce-back: first reconstruct the incoming
            # value at the physical cut point from the outgoing population, then
            # evaluate the incoming polynomial through s=-1,0,eta at s=1.
            fout_b = _interp_fpost_link(fpost, active, q_out, a, i, j, q_out, eta, nx, ny)
            gin_b = D * fout_b + S
            g0 = f_qin_here
            g1 = fpost[q_in * 6 + a, iu, ju, 0]
            wb = wp.float64(2.0) / (eta * (eta + wp.float64(1.0)))
            w0 = -wp.float64(2.0) * (wp.float64(1.0) - eta) / eta
            w1 = (wp.float64(1.0) - eta) / (wp.float64(1.0) + eta)
            val = wb * gin_b + w0 * g0 + w1 * g1
        elif reconstruction_id == BOUNDARY_RECON_QBFL and eta < wp.float64(0.5):
            # Quadratic interpolation of the same BFL characteristic point
            # s=2*eta-1.  This branch degrades automatically near one-cell-thick
            # geometries if the second upstream node is unavailable.
            s_eval = wp.float64(2.0) * eta - wp.float64(1.0)
            interp = _interp_fpost_link(fpost, active, q_out, a, i, j, q_out, s_eval, nx, ny)
            val = D * interp + S
        elif eta >= wp.float64(0.5):
            alpha = wp.float64(1.0) / (wp.float64(2.0) * eta)
            val = alpha * (D * f_qout_here + S) + (wp.float64(1.0) - alpha) * f_qin_here
        else:
            upstream_val = f_qout_here
            if upstream_ok:
                if use_reg_bfl:
                    upstream_val = _regularized_fpost_comp(
                        fpost,
                        q_out,
                        a,
                        iu,
                        ju,
                        U,
                        model,
                        lam,
                        mu,
                        mat_p0,
                        mat_p1,
                        mat_p2,
                        lattice_speed,
                        j_floor,
                    )
                else:
                    upstream_val = fpost[q_out * 6 + a, iu, ju, 0]
            val = (
                D
                * (
                    wp.float64(2.0) * eta * f_qout_here
                    + (wp.float64(1.0) - wp.float64(2.0) * eta) * upstream_val
                )
                + S
            )
        if use_char_reg:
            if cv2 * cv5 - cv3 * cv4 > j_floor:
                eq_jvp = _equilibrium_jvp_comp(
                    q_in,
                    a,
                    cv0,
                    cv1,
                    cv2,
                    cv3,
                    cv4,
                    cv5,
                    cw0,
                    cw1,
                    cw2,
                    cw3,
                    cw4,
                    cw5,
                    model,
                    lam,
                    mu,
                    mat_p0,
                    mat_p1,
                    mat_p2,
                    lattice_speed,
                    j_floor,
                )
                geq_b = _equilibrium_comp(
                    q_in,
                    a,
                    cv0,
                    cv1,
                    cv2,
                    cv3,
                    cv4,
                    cv5,
                    model,
                    lam,
                    mu,
                    mat_p0,
                    mat_p1,
                    mat_p2,
                    lattice_speed,
                    j_floor,
                )
                source_a = _source_comp(
                    a,
                    source_mode,
                    source_p0,
                    source_p1,
                    source_p2,
                    source_p3,
                    source_p4,
                    source_p5,
                    cv0,
                    cv1,
                )
                val = (
                    geq_b
                    - dt / collision_omega * eq_jvp
                    + dt
                    * (wp.float64(1.0) / collision_omega - wp.float64(0.5))
                    * wp.float64(0.25)
                    * source_a
                )
        f1[q_in * 6 + a, i, j, 0] = val


@wp.kernel
def _kinetic_ghost_filter_kernel(
    f: wp.array4d(dtype=wp.float64),
    active: wp.array3d(dtype=wp.int32),
    strength: wp.float64,
):
    """Damp the D2Q4 horizontal/vertical ghost moment without changing U or flux moments."""
    i, j, k = wp.tid()
    if active[i, j, 0] == 0:
        return
    if strength <= wp.float64(0.0):
        return
    for a in range(6):
        idxE = 0 * 6 + a
        idxN = 1 * 6 + a
        idxW = 2 * 6 + a
        idxS = 3 * 6 + a
        m3 = f[idxE, i, j, k] - f[idxN, i, j, k] + f[idxW, i, j, k] - f[idxS, i, j, k]
        delta = -strength * m3
        quarter = wp.float64(0.25) * delta
        f[idxE, i, j, k] = f[idxE, i, j, k] + quarter
        f[idxW, i, j, k] = f[idxW, i, j, k] + quarter
        f[idxN, i, j, k] = f[idxN, i, j, k] - quarter
        f[idxS, i, j, k] = f[idxS, i, j, k] - quarter


@wp.kernel
def _refresh_masked_kernel(
    f0: wp.array4d(dtype=wp.float64),
    u: wp.array4d(dtype=wp.float64),
    u_star: wp.array4d(dtype=wp.float64),
    U: wp.array4d(dtype=wp.float64),
    P_field: wp.array4d(dtype=wp.float64),
    sigma: wp.array4d(dtype=wp.float64),
    active: wp.array3d(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
    dt: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
    source_mode: int,
    source_p0: wp.float64,
    source_p1: wp.float64,
    source_p2: wp.float64,
    source_p3: wp.float64,
    source_p4: wp.float64,
    source_p5: wp.float64,
):
    i, j, k = wp.tid()
    if active[i, j, 0] == 0:
        for a in range(6):
            U[a, i, j, k] = wp.float64(0.0)
        u[0, i, j, k] = wp.float64(0.0)
        u[1, i, j, k] = wp.float64(0.0)
        for a in range(4):
            P_field[a, i, j, k] = wp.float64(0.0)
            sigma[a, i, j, k] = wp.float64(0.0)
        return

    vx = wp.float64(0.0)
    vy = wp.float64(0.0)
    F11 = wp.float64(0.0)
    F12 = wp.float64(0.0)
    F21 = wp.float64(0.0)
    F22 = wp.float64(0.0)
    for q in range(4):
        vx += f0[q * 6 + 0, i, j, k]
        vy += f0[q * 6 + 1, i, j, k]
        F11 += f0[q * 6 + 2, i, j, k]
        F12 += f0[q * 6 + 3, i, j, k]
        F21 += f0[q * 6 + 4, i, j, k]
        F22 += f0[q * 6 + 5, i, j, k]
    half_dt = wp.float64(0.5) * dt
    if source_mode == SOURCE_CONSTANT:
        vx += half_dt * source_p0
        vy += half_dt * source_p1
        F11 += half_dt * source_p2
        F12 += half_dt * source_p3
        F21 += half_dt * source_p4
        F22 += half_dt * source_p5
    elif source_mode == SOURCE_DAMPING:
        denom = wp.float64(1.0) + half_dt * source_p0
        vx = vx / denom
        vy = vy / denom

    U[0, i, j, k] = vx
    U[1, i, j, k] = vy
    U[2, i, j, k] = F11
    U[3, i, j, k] = F12
    U[4, i, j, k] = F21
    U[5, i, j, k] = F22
    u[0, i, j, k] = u_star[0, i, j, k] + wp.float64(0.5) * dt * vx
    u[1, i, j, k] = u_star[1, i, j, k] + wp.float64(0.5) * dt * vy
    J = F11 * F22 - F12 * F21
    if J <= j_floor:
        invalid[0] = wp.int32(1)
    J_safe = _clamp_j(J, j_floor)
    P = _piola_vec(F11, F12, F21, F22, model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor)
    P_field[0, i, j, k] = P[0]
    P_field[1, i, j, k] = P[1]
    P_field[2, i, j, k] = P[2]
    P_field[3, i, j, k] = P[3]
    sigma[0, i, j, k] = (P[0] * F11 + P[1] * F12) / J_safe
    sigma[1, i, j, k] = (P[0] * F21 + P[1] * F22) / J_safe
    sigma[2, i, j, k] = (P[2] * F11 + P[3] * F12) / J_safe
    sigma[3, i, j, k] = (P[2] * F21 + P[3] * F22) / J_safe


@wp.func
def _local_dudx(
    u: wp.array4d(dtype=wp.float64),
    active: wp.array3d(dtype=wp.int32),
    comp: int,
    i: int,
    j: int,
    nx: int,
    dx: wp.float64,
) -> wp.float64:
    if i > 0 and i + 1 < nx and active[i - 1, j, 0] != 0 and active[i + 1, j, 0] != 0:
        return (u[comp, i + 1, j, 0] - u[comp, i - 1, j, 0]) / (wp.float64(2.0) * dx)
    if i + 1 < nx and active[i + 1, j, 0] != 0:
        if i + 2 < nx and active[i + 2, j, 0] != 0:
            return (
                -wp.float64(3.0) * u[comp, i, j, 0]
                + wp.float64(4.0) * u[comp, i + 1, j, 0]
                - u[comp, i + 2, j, 0]
            ) / (wp.float64(2.0) * dx)
        return (u[comp, i + 1, j, 0] - u[comp, i, j, 0]) / dx
    if i > 0 and active[i - 1, j, 0] != 0:
        if i > 1 and active[i - 2, j, 0] != 0:
            return (
                wp.float64(3.0) * u[comp, i, j, 0]
                - wp.float64(4.0) * u[comp, i - 1, j, 0]
                + u[comp, i - 2, j, 0]
            ) / (wp.float64(2.0) * dx)
        return (u[comp, i, j, 0] - u[comp, i - 1, j, 0]) / dx
    return wp.float64(0.0)


@wp.func
def _local_dudy(
    u: wp.array4d(dtype=wp.float64),
    active: wp.array3d(dtype=wp.int32),
    comp: int,
    i: int,
    j: int,
    ny: int,
    dy: wp.float64,
) -> wp.float64:
    if j > 0 and j + 1 < ny and active[i, j - 1, 0] != 0 and active[i, j + 1, 0] != 0:
        return (u[comp, i, j + 1, 0] - u[comp, i, j - 1, 0]) / (wp.float64(2.0) * dy)
    if j + 1 < ny and active[i, j + 1, 0] != 0:
        if j + 2 < ny and active[i, j + 2, 0] != 0:
            return (
                -wp.float64(3.0) * u[comp, i, j, 0]
                + wp.float64(4.0) * u[comp, i, j + 1, 0]
                - u[comp, i, j + 2, 0]
            ) / (wp.float64(2.0) * dy)
        return (u[comp, i, j + 1, 0] - u[comp, i, j, 0]) / dy
    if j > 0 and active[i, j - 1, 0] != 0:
        if j > 1 and active[i, j - 2, 0] != 0:
            return (
                wp.float64(3.0) * u[comp, i, j, 0]
                - wp.float64(4.0) * u[comp, i, j - 1, 0]
                + u[comp, i, j - 2, 0]
            ) / (wp.float64(2.0) * dy)
        return (u[comp, i, j, 0] - u[comp, i, j - 1, 0]) / dy
    return wp.float64(0.0)


@wp.kernel
def _local_displacement_bc_clear_kernel(
    bc_diag: wp.array3d(dtype=wp.float64),
    bc_rhs: wp.array4d(dtype=wp.float64),
):
    """Clear the node-local Dirichlet contributions for one explicit sweep."""
    i, j, k = wp.tid()
    bc_diag[i, j, k] = wp.float64(0.0)
    bc_rhs[0, i, j, k] = wp.float64(0.0)
    bc_rhs[1, i, j, k] = wp.float64(0.0)


@wp.kernel
def _local_displacement_bc_kernel(
    U: wp.array4d(dtype=wp.float64),
    active: wp.array3d(dtype=wp.int32),
    link_i: wp.array(dtype=wp.int32),
    link_j: wp.array(dtype=wp.int32),
    link_boundary_id: wp.array(dtype=wp.int32),
    link_xb: wp.array(dtype=wp.float64),
    link_yb: wp.array(dtype=wp.float64),
    link_nx: wp.array(dtype=wp.float64),
    link_ny: wp.array(dtype=wp.float64),
    boundary_kind_by_id: wp.array(dtype=wp.int32),
    boundary_mode_by_id: wp.array(dtype=wp.int32),
    boundary_params_by_id: wp.array2d(dtype=wp.float64),
    n_links: int,
    dirichlet_boundary_id: int,
    time: wp.float64,
    dx: wp.float64,
    dy: wp.float64,
    weight: wp.float64,
    bc_diag: wp.array3d(dtype=wp.float64),
    bc_rhs: wp.array4d(dtype=wp.float64),
):
    """Accumulate cut-point anchors; every write touches only its cut cell."""
    ell = wp.tid()
    if ell >= n_links:
        return
    bid = link_boundary_id[ell]
    if dirichlet_boundary_id >= 0:
        if bid != dirichlet_boundary_id:
            return
    elif boundary_kind_by_id[bid] != BC_DIRICHLET:
        return
    i = link_i[ell]
    j = link_j[ell]
    if active[i, j, 0] == 0:
        return

    p0 = boundary_params_by_id[bid, 0]
    p1 = boundary_params_by_id[bid, 1]
    p2 = boundary_params_by_id[bid, 2]
    p3 = boundary_params_by_id[bid, 3]
    p4 = boundary_params_by_id[bid, 4]
    p5 = boundary_params_by_id[bid, 5]
    p6 = boundary_params_by_id[bid, 6]
    p7 = boundary_params_by_id[bid, 7]
    mode = boundary_mode_by_id[bid]
    ub0 = wp.float64(0.0)
    ub1 = wp.float64(0.0)
    if mode == CURVED_VALUE_CONSTANT_VECTOR:
        ub0 = time * _curved_boundary_value_comp(
            0,
            mode,
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            link_xb[ell],
            link_yb[ell],
            link_nx[ell],
            link_ny[ell],
            wp.float64(0.0),
        )
        ub1 = time * _curved_boundary_value_comp(
            1,
            mode,
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            link_xb[ell],
            link_yb[ell],
            link_nx[ell],
            link_ny[ell],
            wp.float64(0.0),
        )
    elif mode == CURVED_VALUE_AFFINE_VELOCITY:
        active_time = time
        if p6 > wp.float64(0.0) and active_time > p6:
            active_time = p6
        ub0 = active_time * _curved_boundary_value_comp(
            0,
            mode,
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            link_xb[ell],
            link_yb[ell],
            link_nx[ell],
            link_ny[ell],
            wp.float64(0.0),
        )
        ub1 = active_time * _curved_boundary_value_comp(
            1,
            mode,
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            link_xb[ell],
            link_yb[ell],
            link_nx[ell],
            link_ny[ell],
            wp.float64(0.0),
        )

    dbx = link_xb[ell] - (wp.float64(i) + wp.float64(0.5)) * dx
    dby = link_yb[ell] - (wp.float64(j) + wp.float64(0.5)) * dy
    tx = ub0 - ((U[2, i, j, 0] - wp.float64(1.0)) * dbx + U[3, i, j, 0] * dby)
    ty = ub1 - (U[4, i, j, 0] * dbx + (U[5, i, j, 0] - wp.float64(1.0)) * dby)
    w2 = weight * weight
    wp.atomic_add(bc_diag, i, j, 0, w2)
    wp.atomic_add(bc_rhs, 0, i, j, 0, w2 * tx)
    wp.atomic_add(bc_rhs, 1, i, j, 0, w2 * ty)


@wp.kernel
def _local_displacement_relax_kernel(
    U: wp.array4d(dtype=wp.float64),
    active: wp.array3d(dtype=wp.int32),
    u: wp.array4d(dtype=wp.float64),
    u_next: wp.array4d(dtype=wp.float64),
    bc_diag: wp.array3d(dtype=wp.float64),
    bc_rhs: wp.array4d(dtype=wp.float64),
    nx: int,
    ny: int,
    dx: wp.float64,
    dy: wp.float64,
    relax: wp.float64,
):
    """One explicit nearest-neighbour compatibility relaxation, never a solve."""
    i, j, k = wp.tid()
    if active[i, j, 0] == 0:
        u_next[0, i, j, k] = wp.float64(0.0)
        u_next[1, i, j, k] = wp.float64(0.0)
        return
    diag = bc_diag[i, j, 0]
    rx = bc_rhs[0, i, j, 0]
    ry = bc_rhs[1, i, j, 0]
    if i + 1 < nx and active[i + 1, j, 0] != 0:
        gx = (
            wp.float64(0.5)
            * dx
            * ((U[2, i, j, 0] - wp.float64(1.0)) + (U[2, i + 1, j, 0] - wp.float64(1.0)))
        )
        gy = wp.float64(0.5) * dx * (U[4, i, j, 0] + U[4, i + 1, j, 0])
        rx += u[0, i + 1, j, k] - gx
        ry += u[1, i + 1, j, k] - gy
        diag += wp.float64(1.0)
    if i > 0 and active[i - 1, j, 0] != 0:
        gx = (
            wp.float64(0.5)
            * dx
            * ((U[2, i - 1, j, 0] - wp.float64(1.0)) + (U[2, i, j, 0] - wp.float64(1.0)))
        )
        gy = wp.float64(0.5) * dx * (U[4, i - 1, j, 0] + U[4, i, j, 0])
        rx += u[0, i - 1, j, k] + gx
        ry += u[1, i - 1, j, k] + gy
        diag += wp.float64(1.0)
    if j + 1 < ny and active[i, j + 1, 0] != 0:
        gx = wp.float64(0.5) * dy * (U[3, i, j, 0] + U[3, i, j + 1, 0])
        gy = (
            wp.float64(0.5)
            * dy
            * ((U[5, i, j, 0] - wp.float64(1.0)) + (U[5, i, j + 1, 0] - wp.float64(1.0)))
        )
        rx += u[0, i, j + 1, k] - gx
        ry += u[1, i, j + 1, k] - gy
        diag += wp.float64(1.0)
    if j > 0 and active[i, j - 1, 0] != 0:
        gx = wp.float64(0.5) * dy * (U[3, i, j - 1, 0] + U[3, i, j, 0])
        gy = (
            wp.float64(0.5)
            * dy
            * ((U[5, i, j - 1, 0] - wp.float64(1.0)) + (U[5, i, j, 0] - wp.float64(1.0)))
        )
        rx += u[0, i, j - 1, k] + gx
        ry += u[1, i, j - 1, k] + gy
        diag += wp.float64(1.0)
    if diag > wp.float64(0.0):
        target0 = rx / diag
        target1 = ry / diag
        u_next[0, i, j, k] = u[0, i, j, k] + relax * (target0 - u[0, i, j, k])
        u_next[1, i, j, k] = u[1, i, j, k] + relax * (target1 - u[1, i, j, k])
    else:
        u_next[0, i, j, k] = u[0, i, j, k]
        u_next[1, i, j, k] = u[1, i, j, k]


@wp.kernel
def _local_displacement_commit_kernel(
    active: wp.array3d(dtype=wp.int32),
    u_next: wp.array4d(dtype=wp.float64),
    u: wp.array4d(dtype=wp.float64),
    u_star: wp.array4d(dtype=wp.float64),
):
    """Translate both displacement time levels by the same local correction.

    Unlike an occasional global projection, a correction applied every physical
    step must preserve the trapezoidal half-step phase.  Resetting ``u_star``
    from ``u - dt*v/2`` here would cancel the next step's displacement increment.
    """
    i, j, k = wp.tid()
    if active[i, j, 0] == 0:
        u[0, i, j, k] = wp.float64(0.0)
        u[1, i, j, k] = wp.float64(0.0)
        u_star[0, i, j, k] = wp.float64(0.0)
        u_star[1, i, j, k] = wp.float64(0.0)
        return
    delta0 = u_next[0, i, j, k] - u[0, i, j, k]
    delta1 = u_next[1, i, j, k] - u[1, i, j, k]
    u[0, i, j, k] = u_next[0, i, j, k]
    u[1, i, j, k] = u_next[1, i, j, k]
    u_star[0, i, j, k] = u_star[0, i, j, k] + delta0
    u_star[1, i, j, k] = u_star[1, i, j, k] + delta1


@wp.kernel
def _local_compatibility_repair_kernel(
    f: wp.array4d(dtype=wp.float64),
    U: wp.array4d(dtype=wp.float64),
    P_field: wp.array4d(dtype=wp.float64),
    sigma: wp.array4d(dtype=wp.float64),
    active: wp.array3d(dtype=wp.int32),
    u: wp.array4d(dtype=wp.float64),
    nx: int,
    ny: int,
    dx: wp.float64,
    dy: wp.float64,
    blend: wp.float64,
    interior_only: int,
    lattice_speed: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
):
    """Fixed-stencil GPU repair; no graph solve or domain-wide dependency."""
    i, j, k = wp.tid()
    if active[i, j, 0] == 0:
        return
    if interior_only != 0:
        if i == 0 or i + 1 >= nx or j == 0 or j + 1 >= ny:
            return
        if (
            active[i - 1, j, 0] == 0
            or active[i + 1, j, 0] == 0
            or active[i, j - 1, 0] == 0
            or active[i, j + 1, 0] == 0
        ):
            return
    o11 = U[2, i, j, 0]
    o12 = U[3, i, j, 0]
    o21 = U[4, i, j, 0]
    o22 = U[5, i, j, 0]
    t11 = wp.float64(1.0) + _local_dudx(u, active, 0, i, j, nx, dx)
    t12 = _local_dudy(u, active, 0, i, j, ny, dy)
    t21 = _local_dudx(u, active, 1, i, j, nx, dx)
    t22 = wp.float64(1.0) + _local_dudy(u, active, 1, i, j, ny, dy)
    alpha = wp.max(wp.float64(0.0), wp.min(wp.float64(1.0), blend))
    n11 = o11
    n12 = o12
    n21 = o21
    n22 = o22
    for _guard in range(10):
        n11 = o11 + alpha * (t11 - o11)
        n12 = o12 + alpha * (t12 - o12)
        n21 = o21 + alpha * (t21 - o21)
        n22 = o22 + alpha * (t22 - o22)
        if n11 * n22 - n12 * n21 > j_floor:
            break
        alpha = wp.float64(0.5) * alpha
    # Repair only the equilibrium moments, but evaluate the nonlinear stress
    # once per old/new state instead of once for every (q,a) population.  This
    # is algebraically identical to ``feq(new)-feq(old)``: velocity is fixed,
    # so only the Piola flux in rows 0--1 and the conserved F entries in rows
    # 2--5 change.
    old_P = _piola_vec(o11, o12, o21, o22, model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor)
    new_P = _piola_vec(n11, n12, n21, n22, model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor)
    dP11 = new_P[0] - old_P[0]
    dP12 = new_P[1] - old_P[1]
    dP21 = new_P[2] - old_P[2]
    dP22 = new_P[3] - old_P[3]
    flux_factor = wp.float64(0.5) / lattice_speed
    quarter = wp.float64(0.25)
    for q in range(4):
        cx = wp.float64(CX_I[q])
        cy = wp.float64(CY_I[q])
        base = q * 6
        f[base + 0, i, j, k] = f[base + 0, i, j, k] - flux_factor * (cx * dP11 + cy * dP12)
        f[base + 1, i, j, k] = f[base + 1, i, j, k] - flux_factor * (cx * dP21 + cy * dP22)
        f[base + 2, i, j, k] = f[base + 2, i, j, k] + quarter * (n11 - o11)
        f[base + 3, i, j, k] = f[base + 3, i, j, k] + quarter * (n12 - o12)
        f[base + 4, i, j, k] = f[base + 4, i, j, k] + quarter * (n21 - o21)
        f[base + 5, i, j, k] = f[base + 5, i, j, k] + quarter * (n22 - o22)
    U[2, i, j, k] = n11
    U[3, i, j, k] = n12
    U[4, i, j, k] = n21
    U[5, i, j, k] = n22
    J = n11 * n22 - n12 * n21
    P_field[0, i, j, k] = new_P[0]
    P_field[1, i, j, k] = new_P[1]
    P_field[2, i, j, k] = new_P[2]
    P_field[3, i, j, k] = new_P[3]
    Jsafe = _clamp_j(J, j_floor)
    sigma[0, i, j, k] = (new_P[0] * n11 + new_P[1] * n12) / Jsafe
    sigma[1, i, j, k] = (new_P[0] * n21 + new_P[1] * n22) / Jsafe
    sigma[2, i, j, k] = (new_P[2] * n11 + new_P[3] * n12) / Jsafe
    sigma[3, i, j, k] = (new_P[2] * n21 + new_P[3] * n22) / Jsafe


class WarpCurvedBoundaryLBM2D:
    """Warp-backed vectorial hyperelastic LBM on an implicit 2-D domain."""

    def __init__(
        self,
        geometry: CurvedBoundaryGeometry,
        *,
        material: WarpHyperelasticMaterial,
        boundary: CurvedBoundarySpec,
        lattice_speed: float = 5.0,
        collision_omega: float = 2.0,
        collision_model: Literal["bgk", "mrt_ghost"] = "bgk",
        ghost_omega: float | None = None,
        kinetic_filter_strength: float = 0.0,
        boundary_reconstruction: Literal[
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
        ] = "bfl",
        device: str | None = None,
        check_linearized_cfl: bool = True,
        jacobian_floor: float = 1.0e-12,
        source_mode: Literal["none", "constant", "damping"] = "none",
        source_vector: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        damping_gamma: float = 0.0,
        compatibility_projection_interval: int = 0,
        compatibility_projection_iterations: int = 160,
        compatibility_projection_relax: float = 0.85,
        compatibility_projection_weight: float = 20.0,
        compatibility_projection_repair_F: bool = False,
        compatibility_projection_repair_F_band_width: int = -1,
        compatibility_projection_repair_F_blend: float = 1.0,
        compatibility_projection_boundary_id: int = BOUNDARY_ID_INNER,
        local_compatibility_interval: int = 0,
        local_compatibility_blend: float = 0.05,
        local_compatibility_interior_only: bool = True,
        local_displacement_interval: int = 0,
        local_displacement_sweeps: int = 1,
        local_displacement_relax: float = 0.85,
        local_displacement_boundary_weight: float = 20.0,
        local_displacement_boundary_id: int = BOUNDARY_ID_INNER,
        normal_jacobian: Literal["finite_difference", "acoustic"] | None = None,
        neumann_solver: Literal["newton", "energy"] | None = None,
    ) -> None:
        self.geometry = geometry
        self.nx = int(geometry.nx)
        self.ny = int(geometry.ny)
        self.length_x = float(geometry.length_x)
        self.length_y = float(geometry.length_y)
        self.dx = geometry.dx
        self.dy = geometry.dy
        self.material = material
        self.boundary = boundary
        self.lattice_speed = float(lattice_speed)
        self.collision_omega = float(collision_omega)
        if not (0.0 < self.collision_omega <= 2.0):
            raise ValueError("collision_omega must be in (0,2]")
        if collision_model not in ("bgk", "mrt_ghost"):
            raise ValueError("collision_model must be 'bgk' or 'mrt_ghost'")
        self.collision_model = str(collision_model)
        self.ghost_omega = float(self.collision_omega if ghost_omega is None else ghost_omega)
        if not (0.0 < self.ghost_omega <= 2.0):
            raise ValueError("ghost_omega must be in (0,2]")
        self.kinetic_filter_strength = float(kinetic_filter_strength)
        if not (0.0 <= self.kinetic_filter_strength <= 1.0):
            raise ValueError("kinetic_filter_strength must be in [0,1]")
        self.boundary_reconstruction = boundary_reconstruction_id(str(boundary_reconstruction))
        self.boundary_reconstruction_name = boundary_reconstruction_name(
            self.boundary_reconstruction
        )
        self.boundary_stencil_radius = boundary_reconstruction_stencil_radius(
            self.boundary_reconstruction
        )
        self.normal_jacobian = normal_jacobian_id(normal_jacobian)
        self.normal_jacobian_name = normal_jacobian_name(self.normal_jacobian)
        self.neumann_solver_name = neumann_solver_name(neumann_solver, material)
        self._normal_solver_mode = (
            NORMAL_SOLVER_ENERGY if self.neumann_solver_name == "energy" else self.normal_jacobian
        )
        if (
            check_linearized_cfl
            and self.material.linearized_stability_ratio(self.lattice_speed) >= 1.0
        ):
            raise ValueError("linearized D2Q4 stability ratio must be below one")
        self.dt = self.dx / self.lattice_speed
        self.time = 0.0
        self.steps = 0
        self.device = device or default_device()
        self.shape = (self.nx, self.ny, 1)
        self.jacobian_floor = float(jacobian_floor)

        from .vector_nonlinear_elastic_warp import source_id, source_name

        self._source_name_func = source_name
        self.source_mode = source_id(source_mode)
        self.source_params = np.zeros(NCOMP, dtype=np.float64)
        if self.source_mode == SOURCE_CONSTANT:
            for idx, value in enumerate(tuple(source_vector)[:NCOMP]):
                self.source_params[idx] = float(value)
        elif self.source_mode == SOURCE_DAMPING:
            if damping_gamma < 0.0:
                raise ValueError("damping_gamma must be non-negative")
            self.source_params[0] = float(damping_gamma)

        if int(compatibility_projection_interval) != 0:
            raise ValueError(
                "the fully-local solver forbids compatibility_projection_interval != 0"
            )
        self.compatibility_projection_interval = 0
        self.compatibility_projection_iterations = int(max(1, compatibility_projection_iterations))
        self.compatibility_projection_relax = float(compatibility_projection_relax)
        self.compatibility_projection_weight = float(compatibility_projection_weight)
        self.compatibility_projection_repair_F = False
        self.compatibility_projection_repair_F_band_width = int(
            compatibility_projection_repair_F_band_width
        )
        self.compatibility_projection_repair_F_blend = float(
            compatibility_projection_repair_F_blend
        )
        self.compatibility_projection_boundary_id = int(compatibility_projection_boundary_id)
        self._compatibility_projector = None
        self.local_compatibility_interval = int(max(0, local_compatibility_interval))
        self.local_compatibility_blend = float(local_compatibility_blend)
        self.local_compatibility_interior_only = bool(local_compatibility_interior_only)
        if not (0.0 <= self.local_compatibility_blend <= 1.0):
            raise ValueError("local_compatibility_blend must be in [0,1]")
        self.local_displacement_interval = int(max(0, local_displacement_interval))
        self.local_displacement_sweeps = int(local_displacement_sweeps)
        self.local_displacement_relax = float(local_displacement_relax)
        self.local_displacement_boundary_weight = float(local_displacement_boundary_weight)
        self.local_displacement_boundary_id = int(local_displacement_boundary_id)
        if not (0.0 <= self.local_displacement_relax <= 1.0):
            raise ValueError("local_displacement_relax must be in [0,1]")
        if not (1 <= self.local_displacement_sweeps <= 8):
            raise ValueError(
                "local_displacement_sweeps must be in [1,8] to keep a bounded local dependency radius"
            )
        if self.local_displacement_boundary_weight < 0.0:
            raise ValueError("local_displacement_boundary_weight must be non-negative")

        x = (np.arange(self.nx, dtype=np.float64) + 0.5) * self.dx
        y = (np.arange(self.ny, dtype=np.float64) + 0.5) * self.dy
        self.x, self.y = np.meshgrid(x, y, indexing="ij")
        self.active_np = np.ascontiguousarray(geometry.active.astype(np.int32))
        self.active = _as_warp(self.active_np[..., None], dtype=wp.int32, device=self.device)
        self.link_i = _as_warp(geometry.link_i, dtype=wp.int32, device=self.device)
        self.link_j = _as_warp(geometry.link_j, dtype=wp.int32, device=self.device)
        self.link_q = _as_warp(geometry.link_q, dtype=wp.int32, device=self.device)
        self.link_boundary_id = _as_warp(
            geometry.link_boundary_id, dtype=wp.int32, device=self.device
        )
        self.link_eta = _as_warp(geometry.link_eta, dtype=wp.float64, device=self.device)
        self.link_xb = _as_warp(geometry.link_xb, dtype=wp.float64, device=self.device)
        self.link_yb = _as_warp(geometry.link_yb, dtype=wp.float64, device=self.device)
        self.link_nx = _as_warp(geometry.link_nx, dtype=wp.float64, device=self.device)
        self.link_ny = _as_warp(geometry.link_ny, dtype=wp.float64, device=self.device)
        if (
            geometry.lsq_i is None
            or geometry.lsq_j is None
            or geometry.lsq_wb is None
            or geometry.lsq_valid is None
        ):
            lsq_i_np = np.full((geometry.n_links, MAX_LSQ_STENCIL), -1, dtype=np.int32)
            lsq_j_np = np.full((geometry.n_links, MAX_LSQ_STENCIL), -1, dtype=np.int32)
            lsq_wb_np = np.zeros((geometry.n_links, MAX_LSQ_STENCIL), dtype=np.float64)
            lsq_wt_np = np.zeros((geometry.n_links, MAX_LSQ_STENCIL), dtype=np.float64)
            lsq_wx_np = np.zeros((geometry.n_links, MAX_LSQ_STENCIL), dtype=np.float64)
            lsq_wy_np = np.zeros((geometry.n_links, MAX_LSQ_STENCIL), dtype=np.float64)
            lsq_valid_np = np.zeros(geometry.n_links, dtype=np.int32)
        else:
            lsq_i_np = geometry.lsq_i
            lsq_j_np = geometry.lsq_j
            lsq_wb_np = geometry.lsq_wb
            lsq_wt_np = geometry.lsq_wt if geometry.lsq_wt is not None else np.zeros_like(lsq_wb_np)
            lsq_wx_np = geometry.lsq_wx if geometry.lsq_wx is not None else np.zeros_like(lsq_wb_np)
            lsq_wy_np = geometry.lsq_wy if geometry.lsq_wy is not None else np.zeros_like(lsq_wb_np)
            lsq_valid_np = geometry.lsq_valid
        self.lsq_i = _as_warp(lsq_i_np, dtype=wp.int32, device=self.device)
        self.lsq_j = _as_warp(lsq_j_np, dtype=wp.int32, device=self.device)
        self.lsq_wb = _as_warp(lsq_wb_np, dtype=wp.float64, device=self.device)
        self.lsq_wt = _as_warp(lsq_wt_np, dtype=wp.float64, device=self.device)
        self.lsq_wx = _as_warp(lsq_wx_np, dtype=wp.float64, device=self.device)
        self.lsq_wy = _as_warp(lsq_wy_np, dtype=wp.float64, device=self.device)
        self.lsq_valid = _as_warp(lsq_valid_np, dtype=wp.int32, device=self.device)
        self.lsq_wb_np = np.asarray(lsq_wb_np, dtype=np.float64)
        self.lsq_wt_np = np.asarray(lsq_wt_np, dtype=np.float64)
        self.lsq_wx_np = np.asarray(lsq_wx_np, dtype=np.float64)
        self.lsq_wy_np = np.asarray(lsq_wy_np, dtype=np.float64)
        self.lsq_valid_np = np.asarray(lsq_valid_np, dtype=np.int32)
        self.boundary_params = boundary.params8
        kind_by_id, mode_by_id, params_by_id = boundary.tables(MAX_BOUNDARY_IDS)
        self.boundary_kind_by_id_np = kind_by_id
        self.boundary_mode_by_id_np = mode_by_id
        self.boundary_params_by_id_np = params_by_id
        self.boundary_kind_by_id = _as_warp(kind_by_id, dtype=wp.int32, device=self.device)
        self.boundary_mode_by_id = _as_warp(mode_by_id, dtype=wp.int32, device=self.device)
        self.boundary_params_by_id = _as_warp(params_by_id, dtype=wp.float64, device=self.device)
        self.link_traction_x = wp.zeros(geometry.n_links, dtype=wp.float64, device=self.device)
        self.link_traction_y = wp.zeros(geometry.n_links, dtype=wp.float64, device=self.device)
        self.use_dynamic_link_traction = False

        self.f0 = wp.zeros((FQ, self.nx, self.ny, 1), dtype=wp.float64, device=self.device)
        self.f1 = wp.zeros((FQ, self.nx, self.ny, 1), dtype=wp.float64, device=self.device)
        self.fpost = wp.zeros((FQ, self.nx, self.ny, 1), dtype=wp.float64, device=self.device)
        self.u = wp.zeros((2, self.nx, self.ny, 1), dtype=wp.float64, device=self.device)
        self.u_star = wp.zeros((2, self.nx, self.ny, 1), dtype=wp.float64, device=self.device)
        self.u_local_next = wp.zeros((2, self.nx, self.ny, 1), dtype=wp.float64, device=self.device)
        self.local_bc_diag = wp.zeros(self.shape, dtype=wp.float64, device=self.device)
        self.local_bc_rhs = wp.zeros((2, self.nx, self.ny, 1), dtype=wp.float64, device=self.device)
        self.U = wp.zeros((NCOMP, self.nx, self.ny, 1), dtype=wp.float64, device=self.device)
        self.P = wp.zeros((4, self.nx, self.ny, 1), dtype=wp.float64, device=self.device)
        self.sigma = wp.zeros((4, self.nx, self.ny, 1), dtype=wp.float64, device=self.device)
        self.invalid = wp.zeros((1,), dtype=wp.int32, device=self.device)
        self.initialize_identity(init_order=1)

    def initialize_identity(self, *, init_order: int = 1) -> None:
        U = np.zeros((NCOMP, self.nx, self.ny), dtype=np.float64)
        U[2] = 1.0
        U[5] = 1.0
        self.initialize_from_numpy(
            U=U,
            displacement=np.zeros((2, self.nx, self.ny), dtype=np.float64),
            init_order=init_order,
        )

    def initialize_from_numpy(
        self,
        *,
        U: np.ndarray,
        displacement: np.ndarray | None = None,
        init_order: int = 1,
        apply_initial_boundaries: bool = True,
        initial_boundary_time: float = 0.0,
    ) -> None:
        U = np.asarray(U, dtype=np.float64)
        if U.shape != (NCOMP, self.nx, self.ny):
            raise ValueError(f"U must have shape {(NCOMP, self.nx, self.ny)}")
        displacement = (
            np.zeros((2, self.nx, self.ny), dtype=np.float64)
            if displacement is None
            else np.asarray(displacement, dtype=np.float64)
        )
        if displacement.shape != (2, self.nx, self.ny):
            raise ValueError(f"displacement must have shape {(2, self.nx, self.ny)}")

        if init_order == 1:
            f = _equilibrium_np(U, self.material, self.lattice_speed)
            source = _source_np(U, self.source_mode, self.source_params)
            f -= (0.5 * self.dt / float(Q)) * source[None, ...]
        elif init_order == 2:
            # Full-rectangle smooth initial correction, then mask inactive nodes.
            # This is appropriate for manufactured/affine tests where U is known
            # smoothly beyond the physical boundary.
            f = _second_order_populations_np(
                U,
                self.material,
                self.lattice_speed,
                dx=self.dx,
                dy=self.dy,
                dt=self.dt,
                periodic_x=False,
                periodic_y=False,
                source_mode=self.source_mode,
                source_params=self.source_params,
            )
        elif init_order == 3:
            # Active-mask-aware second-order correction.  Derivatives at curved
            # boundaries use only active neighbors, so this path does not need a
            # physically meaningful continuation of U outside the solid.
            f = _second_order_populations_masked_np(
                U,
                self.active_np.astype(bool),
                self.material,
                self.lattice_speed,
                dx=self.dx,
                dy=self.dy,
                dt=self.dt,
                periodic_x=False,
                periodic_y=False,
                source_mode=self.source_mode,
                source_params=self.source_params,
            )
        else:
            raise ValueError("init_order must be 1, 2, or 3")
        f[:, :, self.active_np == 0] = 0.0
        displacement_masked = np.array(displacement, dtype=np.float64, copy=True)
        displacement_masked[:, self.active_np == 0] = 0.0
        u_star0 = displacement_masked - 0.5 * self.dt * U[:2]
        u_star0[:, self.active_np == 0] = 0.0
        wp.copy(
            self.f0,
            _as_warp(
                f.reshape((FQ, self.nx, self.ny))[..., None], dtype=wp.float64, device=self.device
            ),
        )
        wp.copy(
            self.f1,
            _as_warp(
                np.zeros((FQ, self.nx, self.ny, 1), dtype=np.float64),
                dtype=wp.float64,
                device=self.device,
            ),
        )
        wp.copy(
            self.fpost,
            _as_warp(
                np.zeros((FQ, self.nx, self.ny, 1), dtype=np.float64),
                dtype=wp.float64,
                device=self.device,
            ),
        )
        wp.copy(self.u_star, _as_warp(u_star0[..., None], dtype=wp.float64, device=self.device))
        self.time = 0.0
        self.steps = 0
        self.refresh_current()
        if apply_initial_boundaries:
            self.apply_boundary_reconstruction(float(initial_boundary_time), in_place=True)
            if self.kinetic_filter_strength > 0.0:
                wp.launch(
                    _kinetic_ghost_filter_kernel,
                    dim=self.shape,
                    inputs=[self.f0, self.active, self.kinetic_filter_strength],
                    device=self.device,
                )
            self.refresh_current()

    def _boundary_launch_inputs(
        self, boundary_time: float, fpost_arr: wp.array, ftarget_arr: wp.array
    ) -> list:
        return [
            fpost_arr,
            ftarget_arr,
            self.U,
            self.u,
            self.active,
            self.link_i,
            self.link_j,
            self.link_q,
            self.link_boundary_id,
            self.link_eta,
            self.link_xb,
            self.link_yb,
            self.link_nx,
            self.link_ny,
            self.lsq_i,
            self.lsq_j,
            self.lsq_wb,
            self.lsq_wt,
            self.lsq_wx,
            self.lsq_wy,
            self.lsq_valid,
            self.geometry.n_links,
            self.nx,
            self.ny,
            self.boundary_kind_by_id,
            self.boundary_mode_by_id,
            self.boundary_params_by_id,
            self.link_traction_x,
            self.link_traction_y,
            int(self.use_dynamic_link_traction),
            float(boundary_time),
            self.dt,
            self.collision_omega,
            self.lattice_speed,
            self.material.id,
            self.material.lam,
            self.material.mu,
            self.material.param0,
            self.material.param1,
            self.material.param2,
            self.jacobian_floor,
            self.source_mode,
            self.source_params[0],
            self.source_params[1],
            self.source_params[2],
            self.source_params[3],
            self.source_params[4],
            self.source_params[5],
            self.boundary_reconstruction,
            self._normal_solver_mode,
        ]

    def enable_dynamic_link_traction(self, enabled: bool = True) -> None:
        """Use device-resident per-cut-link traction on Neumann links.

        This keeps the ``local_f`` boundary closure fully local: link ``ell``
        consumes only the traction assigned to that same link.
        """
        self.use_dynamic_link_traction = bool(enabled)

    def apply_boundary_reconstruction(
        self, boundary_time: float, *, in_place: bool = False
    ) -> None:
        if self.geometry.n_links == 0:
            return
        if in_place:
            # Opposing cut links must read the same initial population state.
            wp.copy(self.fpost, self.f0)
        source = self.fpost
        target = self.f0 if in_place else self.f1
        wp.launch(
            _apply_curved_boundary_links_kernel,
            dim=self.geometry.n_links,
            inputs=self._boundary_launch_inputs(boundary_time, source, target),
            device=self.device,
        )

    def configure_compatibility_projection(
        self,
        *,
        interval: int = 1,
        iterations: int = 160,
        relax: float = 0.85,
        weight: float = 20.0,
        repair_F: bool = False,
        repair_F_band_width: int = -1,
        repair_F_blend: float = 1.0,
        boundary_id: int = BOUNDARY_ID_INNER,
    ) -> None:
        if int(interval) != 0 or bool(repair_F):
            raise ValueError(
                "global compatibility projection is not available in the fully-local solver"
            )

    def project_compatible_displacement(
        self, *, iterations: int | None = None, repair_F: bool | None = None
    ) -> None:
        raise RuntimeError("global compatibility projection is disabled by design")

    def _compatibility_projection_due(self) -> bool:
        return False

    def _local_compatibility_due(self) -> bool:
        return (
            self.local_compatibility_interval > 0
            and self.steps % self.local_compatibility_interval == 0
        )

    def _local_displacement_due(self) -> bool:
        return (
            self.local_displacement_interval > 0
            and self.steps % self.local_displacement_interval == 0
        )

    def _apply_local_displacement_relaxation(self, boundary_time: float) -> None:
        wp.launch(
            _local_displacement_bc_clear_kernel,
            dim=self.shape,
            inputs=[self.local_bc_diag, self.local_bc_rhs],
            device=self.device,
        )
        if self.geometry.n_links > 0:
            wp.launch(
                _local_displacement_bc_kernel,
                dim=self.geometry.n_links,
                inputs=[
                    self.U,
                    self.active,
                    self.link_i,
                    self.link_j,
                    self.link_boundary_id,
                    self.link_xb,
                    self.link_yb,
                    self.link_nx,
                    self.link_ny,
                    self.boundary_kind_by_id,
                    self.boundary_mode_by_id,
                    self.boundary_params_by_id,
                    self.geometry.n_links,
                    self.local_displacement_boundary_id,
                    boundary_time,
                    self.dx,
                    self.dy,
                    self.local_displacement_boundary_weight,
                    self.local_bc_diag,
                    self.local_bc_rhs,
                ],
                device=self.device,
            )
        wp.launch(
            _local_displacement_relax_kernel,
            dim=self.shape,
            inputs=[
                self.U,
                self.active,
                self.u,
                self.u_local_next,
                self.local_bc_diag,
                self.local_bc_rhs,
                self.nx,
                self.ny,
                self.dx,
                self.dy,
                self.local_displacement_relax,
            ],
            device=self.device,
        )
        wp.launch(
            _local_displacement_commit_kernel,
            dim=self.shape,
            inputs=[self.active, self.u_local_next, self.u, self.u_star],
            device=self.device,
        )

    def _apply_local_compatibility(self) -> None:
        wp.launch(
            _local_compatibility_repair_kernel,
            dim=self.shape,
            inputs=[
                self.fpost,
                self.U,
                self.P,
                self.sigma,
                self.active,
                self.u,
                self.nx,
                self.ny,
                self.dx,
                self.dy,
                self.local_compatibility_blend,
                int(self.local_compatibility_interior_only),
                self.lattice_speed,
                self.material.id,
                self.material.lam,
                self.material.mu,
                self.material.param0,
                self.material.param1,
                self.material.param2,
                self.jacobian_floor,
            ],
            device=self.device,
        )

    def step(self) -> None:
        wp.launch(_reset_invalid_kernel, dim=1, inputs=[self.invalid], device=self.device)
        if self.collision_model == "mrt_ghost":
            wp.launch(
                _collide_masked_mrt_ghost_kernel,
                dim=self.shape,
                inputs=[
                    self.f0,
                    self.fpost,
                    self.u,
                    self.u_star,
                    self.U,
                    self.P,
                    self.sigma,
                    self.active,
                    self.invalid,
                    self.dt,
                    self.collision_omega,
                    self.ghost_omega,
                    self.lattice_speed,
                    self.material.id,
                    self.material.lam,
                    self.material.mu,
                    self.material.param0,
                    self.material.param1,
                    self.material.param2,
                    self.jacobian_floor,
                    self.source_mode,
                    self.source_params[0],
                    self.source_params[1],
                    self.source_params[2],
                    self.source_params[3],
                    self.source_params[4],
                    self.source_params[5],
                ],
                device=self.device,
            )
        else:
            wp.launch(
                _collide_masked_kernel,
                dim=self.shape,
                inputs=[
                    self.f0,
                    self.fpost,
                    self.u,
                    self.u_star,
                    self.U,
                    self.P,
                    self.sigma,
                    self.active,
                    self.invalid,
                    self.dt,
                    self.collision_omega,
                    self.lattice_speed,
                    self.material.id,
                    self.material.lam,
                    self.material.mu,
                    self.material.param0,
                    self.material.param1,
                    self.material.param2,
                    self.jacobian_floor,
                    self.source_mode,
                    self.source_params[0],
                    self.source_params[1],
                    self.source_params[2],
                    self.source_params[3],
                    self.source_params[4],
                    self.source_params[5],
                ],
                device=self.device,
            )
        if self._local_displacement_due():
            for _ in range(self.local_displacement_sweeps):
                self._apply_local_displacement_relaxation(self.time + 0.5 * self.dt)
        if self._local_compatibility_due():
            self._apply_local_compatibility()
        wp.launch(
            _clear_kernel, dim=(FQ, self.nx, self.ny, 1), inputs=[self.f1], device=self.device
        )
        wp.launch(
            _stream_masked_kernel,
            dim=self.shape,
            inputs=[self.fpost, self.f1, self.active, self.nx, self.ny],
            device=self.device,
        )
        self.apply_boundary_reconstruction(self.time + 0.5 * self.dt)
        if self.kinetic_filter_strength > 0.0:
            wp.launch(
                _kinetic_ghost_filter_kernel,
                dim=self.shape,
                inputs=[self.f1, self.active, self.kinetic_filter_strength],
                device=self.device,
            )
        self.f0, self.f1 = self.f1, self.f0
        self.time += self.dt
        self.steps += 1

    def refresh_current(self) -> None:
        wp.launch(_reset_invalid_kernel, dim=1, inputs=[self.invalid], device=self.device)
        wp.launch(
            _refresh_masked_kernel,
            dim=self.shape,
            inputs=[
                self.f0,
                self.u,
                self.u_star,
                self.U,
                self.P,
                self.sigma,
                self.active,
                self.invalid,
                self.dt,
                self.material.id,
                self.material.lam,
                self.material.mu,
                self.material.param0,
                self.material.param1,
                self.material.param2,
                self.jacobian_floor,
                self.source_mode,
                self.source_params[0],
                self.source_params[1],
                self.source_params[2],
                self.source_params[3],
                self.source_params[4],
                self.source_params[5],
            ],
            device=self.device,
        )

    def run_until(
        self, t_end: float, *, stop_on_invalid: bool = True, check_interval: int = 0
    ) -> None:
        while self.time < float(t_end) - 0.5 * self.dt:
            self.step()
            if (
                stop_on_invalid
                and int(check_interval) > 0
                and self.steps % int(check_interval) == 0
            ):
                if self.invalid_state():
                    break
        self.refresh_current()
        wp.synchronize_device(self.device)

    def invalid_state(self) -> bool:
        wp.synchronize_device(self.device)
        if int(np.asarray(self.invalid.numpy())[0]) != 0:
            return True
        U = self.numpy_field("U")
        mask = self.active_np.astype(bool)
        return not np.isfinite(U[:, mask]).all()

    def numpy_field(
        self, name: Literal["u", "u_star", "U", "P", "sigma", "f0", "fpost"]
    ) -> np.ndarray:
        field = getattr(self, name)
        wp.synchronize_device(self.device)
        arr = np.asarray(field.numpy())
        return arr[..., 0]

    def displacement(self) -> np.ndarray:
        return self.numpy_field("u")

    def system_state(self) -> np.ndarray:
        return self.numpy_field("U")

    def deformation_gradient(self) -> np.ndarray:
        U = self.system_state()
        F = np.empty((2, 2, self.nx, self.ny), dtype=np.float64)
        F[0, 0], F[0, 1], F[1, 0], F[1, 1] = U[2], U[3], U[4], U[5]
        return F

    def deformation_jacobian(self) -> np.ndarray:
        F = self.deformation_gradient()
        return F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]

    def active_mask(self) -> np.ndarray:
        return self.active_np.astype(bool)

    def summary(self) -> dict[str, float | int | str | bool]:
        self.refresh_current()
        wp.synchronize_device(self.device)
        u = self.displacement()
        U = self.system_state()
        J = self.deformation_jacobian()
        mask = self.active_mask()
        displacement_radius = (
            self.local_displacement_sweeps if self.local_displacement_interval > 0 else 0
        )
        composed_radius = (
            displacement_radius + 2
            if self.local_compatibility_interval > 0
            else displacement_radius
        )
        return {
            "device": self.device,
            "geometry": self.geometry.label,
            "boundary_label": self.boundary.label,
            "boundary_kind": self.boundary.kind,
            "boundary_mode": int(self.boundary.value_mode),
            "model": self.material.model,
            "lambda": float(self.material.lam),
            "mu": float(self.material.mu),
            "nx": self.nx,
            "ny": self.ny,
            "n_active": int(np.count_nonzero(mask)),
            "n_cut_links": self.geometry.n_links,
            "steps": self.steps,
            "time": float(self.time),
            "dx": float(self.dx),
            "dt": float(self.dt),
            "lattice_speed": float(self.lattice_speed),
            "collision_omega": float(self.collision_omega),
            "collision_model": self.collision_model,
            "ghost_omega": float(self.ghost_omega),
            "kinetic_filter_strength": float(self.kinetic_filter_strength),
            "boundary_reconstruction": self.boundary_reconstruction_name,
            "boundary_stencil_radius": self.boundary_stencil_radius,
            "normal_jacobian": self.normal_jacobian_name,
            "neumann_solver": self.neumann_solver_name,
            "local_compatibility_interval": self.local_compatibility_interval,
            "local_compatibility_blend": self.local_compatibility_blend,
            "local_compatibility_interior_only": self.local_compatibility_interior_only,
            "local_displacement_interval": self.local_displacement_interval,
            "local_displacement_sweeps": self.local_displacement_sweeps,
            "local_displacement_relax": self.local_displacement_relax,
            "local_displacement_boundary_weight": self.local_displacement_boundary_weight,
            "local_displacement_boundary_id": self.local_displacement_boundary_id,
            "local_kernel_stencil_radius": 2
            if self.local_compatibility_interval > 0
            else (1 if displacement_radius > 0 else 0),
            "local_displacement_dependency_radius": displacement_radius,
            "local_composed_compatibility_radius": composed_radius,
            "linearized_stability_ratio": float(
                self.material.linearized_stability_ratio(self.lattice_speed)
            ),
            "max_abs_u": float(np.max(np.sqrt(np.sum(u[:, mask] * u[:, mask], axis=0))))
            if np.any(mask)
            else 0.0,
            "max_abs_v": float(np.max(np.sqrt(U[0, mask] * U[0, mask] + U[1, mask] * U[1, mask])))
            if np.any(mask)
            else 0.0,
            "min_J": float(np.min(J[mask])) if np.any(mask) else 0.0,
            "max_J": float(np.max(J[mask])) if np.any(mask) else 0.0,
            "finite": bool(
                np.isfinite(u[:, mask]).all()
                and np.isfinite(U[:, mask]).all()
                and np.all(J[mask] > 0.0)
            )
            if np.any(mask)
            else False,
        }

    def save_npz(self, path: str | Path) -> None:
        self.refresh_current()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            x=self.x,
            y=self.y,
            active=self.active_np,
            u=self.displacement(),
            U=self.system_state(),
            F=self.deformation_gradient(),
            P=self.numpy_field("P"),
            sigma=self.numpy_field("sigma"),
            f=self.numpy_field("f0"),
            link_i=self.geometry.link_i,
            link_j=self.geometry.link_j,
            link_q=self.geometry.link_q,
            link_boundary_id=self.geometry.link_boundary_id,
            link_eta=self.geometry.link_eta,
            link_xb=self.geometry.link_xb,
            link_yb=self.geometry.link_yb,
            link_nx=self.geometry.link_nx,
            link_ny=self.geometry.link_ny,
            lsq_valid=self.lsq_valid_np,
            time=np.array(self.time),
            dx=np.array(self.dx),
            dt=np.array(self.dt),
            lattice_speed=np.array(self.lattice_speed),
            collision_omega=np.array(self.collision_omega),
            collision_model=np.array(self.collision_model),
            ghost_omega=np.array(self.ghost_omega),
            kinetic_filter_strength=np.array(self.kinetic_filter_strength),
            boundary_reconstruction=np.array(self.boundary_reconstruction_name),
            normal_jacobian=np.array(self.normal_jacobian_name),
            neumann_solver=np.array(self.neumann_solver_name),
            local_compatibility_interval=np.array(self.local_compatibility_interval),
            local_compatibility_blend=np.array(self.local_compatibility_blend),
            local_compatibility_interior_only=np.array(self.local_compatibility_interior_only),
            local_displacement_interval=np.array(self.local_displacement_interval),
            local_displacement_sweeps=np.array(self.local_displacement_sweeps),
            local_displacement_relax=np.array(self.local_displacement_relax),
            local_displacement_boundary_weight=np.array(self.local_displacement_boundary_weight),
            local_displacement_boundary_id=np.array(self.local_displacement_boundary_id),
            boundary_kind=np.array(self.boundary.kind),
            boundary_mode=np.array(int(self.boundary.value_mode)),
            boundary_params=self.boundary_params,
            boundary_kind_by_id=self.boundary_kind_by_id_np,
            boundary_mode_by_id=self.boundary_mode_by_id_np,
            boundary_params_by_id=self.boundary_params_by_id_np,
            model=np.array(self.material.model),
            lam=np.array(self.material.lam),
            mu=np.array(self.material.mu),
            material_params=np.asarray(self.material.params, dtype=np.float64),
            metadata=np.array(json.dumps(self.summary())),
        )


# Small NumPy utilities used by examples and offline diagnostics.
def relative_l2(error: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    err = np.asarray(error)
    ref = np.asarray(reference)
    mask_bool = np.asarray(mask, dtype=bool)
    num = float(np.sqrt(np.sum(err[..., mask_bool] ** 2)))
    den = float(np.sqrt(np.sum(ref[..., mask_bool] ** 2)))
    return num / max(den, np.finfo(float).eps)


def constant_piola_params(
    material: WarpHyperelasticMaterial, F: np.ndarray
) -> tuple[float, float, float, float]:
    """Return params for ``CURVED_VALUE_PIOLA_NORMAL`` from a constant F."""
    F4 = np.asarray(F, dtype=np.float64).reshape(2, 2)
    P = _first_piola_np(F4, material)
    return float(P[0, 0]), float(P[0, 1]), float(P[1, 0]), float(P[1, 1])
