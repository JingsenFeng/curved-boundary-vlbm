"""Three-dimensional curved-boundary D3Q6x12 hyperelastic solver.

The reference material occupies phi(X) <= 0. Active nodes collide and
stream on a Cartesian lattice. Velocity Dirichlet cut links use BFL
interpolation; local_f traction links use local tangential prediction,
nonlinear nominal-traction inversion, and a characteristic reconstruction
with the BGK relaxation and source coefficients. Local displacement and
deformation-gradient updates couple the two kinematic fields.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal
import math
import os
import json
import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})
from .characteristic_stencil3d import build_characteristic_weights, MAX_CHARACTERISTIC_SAMPLES

from . import vector_nonlinear_elastic_warp3d as _base

wp.init()

Q = int(_base.Q)
NCOMP = int(_base.NCOMP)
FQ = int(_base.FQ)
BC_DIRICHLET = int(_base.BC_DIRICHLET)
BC_NEUMANN = int(_base.BC_NEUMANN)
VALUE_ZERO = int(_base.VALUE_ZERO)
VALUE_CONSTANT = int(_base.VALUE_CONSTANT)
VALUE_AFFINE_VELOCITY = int(_base.VALUE_AFFINE_VELOCITY)
VALUE_SINE2_HOLD = int(getattr(_base, "VALUE_SINE2_HOLD", 5))
SOURCE_CONSTANT = int(_base.SOURCE_CONSTANT)
SOURCE_DAMPING = int(_base.SOURCE_DAMPING)
WarpHyperelasticMaterial3D = _base.WarpHyperelasticMaterial3D
MaterialName = _base.MaterialName
SourceName = _base.SourceName
default_device = _base.default_device
source_id = _base.source_id
source_name = _base.source_name
_as_warp = _base._as_warp
_equilibrium_np = _base._equilibrium_np
_source_np = _base._source_np
_second_order_populations_np = _base._second_order_populations_np
_first_piola_np = _base._first_piola_np
_state_to_F_np = _base._state_to_F_np
_det3_np = _base._det3_np
_cofactor_np = _base._cofactor_np
_piola_mat = _base._piola_mat
_det3_wp = _base._det3
_equilibrium_comp = _base._equilibrium_comp
_source_comp = _base._source_comp
_flatten_pop = _base._flatten_pop
_unflatten_pop = _base._unflatten_pop
_collide_kernel = _base._collide_kernel
_refresh_kernel = _base._refresh_kernel
_reset_invalid_kernel = _base._reset_invalid_kernel
_clear_kernel = _base._clear_kernel
State12 = wp.types.vector(length=12, dtype=wp.float64)

PI = wp.constant(wp.float64(3.141592653589793238462643383279502884))
DIRS3 = np.array(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.int32
)

CURVED3D_VALUE_ZERO = VALUE_ZERO
CURVED3D_VALUE_CONSTANT_VECTOR = VALUE_CONSTANT
CURVED3D_VALUE_AFFINE_VELOCITY = VALUE_AFFINE_VELOCITY
CURVED3D_VALUE_RAMP_VECTOR = 200
CURVED3D_VALUE_SINE2_HOLD_VECTOR = 204
CURVED3D_VALUE_PIOLA_NORMAL = 201
CURVED3D_VALUE_NORMAL_TRACTION = 202
CURVED3D_VALUE_SINE2_NORMAL_TRACTION = 203
CURVED3D_VALUE_AFFINE_NEO_HOOKE_PIOLA_NORMAL = 205
CURVED3D_VALUE_Z_PULL_TORSION_TRACTION = 206
CURVED3D_VALUE_Z_STRETCH_EXACT_TWIST_VELOCITY = 207
# Short aliases accepted by the derivation and examples.
CURVED_VALUE_ZERO = CURVED3D_VALUE_ZERO
CURVED_VALUE_CONSTANT_VECTOR = CURVED3D_VALUE_CONSTANT_VECTOR
CURVED_VALUE_AFFINE_VELOCITY = CURVED3D_VALUE_AFFINE_VELOCITY
CURVED_VALUE_RAMP_VECTOR = CURVED3D_VALUE_RAMP_VECTOR
CURVED_VALUE_SINE2_HOLD_VECTOR = CURVED3D_VALUE_SINE2_HOLD_VECTOR
CURVED_VALUE_PIOLA_NORMAL = CURVED3D_VALUE_PIOLA_NORMAL
CURVED_VALUE_NORMAL_TRACTION = CURVED3D_VALUE_NORMAL_TRACTION
CURVED_VALUE_SINE2_NORMAL_TRACTION = CURVED3D_VALUE_SINE2_NORMAL_TRACTION
CURVED_VALUE_AFFINE_NEO_HOOKE_PIOLA_NORMAL = CURVED3D_VALUE_AFFINE_NEO_HOOKE_PIOLA_NORMAL
CURVED_VALUE_Z_PULL_TORSION_TRACTION = CURVED3D_VALUE_Z_PULL_TORSION_TRACTION
CURVED_VALUE_Z_STRETCH_EXACT_TWIST_VELOCITY = CURVED3D_VALUE_Z_STRETCH_EXACT_TWIST_VELOCITY
MATERIAL_NEO_HOOKE = int(_base.MATERIAL_NEO_HOOKE)
BOUNDARY_RECON_BFL = 0
BOUNDARY_RECON_HALF_WAY = 1
BOUNDARY_RECON_COMPAT_BFL = 2
BOUNDARY_RECON_LOCAL_F = 3
BOUNDARY_RECON_LOCAL_F_BFL = 4
NORMAL_JACOBIAN_FINITE_DIFFERENCE = 0
NORMAL_JACOBIAN_ACOUSTIC = 1
# Internal boundary-solver dispatch; independent of the public Newton Jacobian.
NORMAL_SOLVER_ENERGY = 2
MAX_LSQ_STENCIL = 32
MAX_BOUNDARY_IDS = 16
BOUNDARY_ID_OUTER = 1
BOUNDARY_ID_INNER = 2
BoundaryKind = Literal["dirichlet", "neumann"]
ComponentBoundaryKinds = tuple[BoundaryKind, BoundaryKind, BoundaryKind]
BoundaryReconstruction = Literal["bfl", "halfway", "compat_bfl", "local_f", "local_f_bfl"]


def neumann_solver_name(name: str | None, material: WarpHyperelasticMaterial3D) -> str:
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


def _normal_jacobian_for_material(material: WarpHyperelasticMaterial3D, mode: str | None) -> int:
    # The closed-form acoustic tensor below is specific to the historical
    # quadratic-volumetric Neo-Hookean law.  The section-5.3 SPH energy has a
    # different volumetric response and therefore defaults to the general
    # finite-difference Jacobian unless the caller explicitly chooses a mode.
    if mode is None and material.model == "sph_neo_hooke":
        return NORMAL_JACOBIAN_FINITE_DIFFERENCE
    return normal_jacobian_id(mode)


def _kind_id(k: str) -> int:
    if k == "dirichlet":
        return BC_DIRICHLET
    if k == "neumann":
        return BC_NEUMANN
    raise ValueError(k)


def _recon_id(k: str) -> int:
    if k in ("bfl", "bouzidi", "bouzidi_firdaouss_lallemand"):
        return BOUNDARY_RECON_BFL
    if k in ("halfway", "half_way", "half-way"):
        return BOUNDARY_RECON_HALF_WAY
    if k == "compat_bfl":
        return BOUNDARY_RECON_COMPAT_BFL
    if k in ("local_f", "fully_local"):
        return BOUNDARY_RECON_LOCAL_F
    if k in ("local_f_bfl", "single_history"):
        return BOUNDARY_RECON_LOCAL_F_BFL
    raise ValueError(f"unsupported boundary_reconstruction: {k}")


def _recon_name(rid: int) -> str:
    if rid == BOUNDARY_RECON_BFL:
        return "bfl"
    if rid == BOUNDARY_RECON_HALF_WAY:
        return "halfway"
    if rid == BOUNDARY_RECON_COMPAT_BFL:
        return "compat_bfl"
    if rid == BOUNDARY_RECON_LOCAL_F:
        return "local_f"
    if rid == BOUNDARY_RECON_LOCAL_F_BFL:
        return "local_f_bfl"
    return f"unknown({rid})"


@dataclass(frozen=True)
class CurvedBoundarySpec3D:
    kind: BoundaryKind = "dirichlet"
    value_mode: int = CURVED3D_VALUE_ZERO
    params: tuple[float, ...] = ()
    id_kinds: dict[int, BoundaryKind] | None = None
    id_value_modes: dict[int, int] | None = None
    id_params: dict[int, tuple[float, ...]] | None = None
    label: str = ""
    component_kinds: ComponentBoundaryKinds | None = None
    id_component_kinds: dict[int, ComponentBoundaryKinds] | None = None

    def tables(self, max_ids: int = MAX_BOUNDARY_IDS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kind = np.full(max_ids, _kind_id(self.kind), dtype=np.int32)
        mode = np.full(max_ids, int(self.value_mode), dtype=np.int32)
        p0 = np.zeros(16, dtype=np.float64)
        for i, v in enumerate(tuple(self.params)[:16]):
            p0[i] = float(v)
        params = np.tile(p0[None, :], (max_ids, 1))
        if self.id_kinds:
            for i, v in self.id_kinds.items():
                if 0 <= int(i) < max_ids:
                    kind[int(i)] = _kind_id(v)
        if self.id_value_modes:
            for i, v in self.id_value_modes.items():
                if 0 <= int(i) < max_ids:
                    mode[int(i)] = int(v)
        if self.id_params:
            for i, vals in self.id_params.items():
                if 0 <= int(i) < max_ids:
                    params[int(i), :] = 0.0
                    for j, v in enumerate(tuple(vals)[:16]):
                        params[int(i), j] = float(v)
        return kind, mode, np.ascontiguousarray(params)

    def component_kind_table(self, max_ids: int = MAX_BOUNDARY_IDS) -> np.ndarray:
        """Return per-displacement-row boundary kinds with legacy broadcast.

        Existing ``kind``/``id_kinds`` callers retain exactly their historical
        all-component behavior.  ``component_kinds`` and
        ``id_component_kinds`` are opt-in overrides ordered as ``(x,y,z)``.
        """
        kind, _, _ = self.tables(max_ids)
        out = np.repeat(kind[None, :], 3, axis=0)
        if self.component_kinds is not None:
            values = tuple(self.component_kinds)
            if len(values) != 3:
                raise ValueError("component_kinds must contain exactly three kinds")
            for component, value in enumerate(values):
                out[component, :] = _kind_id(value)
            if self.id_kinds:
                for i, value in self.id_kinds.items():
                    if 0 <= int(i) < max_ids:
                        out[:, int(i)] = _kind_id(value)
        if self.id_component_kinds:
            for i, values in self.id_component_kinds.items():
                if not 0 <= int(i) < max_ids:
                    continue
                component_values = tuple(values)
                if len(component_values) != 3:
                    raise ValueError("id_component_kinds values must contain exactly three kinds")
                for component, value in enumerate(component_values):
                    out[component, int(i)] = _kind_id(value)
        return np.ascontiguousarray(out)


@dataclass(frozen=True)
class SphereDomain:
    center: tuple[float, float, float] = (0.5, 0.5, 0.5)
    radius: float = 0.35

    def phi(self, x: float, y: float, z: float) -> float:
        cx, cy, cz = self.center
        return math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - self.radius

    def normal(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        cx, cy, cz = self.center
        rx, ry, rz = x - cx, y - cy, z - cz
        r = max(math.sqrt(rx * rx + ry * ry + rz * rz), 1e-30)
        return rx / r, ry / r, rz / r


@dataclass
class CurvedBoundaryGeometry3D:
    nx: int
    ny: int
    nz: int
    length_x: float
    length_y: float
    length_z: float
    active: np.ndarray
    link_i: np.ndarray
    link_j: np.ndarray
    link_k: np.ndarray
    link_q: np.ndarray
    link_boundary_id: np.ndarray
    link_eta: np.ndarray
    link_xb: np.ndarray
    link_yb: np.ndarray
    link_zb: np.ndarray
    link_nx: np.ndarray
    link_ny: np.ndarray
    link_nz: np.ndarray
    label: str = "implicit"

    @property
    def dx(self):
        return self.length_x / float(self.nx)

    @property
    def dy(self):
        return self.length_y / float(self.ny)

    @property
    def dz(self):
        return self.length_z / float(self.nz)

    @property
    def n_active(self):
        return int(np.count_nonzero(self.active))

    @property
    def n_links(self):
        return int(self.link_i.size)

    def as_metadata(self):
        return {
            "label": self.label,
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "n_active": self.n_active,
            "n_cut_links": self.n_links,
        }

    @classmethod
    def from_level_set(
        cls,
        *,
        nx: int,
        ny: int,
        nz: int,
        length_x: float,
        length_y: float,
        length_z: float,
        phi: Callable[[float, float, float], float],
        normal: Callable[[float, float, float], tuple[float, float, float]] | None = None,
        boundary_id: Callable[[float, float, float, float, float, float], int] | None = None,
        label: str = "level_set",
        root_iterations: int = 42,
        phi_vectorized: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
        normal_vectorized: Callable[
            [np.ndarray, np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray,
        ]
        | None = None,
        phi_chunk_size: int = 64,
    ):
        dx = length_x / nx
        dy = length_y / ny
        dz = length_z / nz
        if nx < 4 or ny < 4 or nz < 4:
            raise ValueError("nx,ny,nz>=4 required")
        if max(abs(dx - dy), abs(dx - dz)) > 1e-14:
            raise ValueError("D3Q6 requires dx=dy=dz")
        xs = (np.arange(nx) + 0.5) * dx
        ys = (np.arange(ny) + 0.5) * dy
        zs = (np.arange(nz) + 0.5) * dz
        ph = np.empty((nx, ny, nz), float)
        if phi_vectorized is None:
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    for k, z in enumerate(zs):
                        ph[i, j, k] = float(phi(float(x), float(y), float(z)))
        else:
            X2, Y2 = np.meshgrid(xs, ys, indexing="ij")
            chunk = max(1, int(phi_chunk_size))
            for k0 in range(0, nz, chunk):
                k1 = min(nz, k0 + chunk)
                vals = np.asarray(
                    phi_vectorized(X2[:, :, None], Y2[:, :, None], zs[None, None, k0:k1]),
                    dtype=float,
                )
                if vals.shape != ph[:, :, k0:k1].shape:
                    vals = np.broadcast_to(vals, ph[:, :, k0:k1].shape)
                ph[:, :, k0:k1] = vals
        active = ph <= 0.0
        if not np.any(active):
            raise ValueError("no active material nodes")

        def phiv(x, y, z):
            if phi_vectorized is None:
                out = np.empty(np.broadcast(x, y, z).shape, dtype=float)
                it = np.nditer(
                    [
                        np.broadcast_to(x, out.shape),
                        np.broadcast_to(y, out.shape),
                        np.broadcast_to(z, out.shape),
                        out,
                    ],
                    op_flags=[["readonly"], ["readonly"], ["readonly"], ["writeonly"]],
                )
                for xx, yy, zz, oo in it:
                    oo[...] = float(phi(float(xx), float(yy), float(zz)))
                return out
            out = np.asarray(
                phi_vectorized(np.asarray(x), np.asarray(y), np.asarray(z)), dtype=float
            )
            shape = np.broadcast(x, y, z).shape
            if out.shape != shape:
                out = np.broadcast_to(out, shape)
            return out

        def normalv(x, y, z):
            shape = np.broadcast(x, y, z).shape
            if normal_vectorized is not None:
                vals = normal_vectorized(np.asarray(x), np.asarray(y), np.asarray(z))
                if isinstance(vals, (tuple, list)):
                    a, b, c = (
                        np.asarray(vals[0], dtype=float),
                        np.asarray(vals[1], dtype=float),
                        np.asarray(vals[2], dtype=float),
                    )
                else:
                    arr = np.asarray(vals, dtype=float)
                    if arr.shape[:1] == (3,):
                        a, b, c = arr[0], arr[1], arr[2]
                    elif arr.shape[-1:] == (3,):
                        a, b, c = arr[..., 0], arr[..., 1], arr[..., 2]
                    else:
                        raise ValueError(
                            "normal_vectorized must return a 3-tuple or an array with a 3-vector axis"
                        )
                a = np.broadcast_to(a, shape).astype(float, copy=False)
                b = np.broadcast_to(b, shape).astype(float, copy=False)
                c = np.broadcast_to(c, shape).astype(float, copy=False)
                r = np.maximum(np.sqrt(a * a + b * b + c * c), 1.0e-30)
                return a / r, b / r, c / r
            a = np.empty(shape, dtype=float)
            b = np.empty(shape, dtype=float)
            c = np.empty(shape, dtype=float)
            it = np.nditer(
                [
                    np.broadcast_to(x, shape),
                    np.broadcast_to(y, shape),
                    np.broadcast_to(z, shape),
                    a,
                    b,
                    c,
                ],
                op_flags=[
                    ["readonly"],
                    ["readonly"],
                    ["readonly"],
                    ["writeonly"],
                    ["writeonly"],
                    ["writeonly"],
                ],
            )
            for xx, yy, zz, aa, bb, cc in it:
                if normal is not None:
                    nxv, nyv, nzv = normal(float(xx), float(yy), float(zz))
                    nxv, nyv, nzv = float(nxv), float(nyv), float(nzv)
                else:
                    h = 0.5 * dx
                    nxv = (
                        float(phi(float(xx) + h, float(yy), float(zz)))
                        - float(phi(float(xx) - h, float(yy), float(zz)))
                    ) / (2 * h)
                    nyv = (
                        float(phi(float(xx), float(yy) + h, float(zz)))
                        - float(phi(float(xx), float(yy) - h, float(zz)))
                    ) / (2 * h)
                    nzv = (
                        float(phi(float(xx), float(yy), float(zz) + h))
                        - float(phi(float(xx), float(yy), float(zz) - h))
                    ) / (2 * h)
                rr = max(math.sqrt(nxv * nxv + nyv * nyv + nzv * nzv), 1.0e-30)
                aa[...] = nxv / rr
                bb[...] = nyv / rr
                cc[...] = nzv / rr
            return a, b, c

        if phi_vectorized is not None:
            lis = []
            ljs = []
            lks = []
            lqs = []
            lbs = []
            les = []
            lxbs = []
            lybs = []
            lzbs = []
            lnxs = []
            lnys = []
            lnzs = []
            sample = np.linspace(0.0, 1.0, 17)
            for q, (di, dj, dk) in enumerate(DIRS3):
                neighbor = np.zeros_like(active, dtype=bool)
                if di == 1:
                    neighbor[:-1, :, :] = active[1:, :, :]
                elif di == -1:
                    neighbor[1:, :, :] = active[:-1, :, :]
                elif dj == 1:
                    neighbor[:, :-1, :] = active[:, 1:, :]
                elif dj == -1:
                    neighbor[:, 1:, :] = active[:, :-1, :]
                elif dk == 1:
                    neighbor[:, :, :-1] = active[:, :, 1:]
                elif dk == -1:
                    neighbor[:, :, 1:] = active[:, :, :-1]
                cut = active & (~neighbor)
                ii, jj, kk = np.nonzero(cut)
                if ii.size == 0:
                    continue
                x0 = xs[ii]
                y0 = ys[jj]
                z0 = zs[kk]
                x1 = x0 + float(di) * dx
                y1 = y0 + float(dj) * dy
                z1 = z0 + float(dk) * dz
                p0 = ph[ii, jj, kk]
                p1 = phiv(x1, y1, z1)
                lo = np.zeros(ii.size, dtype=float)
                hi = np.ones(ii.size, dtype=float)
                needs = np.where(~((p0 <= 0.0) & (p1 >= 0.0)))[0]
                if needs.size:
                    found = np.zeros(needs.size, dtype=bool)
                    lo_n = np.full(needs.size, 0.5, dtype=float)
                    hi_n = np.full(needs.size, 0.5, dtype=float)
                    prev_s = sample[0]
                    prev_v = p0[needs]
                    for ss in sample[1:]:
                        pm = phiv(
                            x0[needs] + ss * (x1[needs] - x0[needs]),
                            y0[needs] + ss * (y1[needs] - y0[needs]),
                            z0[needs] + ss * (z1[needs] - z0[needs]),
                        )
                        hit = (~found) & (prev_v <= 0.0) & (pm >= 0.0)
                        lo_n[hit] = prev_s
                        hi_n[hit] = ss
                        found[hit] = True
                        prev_s = ss
                        prev_v = pm
                    lo[needs] = lo_n
                    hi[needs] = hi_n
                for _ in range(root_iterations):
                    mid = 0.5 * (lo + hi)
                    pm = phiv(x0 + mid * (x1 - x0), y0 + mid * (y1 - y0), z0 + mid * (z1 - z0))
                    inside = pm <= 0.0
                    lo = np.where(inside, mid, lo)
                    hi = np.where(inside, hi, mid)
                eta = np.clip(0.5 * (lo + hi), 1.0e-6, 1.0 - 1.0e-6)
                xb = x0 + eta * (x1 - x0)
                yb = y0 + eta * (y1 - y0)
                zb = z0 + eta * (z1 - z0)
                nxv, nyv, nzv = normalv(xb, yb, zb)
                if boundary_id is None:
                    bid = np.ones(ii.size, dtype=np.int32)
                else:
                    bid = np.empty(ii.size, dtype=np.int32)
                    for m in range(ii.size):
                        bid[m] = int(
                            boundary_id(
                                float(xb[m]),
                                float(yb[m]),
                                float(zb[m]),
                                float(nxv[m]),
                                float(nyv[m]),
                                float(nzv[m]),
                            )
                        )
                lis.append(ii.astype(np.int32))
                ljs.append(jj.astype(np.int32))
                lks.append(kk.astype(np.int32))
                lqs.append(np.full(ii.size, q, dtype=np.int32))
                lbs.append(np.clip(bid, 0, MAX_BOUNDARY_IDS - 1).astype(np.int32))
                les.append(eta.astype(float))
                lxbs.append(xb.astype(float))
                lybs.append(yb.astype(float))
                lzbs.append(zb.astype(float))
                lnxs.append(np.asarray(nxv, dtype=float))
                lnys.append(np.asarray(nyv, dtype=float))
                lnzs.append(np.asarray(nzv, dtype=float))

            def cat(parts, dtype):
                if parts:
                    return np.ascontiguousarray(np.concatenate(parts).astype(dtype, copy=False))
                return np.ascontiguousarray(np.empty(0, dtype=dtype))

            return cls(
                nx,
                ny,
                nz,
                float(length_x),
                float(length_y),
                float(length_z),
                np.ascontiguousarray(active.astype(np.int32)),
                cat(lis, np.int32),
                cat(ljs, np.int32),
                cat(lks, np.int32),
                cat(lqs, np.int32),
                cat(lbs, np.int32),
                cat(les, float),
                cat(lxbs, float),
                cat(lybs, float),
                cat(lzbs, float),
                cat(lnxs, float),
                cat(lnys, float),
                cat(lnzs, float),
                label,
            )
        li = []
        lj = []
        lk = []
        lq = []
        lb = []
        le = []
        lxb = []
        lyb = []
        lzb = []
        lnx = []
        lny = []
        lnz = []

        def phis(x, y, z):
            return float(phi(float(x), float(y), float(z)))

        def norms(x, y, z):
            if normal is not None:
                a, b, c = normal(float(x), float(y), float(z))
                a, b, c = float(a), float(b), float(c)
            else:
                h = 0.5 * dx
                a = (phis(x + h, y, z) - phis(x - h, y, z)) / (2 * h)
                b = (phis(x, y + h, z) - phis(x, y - h, z)) / (2 * h)
                c = (phis(x, y, z + h) - phis(x, y, z - h)) / (2 * h)
            r = max(math.sqrt(a * a + b * b + c * c), 1e-30)
            return a / r, b / r, c / r

        sample = np.linspace(0.0, 1.0, 17)
        for i, x0 in enumerate(xs):
            for j, y0 in enumerate(ys):
                for k, z0 in enumerate(zs):
                    if not active[i, j, k]:
                        continue
                    p0 = ph[i, j, k]
                    for q, (di, dj, dk) in enumerate(DIRS3):
                        ii, jj, kk = i + int(di), j + int(dj), k + int(dk)
                        if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz and active[ii, jj, kk]:
                            continue
                        x1 = x0 + di * dx
                        y1 = y0 + dj * dy
                        z1 = z0 + dk * dz
                        p1 = phis(x1, y1, z1)
                        if p0 <= 0 and p1 >= 0:
                            lo, hi = 0.0, 1.0
                        else:
                            vals = [
                                phis(x0 + s * (x1 - x0), y0 + s * (y1 - y0), z0 + s * (z1 - z0))
                                for s in sample
                            ]
                            idx = None
                            for m in range(len(sample) - 1):
                                if vals[m] <= 0 and vals[m + 1] >= 0:
                                    idx = m
                                    break
                            lo, hi = (
                                (0.5, 0.5)
                                if idx is None
                                else (float(sample[idx]), float(sample[idx + 1]))
                            )
                        for _ in range(root_iterations):
                            mid = 0.5 * (lo + hi)
                            pm = phis(
                                x0 + mid * (x1 - x0), y0 + mid * (y1 - y0), z0 + mid * (z1 - z0)
                            )
                            if pm <= 0:
                                lo = mid
                            else:
                                hi = mid
                        eta = float(np.clip(0.5 * (lo + hi), 1e-6, 1 - 1e-6))
                        xb = x0 + eta * (x1 - x0)
                        yb = y0 + eta * (y1 - y0)
                        zb = z0 + eta * (z1 - z0)
                        nxv, nyv, nzv = norms(xb, yb, zb)
                        bid = (
                            1
                            if boundary_id is None
                            else int(boundary_id(xb, yb, zb, nxv, nyv, nzv))
                        )
                        li.append(i)
                        lj.append(j)
                        lk.append(k)
                        lq.append(q)
                        lb.append(max(0, min(MAX_BOUNDARY_IDS - 1, bid)))
                        le.append(eta)
                        lxb.append(xb)
                        lyb.append(yb)
                        lzb.append(zb)
                        lnx.append(nxv)
                        lny.append(nyv)
                        lnz.append(nzv)
        return cls(
            nx,
            ny,
            nz,
            float(length_x),
            float(length_y),
            float(length_z),
            np.ascontiguousarray(active.astype(np.int32)),
            np.ascontiguousarray(np.array(li, np.int32)),
            np.ascontiguousarray(np.array(lj, np.int32)),
            np.ascontiguousarray(np.array(lk, np.int32)),
            np.ascontiguousarray(np.array(lq, np.int32)),
            np.ascontiguousarray(np.array(lb, np.int32)),
            np.ascontiguousarray(np.array(le, float)),
            np.ascontiguousarray(np.array(lxb, float)),
            np.ascontiguousarray(np.array(lyb, float)),
            np.ascontiguousarray(np.array(lzb, float)),
            np.ascontiguousarray(np.array(lnx, float)),
            np.ascontiguousarray(np.array(lny, float)),
            np.ascontiguousarray(np.array(lnz, float)),
            label,
        )

    @classmethod
    def sphere(
        cls,
        *,
        nx: int,
        ny: int,
        nz: int,
        length_x: float = 1.0,
        length_y: float = 1.0,
        length_z: float = 1.0,
        center: tuple[float, float, float] = (0.5, 0.5, 0.5),
        radius: float = 0.35,
    ):
        sph = SphereDomain(center, radius)
        return cls.from_level_set(
            nx=nx,
            ny=ny,
            nz=nz,
            length_x=length_x,
            length_y=length_y,
            length_z=length_z,
            phi=sph.phi,
            normal=sph.normal,
            label=f"sphere(center={center},r={radius})",
        )

    @classmethod
    def spherical_shell(
        cls,
        *,
        nx: int,
        ny: int,
        nz: int,
        length_x: float = 1.0,
        length_y: float = 1.0,
        length_z: float = 1.0,
        center: tuple[float, float, float] = (0.5, 0.5, 0.5),
        inner_radius: float = 0.18,
        outer_radius: float = 0.42,
    ):
        if not (0.0 < inner_radius < outer_radius):
            raise ValueError("spherical_shell requires 0 < inner_radius < outer_radius")
        cx, cy, cz = center

        def rr(x: float, y: float, z: float) -> float:
            return math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)

        def phi(x: float, y: float, z: float) -> float:
            r = rr(x, y, z)
            return max(r - outer_radius, inner_radius - r)

        def normal(x: float, y: float, z: float) -> tuple[float, float, float]:
            r = max(rr(x, y, z), 1.0e-30)
            ex, ey, ez = (x - cx) / r, (y - cy) / r, (z - cz) / r
            if abs(r - inner_radius) <= abs(r - outer_radius):
                return -ex, -ey, -ez
            return ex, ey, ez

        def bid(x: float, y: float, z: float, nxv: float, nyv: float, nzv: float) -> int:
            r = rr(x, y, z)
            return (
                BOUNDARY_ID_INNER
                if abs(r - inner_radius) <= abs(r - outer_radius)
                else BOUNDARY_ID_OUTER
            )

        return cls.from_level_set(
            nx=nx,
            ny=ny,
            nz=nz,
            length_x=length_x,
            length_y=length_y,
            length_z=length_z,
            phi=phi,
            normal=normal,
            boundary_id=bid,
            label=f"spherical_shell(center={center},ri={inner_radius},ro={outer_radius})",
        )


# ---- host Neumann solve ------------------------------------------------------
def _affine_neo_hooke_F_P_np(p: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    pp = np.zeros(16)
    pp[: min(16, len(p))] = p[: min(16, len(p))]
    tt = float(t) if pp[11] <= 0.0 else min(float(t), float(pp[11]))
    F = np.eye(3) + tt * pp[:9].reshape(3, 3)
    J = float(np.linalg.det(F))
    cof = _cofactor_np(F)
    H = cof / max(J, np.finfo(float).eps)
    lam = float(pp[9])
    mu = float(pp[10])
    P = mu * (F - H) + 0.5 * lam * (J * J - 1.0) * H
    return F, P


def _value_np(
    mode: int, p: np.ndarray, comp: int, x: float, y: float, z: float, n: np.ndarray, t: float
) -> float:
    pp = np.zeros(16)
    pp[: min(16, len(p))] = p[: min(16, len(p))]
    if mode == CURVED3D_VALUE_ZERO:
        return 0.0
    if mode == CURVED3D_VALUE_CONSTANT_VECTOR:
        return float(pp[comp])
    if mode == CURVED3D_VALUE_RAMP_VECTOR:
        scale = 1.0 if pp[3] <= 0 or t >= pp[3] else math.sin(math.pi * t / pp[3]) ** 2
        return float(pp[comp] * scale)
    if mode == CURVED3D_VALUE_AFFINE_VELOCITY:
        if pp[12] > 0 and t > pp[12]:
            return 0.0
        return float((pp[:9].reshape(3, 3) @ np.array([x, y, z]) + pp[9:12])[comp])
    if mode == CURVED3D_VALUE_SINE2_HOLD_VECTOR:
        active = float(pp[3])
        denom = float(pp[4]) if pp[4] > 0 else 2.0 * active
        scale = 1.0 if active <= 0 or t >= active else math.sin(math.pi * t / denom) ** 2
        return float(pp[comp] * scale)
    if mode == CURVED3D_VALUE_PIOLA_NORMAL:
        return float(pp[:9].reshape(3, 3)[comp, :] @ n)
    if mode == CURVED3D_VALUE_AFFINE_NEO_HOOKE_PIOLA_NORMAL:
        _, P = _affine_neo_hooke_F_P_np(pp, t)
        return float(P[comp, :] @ n)
    if mode == CURVED3D_VALUE_NORMAL_TRACTION:
        return float(pp[0] * n[comp])
    if mode == CURVED3D_VALUE_SINE2_NORMAL_TRACTION:
        amp = float(pp[0])
        active = float(pp[1])
        denom = float(pp[2]) if pp[2] > 0 else 2.0 * active
        scale = 1.0 if active <= 0 or t >= active else math.sin(math.pi * t / denom) ** 2
        return float(amp * scale * n[comp])
    if mode == CURVED3D_VALUE_Z_PULL_TORSION_TRACTION:
        active = float(pp[4])
        denom = float(pp[5]) if pp[5] > 0 else 2.0 * active
        scale = 1.0 if active <= 0 or t >= active else math.sin(math.pi * t / denom) ** 2
        dx = x - float(pp[0])
        dy = y - float(pp[1])
        axial = float(pp[2])
        alpha = float(pp[3])
        if comp == 0:
            return float(-scale * alpha * dy)
        if comp == 1:
            return float(scale * alpha * dx)
        return float(scale * axial)
    if mode == CURVED3D_VALUE_Z_STRETCH_EXACT_TWIST_VELOCITY:
        active = float(pp[4])
        if active <= 0.0 or t >= active:
            return 0.0
        theta_final = float(pp[3])
        axial_final = float(pp[2])
        phase = math.pi * t / active
        s = 0.5 * (1.0 - math.cos(phase))
        sdot = 0.5 * math.pi / active * math.sin(phase)
        theta = theta_final * s
        thetadot = theta_final * sdot
        dx = x - float(pp[0])
        dy = y - float(pp[1])
        if comp == 0:
            return float(thetadot * (-math.sin(theta) * dx - math.cos(theta) * dy))
        if comp == 1:
            return float(thetadot * (math.cos(theta) * dx - math.sin(theta) * dy))
        return float(axial_final * sdot)
    return 0.0


def _basis_np(n):
    n = np.asarray(n, float)
    n = n / max(np.linalg.norm(n), np.finfo(float).eps)
    ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.75 else np.array([0, 1.0, 0])
    t1 = np.cross(n, ref)
    t1 = t1 / max(np.linalg.norm(t1), np.finfo(float).eps)
    return t1, np.cross(n, t1)


def _solve_Fb_np(Fg, n, T, mat, j_floor=1e-12):
    n = np.asarray(n, float)
    n = n / max(np.linalg.norm(n), np.finfo(float).eps)
    t1, t2 = _basis_np(n)
    Fg = np.asarray(Fg, float).reshape(3, 3)
    h = Fg @ n
    g1 = Fg @ t1
    g2 = Fg @ t2

    def make(hv):
        return np.outer(hv, n) + np.outer(g1, t1) + np.outer(g2, t2)

    def res(hv):
        return _first_piola_np(make(hv), mat) @ n - T

    for _ in range(24):
        r = res(h)
        if np.linalg.norm(r) < 1e-11:
            break
        J = np.zeros((3, 3))
        eps = 1e-6
        for a in range(3):
            hp = h.copy()
            hp[a] += eps
            J[:, a] = (res(hp) - r) / eps
        try:
            d = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            break
        step = 1.0
        for _ls in range(16):
            hc = h + step * d
            if np.linalg.det(make(hc)) > j_floor:
                h = hc
                break
            step *= 0.5
    return make(h)


def constant_piola_params3d(
    material: WarpHyperelasticMaterial3D, F: np.ndarray
) -> tuple[float, ...]:
    return tuple(
        float(x) for x in _first_piola_np(np.asarray(F, float).reshape(3, 3), material).reshape(-1)
    )


# ---- light Warp kernels ------------------------------------------------------
@wp.func
def _dx(q: int) -> int:
    if q == 0:
        return 1
    if q == 3:
        return -1
    return 0


@wp.func
def _dy(q: int) -> int:
    if q == 1:
        return 1
    if q == 4:
        return -1
    return 0


@wp.func
def _dz(q: int) -> int:
    if q == 2:
        return 1
    if q == 5:
        return -1
    return 0


@wp.func
def _opp(q: int) -> int:
    if q < 3:
        return q + 3
    return q - 3


@wp.func
def _norm3(x: wp.float64, y: wp.float64, z: wp.float64) -> wp.float64:
    return wp.sqrt(x * x + y * y + z * z)


@wp.func
def _normalize3(v: wp.vec3d) -> wp.vec3d:
    r = _norm3(v[0], v[1], v[2])
    if r < wp.float64(1.0e-30):
        r = wp.float64(1.0e-30)
    return wp.vec3d(v[0] / r, v[1] / r, v[2] / r)


@wp.func
def _cross3(a: wp.vec3d, b: wp.vec3d) -> wp.vec3d:
    return wp.vec3d(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


@wp.func
def _curved_value_comp(
    mode: int,
    params: wp.array2d(dtype=wp.float64),
    bid: int,
    comp: int,
    x: wp.float64,
    y: wp.float64,
    z: wp.float64,
    nx: wp.float64,
    ny: wp.float64,
    nz: wp.float64,
    t: wp.float64,
    j_floor: wp.float64,
) -> wp.float64:
    if mode == CURVED3D_VALUE_ZERO:
        return wp.float64(0.0)
    if mode == CURVED3D_VALUE_CONSTANT_VECTOR:
        return params[bid, comp]
    if mode == CURVED3D_VALUE_RAMP_VECTOR:
        scale = wp.float64(1.0)
        active = params[bid, 3]
        if active > wp.float64(0.0) and t < active:
            scale = wp.sin(PI * t / active) * wp.sin(PI * t / active)
        return params[bid, comp] * scale
    if mode == CURVED3D_VALUE_AFFINE_VELOCITY:
        active = params[bid, 12]
        if active > wp.float64(0.0) and t > active:
            return wp.float64(0.0)
        out = params[bid, 0] * x + params[bid, 1] * y + params[bid, 2] * z + params[bid, 9]
        if comp == 1:
            out = params[bid, 3] * x + params[bid, 4] * y + params[bid, 5] * z + params[bid, 10]
        elif comp == 2:
            out = params[bid, 6] * x + params[bid, 7] * y + params[bid, 8] * z + params[bid, 11]
        return out
    if mode == CURVED3D_VALUE_SINE2_HOLD_VECTOR:
        active = params[bid, 3]
        denom = params[bid, 4]
        if denom <= wp.float64(0.0):
            denom = wp.float64(2.0) * active
        scale = wp.float64(1.0)
        if active > wp.float64(0.0) and t < active:
            scale = wp.sin(PI * t / denom) * wp.sin(PI * t / denom)
        return params[bid, comp] * scale
    if mode == CURVED3D_VALUE_PIOLA_NORMAL:
        return (
            params[bid, 3 * comp + 0] * nx
            + params[bid, 3 * comp + 1] * ny
            + params[bid, 3 * comp + 2] * nz
        )
    if mode == CURVED3D_VALUE_AFFINE_NEO_HOOKE_PIOLA_NORMAL:
        tt = t
        active = params[bid, 11]
        if active > wp.float64(0.0) and tt > active:
            tt = active
        F11 = wp.float64(1.0) + tt * params[bid, 0]
        F12 = tt * params[bid, 1]
        F13 = tt * params[bid, 2]
        F21 = tt * params[bid, 3]
        F22 = wp.float64(1.0) + tt * params[bid, 4]
        F23 = tt * params[bid, 5]
        F31 = tt * params[bid, 6]
        F32 = tt * params[bid, 7]
        F33 = wp.float64(1.0) + tt * params[bid, 8]
        P = _piola_mat(
            F11,
            F12,
            F13,
            F21,
            F22,
            F23,
            F31,
            F32,
            F33,
            MATERIAL_NEO_HOOKE,
            params[bid, 9],
            params[bid, 10],
            wp.float64(0.0),
            wp.float64(0.0),
            wp.float64(0.0),
            j_floor,
        )
        return P[comp, 0] * nx + P[comp, 1] * ny + P[comp, 2] * nz
    if mode == CURVED3D_VALUE_NORMAL_TRACTION:
        out = params[bid, 0] * nx
        if comp == 1:
            out = params[bid, 0] * ny
        elif comp == 2:
            out = params[bid, 0] * nz
        return out
    if mode == CURVED3D_VALUE_SINE2_NORMAL_TRACTION:
        amp = params[bid, 0]
        active = params[bid, 1]
        denom = params[bid, 2]
        if denom <= wp.float64(0.0):
            denom = wp.float64(2.0) * active
        scale = wp.float64(1.0)
        if active > wp.float64(0.0) and t < active:
            scale = wp.sin(PI * t / denom) * wp.sin(PI * t / denom)
        out = amp * scale * nx
        if comp == 1:
            out = amp * scale * ny
        elif comp == 2:
            out = amp * scale * nz
        return out
    if mode == CURVED3D_VALUE_Z_PULL_TORSION_TRACTION:
        active = params[bid, 4]
        denom = params[bid, 5]
        if denom <= wp.float64(0.0):
            denom = wp.float64(2.0) * active
        scale = wp.float64(1.0)
        if active > wp.float64(0.0) and t < active:
            scale = wp.sin(PI * t / denom) * wp.sin(PI * t / denom)
        dx = x - params[bid, 0]
        dy = y - params[bid, 1]
        axial = params[bid, 2]
        alpha = params[bid, 3]
        if comp == 0:
            return -scale * alpha * dy
        if comp == 1:
            return scale * alpha * dx
        return scale * axial
    if mode == CURVED3D_VALUE_Z_STRETCH_EXACT_TWIST_VELOCITY:
        active = params[bid, 4]
        if active <= wp.float64(0.0) or t >= active:
            return wp.float64(0.0)
        phase = PI * t / active
        s = wp.float64(0.5) * (wp.float64(1.0) - wp.cos(phase))
        sdot = wp.float64(0.5) * PI / active * wp.sin(phase)
        theta = params[bid, 3] * s
        thetadot = params[bid, 3] * sdot
        dx = x - params[bid, 0]
        dy = y - params[bid, 1]
        if comp == 0:
            return thetadot * (-wp.sin(theta) * dx - wp.cos(theta) * dy)
        if comp == 1:
            return thetadot * (wp.cos(theta) * dx - wp.sin(theta) * dy)
        return params[bid, 2] * sdot
    return wp.float64(0.0)


@wp.func
def _make_F_from_h(
    h0: wp.float64,
    h1: wp.float64,
    h2: wp.float64,
    n: wp.vec3d,
    t1: wp.vec3d,
    t2: wp.vec3d,
    g1: wp.vec3d,
    g2: wp.vec3d,
) -> wp.mat33d:
    return wp.mat33d(
        h0 * n[0] + g1[0] * t1[0] + g2[0] * t2[0],
        h0 * n[1] + g1[0] * t1[1] + g2[0] * t2[1],
        h0 * n[2] + g1[0] * t1[2] + g2[0] * t2[2],
        h1 * n[0] + g1[1] * t1[0] + g2[1] * t2[0],
        h1 * n[1] + g1[1] * t1[1] + g2[1] * t2[1],
        h1 * n[2] + g1[1] * t1[2] + g2[1] * t2[2],
        h2 * n[0] + g1[2] * t1[0] + g2[2] * t2[0],
        h2 * n[1] + g1[2] * t1[1] + g2[2] * t2[1],
        h2 * n[2] + g1[2] * t1[2] + g2[2] * t2[2],
    )


@wp.func
def _normal_residual_h(
    h0: wp.float64,
    h1: wp.float64,
    h2: wp.float64,
    n: wp.vec3d,
    t1: wp.vec3d,
    t2: wp.vec3d,
    g1: wp.vec3d,
    g2: wp.vec3d,
    T: wp.vec3d,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
) -> wp.vec3d:
    F = _make_F_from_h(h0, h1, h2, n, t1, t2, g1, g2)
    P = _piola_mat(
        F[0, 0],
        F[0, 1],
        F[0, 2],
        F[1, 0],
        F[1, 1],
        F[1, 2],
        F[2, 0],
        F[2, 1],
        F[2, 2],
        model,
        lam,
        mu,
        mat_p0,
        mat_p1,
        mat_p2,
        j_floor,
    )
    return wp.vec3d(
        P[0, 0] * n[0] + P[0, 1] * n[1] + P[0, 2] * n[2] - T[0],
        P[1, 0] * n[0] + P[1, 1] * n[1] + P[1, 2] * n[2] - T[1],
        P[2, 0] * n[0] + P[2, 1] * n[1] + P[2, 2] * n[2] - T[2],
    )


@wp.func
def _neo_hooke_acoustic_normal_3d(
    F: wp.mat33d, n: wp.vec3d, lam: wp.float64, mu: wp.float64, j_floor: wp.float64
) -> wp.mat33d:
    F11 = F[0, 0]
    F12 = F[0, 1]
    F13 = F[0, 2]
    F21 = F[1, 0]
    F22 = F[1, 1]
    F23 = F[1, 2]
    F31 = F[2, 0]
    F32 = F[2, 1]
    F33 = F[2, 2]
    J = _det3_wp(F11, F12, F13, F21, F22, F23, F31, F32, F33)
    if J < j_floor:
        J = j_floor
    H11 = (F22 * F33 - F23 * F32) / J
    H12 = (F23 * F31 - F21 * F33) / J
    H13 = (F21 * F32 - F22 * F31) / J
    H21 = (F13 * F32 - F12 * F33) / J
    H22 = (F11 * F33 - F13 * F31) / J
    H23 = (F12 * F31 - F11 * F32) / J
    H31 = (F12 * F23 - F13 * F22) / J
    H32 = (F13 * F21 - F11 * F23) / J
    H33 = (F11 * F22 - F12 * F21) / J
    beta = wp.float64(0.5) * lam * (J * J - wp.float64(1.0)) - mu
    coeff = lam * J * J - beta
    n2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2]
    v0 = H11 * n[0] + H12 * n[1] + H13 * n[2]
    v1 = H21 * n[0] + H22 * n[1] + H23 * n[2]
    v2 = H31 * n[0] + H32 * n[1] + H33 * n[2]
    return wp.mat33d(
        mu * n2 + coeff * v0 * v0,
        coeff * v0 * v1,
        coeff * v0 * v2,
        coeff * v1 * v0,
        mu * n2 + coeff * v1 * v1,
        coeff * v1 * v2,
        coeff * v2 * v0,
        coeff * v2 * v1,
        mu * n2 + coeff * v2 * v2,
    )


@wp.func
def _energy_Fb_neo_hooke_3d(
    F: wp.mat33d,
    n: wp.vec3d,
    T: wp.vec3d,
    kind0: int,
    kind1: int,
    kind2: int,
    lam: wp.float64,
    mu: wp.float64,
    j_floor: wp.float64,
) -> wp.mat33d:
    """Minimise W(G+h tensor n)-T.h over the Neumann normal-image rows.

    For the quadratic-volumetric NH law the minimiser is a positive quadratic
    root.  G, n and the Dirichlet rows are held fixed.  A zero matrix signals
    a degenerate/out-of-floor state to the legacy bounded Newton fallback.
    """
    cof = wp.mat33d(
        F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1],
        F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2],
        F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0],
        F[0, 2] * F[2, 1] - F[0, 1] * F[2, 2],
        F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0],
        F[0, 1] * F[2, 0] - F[0, 0] * F[2, 1],
        F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1],
        F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2],
        F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0],
    )
    a = cof * n
    h_old = F * n
    if kind0 == BC_NEUMANN and kind1 == BC_NEUMANN and kind2 == BC_NEUMANN:
        a2 = wp.dot(a, a)
        if a2 > wp.float64(1.0e-28):
            area = wp.sqrt(a2)
            e = a / area
            tau = wp.dot(T, e)
            aa = mu + wp.float64(0.5) * lam * a2
            bb = mu + wp.float64(0.5) * lam
            disc = wp.sqrt(tau * tau + wp.float64(4.0) * aa * bb)
            s = (tau + disc) / (wp.float64(2.0) * aa)
            if tau < wp.float64(0.0):
                s = wp.float64(2.0) * bb / (disc - tau)
            if area * s > j_floor:
                h = (T - tau * e) / mu + s * e
                return F + wp.outer(h - h_old, n)
    else:
        # J=d+a_N.h_N; d comes from the unchanged Dirichlet rows.
        an = wp.vec3d(wp.float64(0.0))
        tn = wp.vec3d(wp.float64(0.0))
        fixed = h_old
        if kind0 == BC_NEUMANN:
            an[0] = a[0]
            tn[0] = T[0]
            fixed[0] = wp.float64(0.0)
        if kind1 == BC_NEUMANN:
            an[1] = a[1]
            tn[1] = T[1]
            fixed[1] = wp.float64(0.0)
        if kind2 == BC_NEUMANN:
            an[2] = a[2]
            tn[2] = T[2]
            fixed[2] = wp.float64(0.0)
        a2 = wp.dot(an, an)
        d = wp.dot(a, fixed)
        if a2 > wp.float64(1.0e-28):
            aa = mu + wp.float64(0.5) * lam * a2
            bb = (mu + wp.float64(0.5) * lam) * a2
            cc = mu * d + wp.dot(an, tn)
            disc = wp.sqrt(cc * cc + wp.float64(4.0) * aa * bb)
            J = (cc + disc) / (wp.float64(2.0) * aa)
            if cc < wp.float64(0.0):
                J = wp.float64(2.0) * bb / (disc - cc)
            if J > j_floor:
                area = wp.sqrt(a2)
                e = an / area
                h = fixed + (tn - wp.dot(tn, e) * e) / mu + ((J - d) / area) * e
                return F + wp.outer(h - h_old, n)
        elif a2 == wp.float64(0.0) and d > j_floor:
            return F + wp.outer(fixed + tn / mu - h_old, n)
    return wp.mat33d(wp.float64(0.0))


@wp.func
def _solve_Fb_arbitrary_normal(
    F11: wp.float64,
    F12: wp.float64,
    F13: wp.float64,
    F21: wp.float64,
    F22: wp.float64,
    F23: wp.float64,
    F31: wp.float64,
    F32: wp.float64,
    F33: wp.float64,
    nx: wp.float64,
    ny: wp.float64,
    nz: wp.float64,
    tx: wp.float64,
    ty: wp.float64,
    tz: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
    normal_jacobian: int,
) -> wp.mat33d:
    n = _normalize3(wp.vec3d(nx, ny, nz))
    if normal_jacobian == NORMAL_SOLVER_ENERGY and model == MATERIAL_NEO_HOOKE:
        Fb = _energy_Fb_neo_hooke_3d(
            wp.mat33d(F11, F12, F13, F21, F22, F23, F31, F32, F33),
            n,
            wp.vec3d(tx, ty, tz),
            BC_NEUMANN,
            BC_NEUMANN,
            BC_NEUMANN,
            lam,
            mu,
            j_floor,
        )
        if wp.determinant(Fb) > j_floor:
            return Fb
    ref = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
    if wp.abs(n[0]) >= wp.float64(0.75):
        ref = wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0))
    t1 = _normalize3(_cross3(n, ref))
    t2 = _cross3(n, t1)
    h0 = F11 * n[0] + F12 * n[1] + F13 * n[2]
    h1 = F21 * n[0] + F22 * n[1] + F23 * n[2]
    h2 = F31 * n[0] + F32 * n[1] + F33 * n[2]
    g1 = wp.vec3d(
        F11 * t1[0] + F12 * t1[1] + F13 * t1[2],
        F21 * t1[0] + F22 * t1[1] + F23 * t1[2],
        F31 * t1[0] + F32 * t1[1] + F33 * t1[2],
    )
    g2 = wp.vec3d(
        F11 * t2[0] + F12 * t2[1] + F13 * t2[2],
        F21 * t2[0] + F22 * t2[1] + F23 * t2[2],
        F31 * t2[0] + F32 * t2[1] + F33 * t2[2],
    )
    T = wp.vec3d(tx, ty, tz)
    eps = wp.float64(1.0e-6)
    for _it in range(16):
        r = _normal_residual_h(
            h0, h1, h2, n, t1, t2, g1, g2, T, model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor
        )
        rnorm = wp.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
        if rnorm < wp.float64(1.0e-10):
            break
        a = wp.float64(0.0)
        b = wp.float64(0.0)
        c = wp.float64(0.0)
        d = wp.float64(0.0)
        e = wp.float64(0.0)
        f = wp.float64(0.0)
        g = wp.float64(0.0)
        hh = wp.float64(0.0)
        ii = wp.float64(0.0)
        if normal_jacobian != NORMAL_JACOBIAN_FINITE_DIFFERENCE and model == MATERIAL_NEO_HOOKE:
            F = _make_F_from_h(h0, h1, h2, n, t1, t2, g1, g2)
            Qn = _neo_hooke_acoustic_normal_3d(F, n, lam, mu, j_floor)
            a = Qn[0, 0]
            b = Qn[0, 1]
            c = Qn[0, 2]
            d = Qn[1, 0]
            e = Qn[1, 1]
            f = Qn[1, 2]
            g = Qn[2, 0]
            hh = Qn[2, 1]
            ii = Qn[2, 2]
        else:
            rp0 = _normal_residual_h(
                h0 + eps,
                h1,
                h2,
                n,
                t1,
                t2,
                g1,
                g2,
                T,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
            )
            rp1 = _normal_residual_h(
                h0,
                h1 + eps,
                h2,
                n,
                t1,
                t2,
                g1,
                g2,
                T,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
            )
            rp2 = _normal_residual_h(
                h0,
                h1,
                h2 + eps,
                n,
                t1,
                t2,
                g1,
                g2,
                T,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
            )
            a = (rp0[0] - r[0]) / eps
            b = (rp1[0] - r[0]) / eps
            c = (rp2[0] - r[0]) / eps
            d = (rp0[1] - r[1]) / eps
            e = (rp1[1] - r[1]) / eps
            f = (rp2[1] - r[1]) / eps
            g = (rp0[2] - r[2]) / eps
            hh = (rp1[2] - r[2]) / eps
            ii = (rp2[2] - r[2]) / eps
        det = a * (e * ii - f * hh) - b * (d * ii - f * g) + c * (d * hh - e * g)
        if wp.abs(det) < wp.float64(1.0e-14):
            break
        inv00 = (e * ii - f * hh) / det
        inv01 = (c * hh - b * ii) / det
        inv02 = (b * f - c * e) / det
        inv10 = (f * g - d * ii) / det
        inv11 = (a * ii - c * g) / det
        inv12 = (c * d - a * f) / det
        inv20 = (d * hh - e * g) / det
        inv21 = (b * g - a * hh) / det
        inv22 = (a * e - b * d) / det
        delta0 = -(inv00 * r[0] + inv01 * r[1] + inv02 * r[2])
        delta1 = -(inv10 * r[0] + inv11 * r[1] + inv12 * r[2])
        delta2 = -(inv20 * r[0] + inv21 * r[1] + inv22 * r[2])
        step = wp.float64(1.0)
        for _ls in range(12):
            c0 = h0 + step * delta0
            c1 = h1 + step * delta1
            c2 = h2 + step * delta2
            F = _make_F_from_h(c0, c1, c2, n, t1, t2, g1, g2)
            J = _det3_wp(
                F[0, 0], F[0, 1], F[0, 2], F[1, 0], F[1, 1], F[1, 2], F[2, 0], F[2, 1], F[2, 2]
            )
            if J > j_floor:
                h0 = c0
                h1 = c1
                h2 = c2
                break
            step = wp.float64(0.5) * step
    return _make_F_from_h(h0, h1, h2, n, t1, t2, g1, g2)


@wp.func
def _solve_Fb_componentwise_normal(
    F11: wp.float64,
    F12: wp.float64,
    F13: wp.float64,
    F21: wp.float64,
    F22: wp.float64,
    F23: wp.float64,
    F31: wp.float64,
    F32: wp.float64,
    F33: wp.float64,
    nx: wp.float64,
    ny: wp.float64,
    nz: wp.float64,
    tx: wp.float64,
    ty: wp.float64,
    tz: wp.float64,
    kind0: int,
    kind1: int,
    kind2: int,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
    normal_jacobian: int,
) -> wp.mat33d:
    """Solve only the normal-gradient rows carrying Neumann data.

    A Dirichlet displacement row keeps its extrapolated normal derivative.
    Its corresponding traction is therefore not prescribed and is retained in
    ``P(F_b) n`` as the boundary reaction.
    """
    n = _normalize3(wp.vec3d(nx, ny, nz))
    if normal_jacobian == NORMAL_SOLVER_ENERGY and model == MATERIAL_NEO_HOOKE:
        Fb = _energy_Fb_neo_hooke_3d(
            wp.mat33d(F11, F12, F13, F21, F22, F23, F31, F32, F33),
            n,
            wp.vec3d(tx, ty, tz),
            kind0,
            kind1,
            kind2,
            lam,
            mu,
            j_floor,
        )
        if wp.determinant(Fb) > j_floor:
            return Fb
    ref = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
    if wp.abs(n[0]) >= wp.float64(0.75):
        ref = wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0))
    t1 = _normalize3(_cross3(n, ref))
    t2 = _cross3(n, t1)
    h0 = F11 * n[0] + F12 * n[1] + F13 * n[2]
    h1 = F21 * n[0] + F22 * n[1] + F23 * n[2]
    h2 = F31 * n[0] + F32 * n[1] + F33 * n[2]
    g1 = wp.vec3d(
        F11 * t1[0] + F12 * t1[1] + F13 * t1[2],
        F21 * t1[0] + F22 * t1[1] + F23 * t1[2],
        F31 * t1[0] + F32 * t1[1] + F33 * t1[2],
    )
    g2 = wp.vec3d(
        F11 * t2[0] + F12 * t2[1] + F13 * t2[2],
        F21 * t2[0] + F22 * t2[1] + F23 * t2[2],
        F31 * t2[0] + F32 * t2[1] + F33 * t2[2],
    )
    T = wp.vec3d(tx, ty, tz)
    eps = wp.float64(1.0e-6)
    for _it in range(16):
        r = _normal_residual_h(
            h0, h1, h2, n, t1, t2, g1, g2, T, model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor
        )
        r0 = r[0]
        r1 = r[1]
        r2 = r[2]
        if kind0 != BC_NEUMANN:
            r0 = wp.float64(0.0)
        if kind1 != BC_NEUMANN:
            r1 = wp.float64(0.0)
        if kind2 != BC_NEUMANN:
            r2 = wp.float64(0.0)
        rnorm = wp.sqrt(r0 * r0 + r1 * r1 + r2 * r2)
        if rnorm < wp.float64(1.0e-10):
            break
        a = wp.float64(0.0)
        b = wp.float64(0.0)
        c = wp.float64(0.0)
        d = wp.float64(0.0)
        e = wp.float64(0.0)
        f = wp.float64(0.0)
        g = wp.float64(0.0)
        hh = wp.float64(0.0)
        ii = wp.float64(0.0)
        if normal_jacobian != NORMAL_JACOBIAN_FINITE_DIFFERENCE and model == MATERIAL_NEO_HOOKE:
            F = _make_F_from_h(h0, h1, h2, n, t1, t2, g1, g2)
            Qn = _neo_hooke_acoustic_normal_3d(F, n, lam, mu, j_floor)
            a = Qn[0, 0]
            b = Qn[0, 1]
            c = Qn[0, 2]
            d = Qn[1, 0]
            e = Qn[1, 1]
            f = Qn[1, 2]
            g = Qn[2, 0]
            hh = Qn[2, 1]
            ii = Qn[2, 2]
        else:
            rp0 = _normal_residual_h(
                h0 + eps,
                h1,
                h2,
                n,
                t1,
                t2,
                g1,
                g2,
                T,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
            )
            rp1 = _normal_residual_h(
                h0,
                h1 + eps,
                h2,
                n,
                t1,
                t2,
                g1,
                g2,
                T,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
            )
            rp2 = _normal_residual_h(
                h0,
                h1,
                h2 + eps,
                n,
                t1,
                t2,
                g1,
                g2,
                T,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
            )
            a = (rp0[0] - r[0]) / eps
            b = (rp1[0] - r[0]) / eps
            c = (rp2[0] - r[0]) / eps
            d = (rp0[1] - r[1]) / eps
            e = (rp1[1] - r[1]) / eps
            f = (rp2[1] - r[1]) / eps
            g = (rp0[2] - r[2]) / eps
            hh = (rp1[2] - r[2]) / eps
            ii = (rp2[2] - r[2]) / eps
        if kind0 != BC_NEUMANN:
            a = wp.float64(1.0)
            b = wp.float64(0.0)
            c = wp.float64(0.0)
        if kind1 != BC_NEUMANN:
            d = wp.float64(0.0)
            e = wp.float64(1.0)
            f = wp.float64(0.0)
        if kind2 != BC_NEUMANN:
            g = wp.float64(0.0)
            hh = wp.float64(0.0)
            ii = wp.float64(1.0)
        det = a * (e * ii - f * hh) - b * (d * ii - f * g) + c * (d * hh - e * g)
        if wp.abs(det) < wp.float64(1.0e-14):
            break
        inv00 = (e * ii - f * hh) / det
        inv01 = (c * hh - b * ii) / det
        inv02 = (b * f - c * e) / det
        inv10 = (f * g - d * ii) / det
        inv11 = (a * ii - c * g) / det
        inv12 = (c * d - a * f) / det
        inv20 = (d * hh - e * g) / det
        inv21 = (b * g - a * hh) / det
        inv22 = (a * e - b * d) / det
        delta0 = -(inv00 * r0 + inv01 * r1 + inv02 * r2)
        delta1 = -(inv10 * r0 + inv11 * r1 + inv12 * r2)
        delta2 = -(inv20 * r0 + inv21 * r1 + inv22 * r2)
        step = wp.float64(1.0)
        for _ls in range(12):
            c0 = h0
            c1 = h1
            c2 = h2
            if kind0 == BC_NEUMANN:
                c0 = h0 + step * delta0
            if kind1 == BC_NEUMANN:
                c1 = h1 + step * delta1
            if kind2 == BC_NEUMANN:
                c2 = h2 + step * delta2
            F = _make_F_from_h(c0, c1, c2, n, t1, t2, g1, g2)
            J = _det3_wp(
                F[0, 0], F[0, 1], F[0, 2], F[1, 0], F[1, 1], F[1, 2], F[2, 0], F[2, 1], F[2, 2]
            )
            if J > j_floor:
                h0 = c0
                h1 = c1
                h2 = c2
                break
            step = wp.float64(0.5) * step
    return _make_F_from_h(h0, h1, h2, n, t1, t2, g1, g2)


@wp.kernel
def _fill_inactive(f: wp.array4d(dtype=wp.float64), active: wp.array3d(dtype=wp.int32)):
    i, j, k = wp.tid()
    if active[i, j, k] != 0:
        return
    for q in range(6):
        for a in range(12):
            v = wp.float64(0.0)
            if a == 3 or a == 7 or a == 11:
                v = wp.float64(1.0) / wp.float64(6.0)
            f[q * 12 + a, i, j, k] = v


@wp.kernel
def _stream_masked(
    fpost: wp.array4d(dtype=wp.float64),
    f1: wp.array4d(dtype=wp.float64),
    active: wp.array3d(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    if active[i, j, k] == 0:
        return
    for q in range(6):
        ii = i + _dx(q)
        jj = j + _dy(q)
        kk = k + _dz(q)
        if (
            ii >= 0
            and ii < nx
            and jj >= 0
            and jj < ny
            and kk >= 0
            and kk < nz
            and active[ii, jj, kk] != 0
        ):
            for a in range(12):
                f1[q * 12 + a, ii, jj, kk] = fpost[q * 12 + a, i, j, k]


@wp.kernel
def _set_ustar(
    displ: wp.array4d(dtype=wp.float64),
    U: wp.array4d(dtype=wp.float64),
    us: wp.array4d(dtype=wp.float64),
    active: wp.array3d(dtype=wp.int32),
    dt: wp.float64,
):
    i, j, k = wp.tid()
    if active[i, j, k] != 0:
        us[0, i, j, k] = displ[0, i, j, k] - wp.float64(0.5) * dt * U[0, i, j, k]
        us[1, i, j, k] = displ[1, i, j, k] - wp.float64(0.5) * dt * U[1, i, j, k]
        us[2, i, j, k] = displ[2, i, j, k] - wp.float64(0.5) * dt * U[2, i, j, k]
    else:
        us[0, i, j, k] = wp.float64(0.0)
        us[1, i, j, k] = wp.float64(0.0)
        us[2, i, j, k] = wp.float64(0.0)


@wp.kernel
def _prepare_links_gpu(
    U: wp.array4d(dtype=wp.float64),
    li: wp.array(dtype=wp.int32),
    lj: wp.array(dtype=wp.int32),
    lk: wp.array(dtype=wp.int32),
    lbid: wp.array(dtype=wp.int32),
    lxb: wp.array(dtype=wp.float64),
    lyb: wp.array(dtype=wp.float64),
    lzb: wp.array(dtype=wp.float64),
    lnx: wp.array(dtype=wp.float64),
    lny: wp.array(dtype=wp.float64),
    lnz: wp.array(dtype=wp.float64),
    kind_by_id: wp.array(dtype=wp.int32),
    component_kind_by_id: wp.array2d(dtype=wp.int32),
    mode_by_id: wp.array(dtype=wp.int32),
    params_by_id: wp.array2d(dtype=wp.float64),
    lkind: wp.array(dtype=wp.int32),
    lcomponent_kind: wp.array2d(dtype=wp.int32),
    lbv: wp.array2d(dtype=wp.float64),
    lFb: wp.array2d(dtype=wp.float64),
    lPb: wp.array2d(dtype=wp.float64),
    nlinks: int,
    t: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
    normal_jacobian: int,
):
    ell = wp.tid()
    if ell >= nlinks:
        return
    bid = lbid[ell]
    kind = kind_by_id[bid]
    mode = mode_by_id[bid]
    kind0 = component_kind_by_id[0, bid]
    kind1 = component_kind_by_id[1, bid]
    kind2 = component_kind_by_id[2, bid]
    x = lxb[ell]
    y = lyb[ell]
    z = lzb[ell]
    nx = lnx[ell]
    ny = lny[ell]
    nz = lnz[ell]
    tx = _curved_value_comp(mode, params_by_id, bid, 0, x, y, z, nx, ny, nz, t, j_floor)
    ty = _curved_value_comp(mode, params_by_id, bid, 1, x, y, z, nx, ny, nz, t, j_floor)
    tz = _curved_value_comp(mode, params_by_id, bid, 2, x, y, z, nx, ny, nz, t, j_floor)
    lkind[ell] = kind
    lcomponent_kind[0, ell] = kind0
    lcomponent_kind[1, ell] = kind1
    lcomponent_kind[2, ell] = kind2
    lbv[0, ell] = tx
    lbv[1, ell] = ty
    lbv[2, ell] = tz
    Fb = wp.mat33d(
        wp.float64(1.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(1.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(1.0),
    )
    Pb = wp.mat33d(
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
    )
    if kind0 == BC_NEUMANN or kind1 == BC_NEUMANN or kind2 == BC_NEUMANN:
        i = li[ell]
        j = lj[ell]
        k = lk[ell]
        if kind0 == BC_NEUMANN and kind1 == BC_NEUMANN and kind2 == BC_NEUMANN:
            Fb = _solve_Fb_arbitrary_normal(
                U[3, i, j, k],
                U[4, i, j, k],
                U[5, i, j, k],
                U[6, i, j, k],
                U[7, i, j, k],
                U[8, i, j, k],
                U[9, i, j, k],
                U[10, i, j, k],
                U[11, i, j, k],
                nx,
                ny,
                nz,
                tx,
                ty,
                tz,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
                normal_jacobian,
            )
        else:
            Fb = _solve_Fb_componentwise_normal(
                U[3, i, j, k],
                U[4, i, j, k],
                U[5, i, j, k],
                U[6, i, j, k],
                U[7, i, j, k],
                U[8, i, j, k],
                U[9, i, j, k],
                U[10, i, j, k],
                U[11, i, j, k],
                nx,
                ny,
                nz,
                tx,
                ty,
                tz,
                kind0,
                kind1,
                kind2,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
                normal_jacobian,
            )
        Pb = _piola_mat(
            Fb[0, 0],
            Fb[0, 1],
            Fb[0, 2],
            Fb[1, 0],
            Fb[1, 1],
            Fb[1, 2],
            Fb[2, 0],
            Fb[2, 1],
            Fb[2, 2],
            model,
            lam,
            mu,
            mat_p0,
            mat_p1,
            mat_p2,
            j_floor,
        )
    lFb[0, ell] = Fb[0, 0]
    lFb[1, ell] = Fb[0, 1]
    lFb[2, ell] = Fb[0, 2]
    lFb[3, ell] = Fb[1, 0]
    lFb[4, ell] = Fb[1, 1]
    lFb[5, ell] = Fb[1, 2]
    lFb[6, ell] = Fb[2, 0]
    lFb[7, ell] = Fb[2, 1]
    lFb[8, ell] = Fb[2, 2]
    lPb[0, ell] = Pb[0, 0]
    lPb[1, ell] = Pb[0, 1]
    lPb[2, ell] = Pb[0, 2]
    lPb[3, ell] = Pb[1, 0]
    lPb[4, ell] = Pb[1, 1]
    lPb[5, ell] = Pb[1, 2]
    lPb[6, ell] = Pb[2, 0]
    lPb[7, ell] = Pb[2, 1]
    lPb[8, ell] = Pb[2, 2]


@wp.kernel
def _apply_links(
    fpost: wp.array4d(dtype=wp.float64),
    f1: wp.array4d(dtype=wp.float64),
    active: wp.array3d(dtype=wp.int32),
    li: wp.array(dtype=wp.int32),
    lj: wp.array(dtype=wp.int32),
    lk: wp.array(dtype=wp.int32),
    lq: wp.array(dtype=wp.int32),
    leta: wp.array(dtype=wp.float64),
    lkind: wp.array(dtype=wp.int32),
    lcomponent_kind: wp.array2d(dtype=wp.int32),
    lbv: wp.array2d(dtype=wp.float64),
    lFb: wp.array2d(dtype=wp.float64),
    lPb: wp.array2d(dtype=wp.float64),
    nlinks: int,
    nx: int,
    ny: int,
    nz: int,
    c: wp.float64,
    recon: int,
):
    ell = wp.tid()
    if ell >= nlinks:
        return
    i = li[ell]
    j = lj[ell]
    k = lk[ell]
    if active[i, j, k] == 0:
        return
    qo = lq[ell]
    qi = _opp(qo)
    eta = leta[ell]
    if recon == BOUNDARY_RECON_HALF_WAY:
        eta = wp.float64(0.5)
    dxin = wp.float64(_dx(qi))
    dyin = wp.float64(_dy(qi))
    dzin = wp.float64(_dz(qi))
    iu = i - _dx(qo)
    ju = j - _dy(qo)
    ku = k - _dz(qo)
    ok = False
    if (
        iu >= 0
        and iu < nx
        and ju >= 0
        and ju < ny
        and ku >= 0
        and ku < nz
        and active[iu, ju, ku] != 0
    ):
        ok = True
    v0 = lbv[0, ell]
    v1 = lbv[1, ell]
    v2 = lbv[2, ell]
    for a in range(12):
        row = a
        if a >= 3:
            row = (a - 3) // 3
        kind = lcomponent_kind[row, ell]
        D = wp.float64(1.0)
        S = wp.float64(0.0)
        if kind == BC_DIRICHLET:
            if a == 0:
                D = wp.float64(-1.0)
                S = v0 / wp.float64(3.0)
            elif a == 1:
                D = wp.float64(-1.0)
                S = v1 / wp.float64(3.0)
            elif a == 2:
                D = wp.float64(-1.0)
                S = v2 / wp.float64(3.0)
            else:
                row = (a - 3) // 3
                col = (a - 3) - row * 3
                vi = v0
                if row == 1:
                    vi = v1
                elif row == 2:
                    vi = v2
                dB = dxin
                if col == 1:
                    dB = dyin
                elif col == 2:
                    dB = dzin
                S = -dB * vi / c
        else:
            if a == 0:
                S = -(dxin * lPb[0, ell] + dyin * lPb[1, ell] + dzin * lPb[2, ell]) / c
            elif a == 1:
                S = -(dxin * lPb[3, ell] + dyin * lPb[4, ell] + dzin * lPb[5, ell]) / c
            elif a == 2:
                S = -(dxin * lPb[6, ell] + dyin * lPb[7, ell] + dzin * lPb[8, ell]) / c
            else:
                D = wp.float64(-1.0)
                S = lFb[a - 3, ell] / wp.float64(3.0)
        fo = fpost[qo * 12 + a, i, j, k]
        fi = fpost[qi * 12 + a, i, j, k]
        if eta >= wp.float64(0.5):
            alpha = wp.float64(1.0) / (wp.float64(2.0) * eta)
            val = alpha * (D * fo + S) + (wp.float64(1.0) - alpha) * fi
        else:
            up = fo
            if ok:
                up = fpost[qo * 12 + a, iu, ju, ku]
            val = (
                D * (wp.float64(2.0) * eta * fo + (wp.float64(1.0) - wp.float64(2.0) * eta) * up)
                + S
            )
        f1[qi * 12 + a, i, j, k] = val


@wp.kernel
def _sparse_clear(arr: wp.array2d(dtype=wp.float64)):
    a, p = wp.tid()
    arr[a, p] = wp.float64(0.0)


@wp.kernel
def _sparse_set_ustar(
    displacement: wp.array2d(dtype=wp.float64),
    U: wp.array2d(dtype=wp.float64),
    u_star: wp.array2d(dtype=wp.float64),
    dt: wp.float64,
):
    p = wp.tid()
    u_star[0, p] = displacement[0, p] - wp.float64(0.5) * dt * U[0, p]
    u_star[1, p] = displacement[1, p] - wp.float64(0.5) * dt * U[1, p]
    u_star[2, p] = displacement[2, p] - wp.float64(0.5) * dt * U[2, p]


@wp.kernel
def _sparse_refresh_kernel(
    f0: wp.array2d(dtype=wp.float64),
    u: wp.array2d(dtype=wp.float64),
    u_star: wp.array2d(dtype=wp.float64),
    U: wp.array2d(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
    source_params: wp.array(dtype=wp.float64),
    dt: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
    source_mode: int,
):
    p = wp.tid()
    vx = wp.float64(0.0)
    vy = wp.float64(0.0)
    vz = wp.float64(0.0)
    F11 = wp.float64(0.0)
    F12 = wp.float64(0.0)
    F13 = wp.float64(0.0)
    F21 = wp.float64(0.0)
    F22 = wp.float64(0.0)
    F23 = wp.float64(0.0)
    F31 = wp.float64(0.0)
    F32 = wp.float64(0.0)
    F33 = wp.float64(0.0)
    for q in range(6):
        vx += f0[q * 12 + 0, p]
        vy += f0[q * 12 + 1, p]
        vz += f0[q * 12 + 2, p]
        F11 += f0[q * 12 + 3, p]
        F12 += f0[q * 12 + 4, p]
        F13 += f0[q * 12 + 5, p]
        F21 += f0[q * 12 + 6, p]
        F22 += f0[q * 12 + 7, p]
        F23 += f0[q * 12 + 8, p]
        F31 += f0[q * 12 + 9, p]
        F32 += f0[q * 12 + 10, p]
        F33 += f0[q * 12 + 11, p]
    half_dt = wp.float64(0.5) * dt
    if source_mode == SOURCE_CONSTANT:
        vx += half_dt * source_params[0]
        vy += half_dt * source_params[1]
        vz += half_dt * source_params[2]
        F11 += half_dt * source_params[3]
        F12 += half_dt * source_params[4]
        F13 += half_dt * source_params[5]
        F21 += half_dt * source_params[6]
        F22 += half_dt * source_params[7]
        F23 += half_dt * source_params[8]
        F31 += half_dt * source_params[9]
        F32 += half_dt * source_params[10]
        F33 += half_dt * source_params[11]
    elif source_mode == SOURCE_DAMPING:
        denom = wp.float64(1.0) + half_dt * source_params[0]
        vx = vx / denom
        vy = vy / denom
        vz = vz / denom
    U[0, p] = vx
    U[1, p] = vy
    U[2, p] = vz
    U[3, p] = F11
    U[4, p] = F12
    U[5, p] = F13
    U[6, p] = F21
    U[7, p] = F22
    U[8, p] = F23
    U[9, p] = F31
    U[10, p] = F32
    U[11, p] = F33
    u[0, p] = u_star[0, p] + half_dt * vx
    u[1, p] = u_star[1, p] + half_dt * vy
    u[2, p] = u_star[2, p] + half_dt * vz
    J = _det3_wp(F11, F12, F13, F21, F22, F23, F31, F32, F33)
    if J <= j_floor:
        invalid[0] = wp.int32(1)


@wp.kernel
def _sparse_collide_inplace_kernel(
    f0: wp.array2d(dtype=wp.float64),
    u: wp.array2d(dtype=wp.float64),
    u_star: wp.array2d(dtype=wp.float64),
    U: wp.array2d(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
    source_params: wp.array(dtype=wp.float64),
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
):
    p = wp.tid()
    vx = wp.float64(0.0)
    vy = wp.float64(0.0)
    vz = wp.float64(0.0)
    F11 = wp.float64(0.0)
    F12 = wp.float64(0.0)
    F13 = wp.float64(0.0)
    F21 = wp.float64(0.0)
    F22 = wp.float64(0.0)
    F23 = wp.float64(0.0)
    F31 = wp.float64(0.0)
    F32 = wp.float64(0.0)
    F33 = wp.float64(0.0)
    for q in range(6):
        vx += f0[q * 12 + 0, p]
        vy += f0[q * 12 + 1, p]
        vz += f0[q * 12 + 2, p]
        F11 += f0[q * 12 + 3, p]
        F12 += f0[q * 12 + 4, p]
        F13 += f0[q * 12 + 5, p]
        F21 += f0[q * 12 + 6, p]
        F22 += f0[q * 12 + 7, p]
        F23 += f0[q * 12 + 8, p]
        F31 += f0[q * 12 + 9, p]
        F32 += f0[q * 12 + 10, p]
        F33 += f0[q * 12 + 11, p]
    half_dt = wp.float64(0.5) * dt
    if source_mode == SOURCE_CONSTANT:
        vx += half_dt * source_params[0]
        vy += half_dt * source_params[1]
        vz += half_dt * source_params[2]
        F11 += half_dt * source_params[3]
        F12 += half_dt * source_params[4]
        F13 += half_dt * source_params[5]
        F21 += half_dt * source_params[6]
        F22 += half_dt * source_params[7]
        F23 += half_dt * source_params[8]
        F31 += half_dt * source_params[9]
        F32 += half_dt * source_params[10]
        F33 += half_dt * source_params[11]
    elif source_mode == SOURCE_DAMPING:
        denom = wp.float64(1.0) + half_dt * source_params[0]
        vx = vx / denom
        vy = vy / denom
        vz = vz / denom
    U[0, p] = vx
    U[1, p] = vy
    U[2, p] = vz
    U[3, p] = F11
    U[4, p] = F12
    U[5, p] = F13
    U[6, p] = F21
    U[7, p] = F22
    U[8, p] = F23
    U[9, p] = F31
    U[10, p] = F32
    U[11, p] = F33
    ux = u_star[0, p] + half_dt * vx
    uy = u_star[1, p] + half_dt * vy
    uz = u_star[2, p] + half_dt * vz
    u[0, p] = ux
    u[1, p] = uy
    u[2, p] = uz
    u_star[0, p] = ux + half_dt * vx
    u_star[1, p] = uy + half_dt * vy
    u_star[2, p] = uz + half_dt * vz
    J = _det3_wp(F11, F12, F13, F21, F22, F23, F31, F32, F33)
    if J <= j_floor:
        invalid[0] = wp.int32(1)
    for q in range(6):
        for a in range(12):
            feq = _equilibrium_comp(
                q,
                a,
                vx,
                vy,
                vz,
                F11,
                F12,
                F13,
                F21,
                F22,
                F23,
                F31,
                F32,
                F33,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                lattice_speed,
                j_floor,
            )
            old = f0[q * 12 + a, p]
            source_term = _source_comp(a, source_mode, source_params, vx, vy, vz)
            f0[q * 12 + a, p] = (
                old
                + omega * (feq - old)
                + dt
                * (wp.float64(2.0) - omega)
                * (wp.float64(1.0) / wp.float64(12.0))
                * source_term
            )


@wp.kernel
def _sparse_stream_kernel(
    fpost: wp.array2d(dtype=wp.float64),
    f1: wp.array2d(dtype=wp.float64),
    neighbor: wp.array2d(dtype=wp.int32),
):
    p = wp.tid()
    for q in range(6):
        nid = neighbor[q, p]
        if nid >= 0:
            for a in range(12):
                f1[q * 12 + a, nid] = fpost[q * 12 + a, p]


@wp.func
def _local_sine2_integral(time: wp.float64, active: wp.float64, denom: wp.float64) -> wp.float64:
    if active <= wp.float64(0.0):
        return time
    if denom <= wp.float64(0.0):
        denom = active
    t = wp.max(time, wp.float64(0.0))
    if t < active:
        return wp.float64(0.5) * t - denom * wp.sin(wp.float64(2.0) * PI * t / denom) / (
            wp.float64(4.0) * PI
        )
    prefix = wp.float64(0.5) * active - denom * wp.sin(wp.float64(2.0) * PI * active / denom) / (
        wp.float64(4.0) * PI
    )
    return prefix + (t - active)


@wp.func
def _local_dirichlet_displacement_comp(
    mode: int,
    params: wp.array2d(dtype=wp.float64),
    bid: int,
    comp: int,
    x: wp.float64,
    y: wp.float64,
    z: wp.float64,
    time: wp.float64,
) -> wp.float64:
    if mode == CURVED3D_VALUE_ZERO:
        return wp.float64(0.0)
    if mode == CURVED3D_VALUE_CONSTANT_VECTOR:
        return time * params[bid, comp]
    if mode == CURVED3D_VALUE_RAMP_VECTOR:
        return params[bid, comp] * _local_sine2_integral(time, params[bid, 3], params[bid, 3])
    if mode == CURVED3D_VALUE_SINE2_HOLD_VECTOR:
        active = params[bid, 3]
        denom = params[bid, 4]
        if denom <= wp.float64(0.0):
            denom = wp.float64(2.0) * active
        return params[bid, comp] * _local_sine2_integral(time, active, denom)
    if mode == CURVED3D_VALUE_AFFINE_VELOCITY:
        active_time = time
        active = params[bid, 12]
        if active > wp.float64(0.0) and active_time > active:
            active_time = active
        out = params[bid, 0] * x + params[bid, 1] * y + params[bid, 2] * z + params[bid, 9]
        if comp == 1:
            out = params[bid, 3] * x + params[bid, 4] * y + params[bid, 5] * z + params[bid, 10]
        elif comp == 2:
            out = params[bid, 6] * x + params[bid, 7] * y + params[bid, 8] * z + params[bid, 11]
        return active_time * out
    if mode == CURVED3D_VALUE_Z_STRETCH_EXACT_TWIST_VELOCITY:
        active = params[bid, 4]
        s = wp.float64(1.0)
        if active > wp.float64(0.0) and time < active:
            s = wp.float64(0.5) * (wp.float64(1.0) - wp.cos(PI * time / active))
        theta = params[bid, 3] * s
        dx0 = x - params[bid, 0]
        dy0 = y - params[bid, 1]
        c = wp.cos(theta)
        ss = wp.sin(theta)
        if comp == 0:
            return (c - wp.float64(1.0)) * dx0 - ss * dy0
        if comp == 1:
            return ss * dx0 + (c - wp.float64(1.0)) * dy0
        return params[bid, 2] * s
    return wp.float64(0.0)


@wp.kernel
def _sparse_local_bc_clear(
    bc_diag: wp.array2d(dtype=wp.float64), bc_rhs: wp.array2d(dtype=wp.float64)
):
    p = wp.tid()
    for comp in range(3):
        bc_diag[comp, p] = wp.float64(0.0)
        bc_rhs[comp, p] = wp.float64(0.0)


@wp.kernel
def _sparse_local_build_bc(
    U: wp.array2d(dtype=wp.float64),
    active_i: wp.array(dtype=wp.int32),
    active_j: wp.array(dtype=wp.int32),
    active_k: wp.array(dtype=wp.int32),
    link_p: wp.array(dtype=wp.int32),
    link_boundary_id: wp.array(dtype=wp.int32),
    link_xb: wp.array(dtype=wp.float64),
    link_yb: wp.array(dtype=wp.float64),
    link_zb: wp.array(dtype=wp.float64),
    component_kind_by_id: wp.array2d(dtype=wp.int32),
    mode_by_id: wp.array(dtype=wp.int32),
    params_by_id: wp.array2d(dtype=wp.float64),
    n_links: int,
    dirichlet_boundary_id: int,
    time: wp.float64,
    dx: wp.float64,
    dy: wp.float64,
    dz: wp.float64,
    weight: wp.float64,
    bc_diag: wp.array2d(dtype=wp.float64),
    bc_rhs: wp.array2d(dtype=wp.float64),
):
    ell = wp.tid()
    if ell >= n_links:
        return
    bid = link_boundary_id[ell]
    force_all = False
    if dirichlet_boundary_id >= 0:
        if bid != dirichlet_boundary_id:
            return
        force_all = True
    p = link_p[ell]
    if p < 0:
        return
    x = link_xb[ell]
    y = link_yb[ell]
    z = link_zb[ell]
    dbx = x - (wp.float64(active_i[p]) + wp.float64(0.5)) * dx
    dby = y - (wp.float64(active_j[p]) + wp.float64(0.5)) * dy
    dbz = z - (wp.float64(active_k[p]) + wp.float64(0.5)) * dz
    ub0 = _local_dirichlet_displacement_comp(mode_by_id[bid], params_by_id, bid, 0, x, y, z, time)
    ub1 = _local_dirichlet_displacement_comp(mode_by_id[bid], params_by_id, bid, 1, x, y, z, time)
    ub2 = _local_dirichlet_displacement_comp(mode_by_id[bid], params_by_id, bid, 2, x, y, z, time)
    w2 = weight * weight
    if force_all or component_kind_by_id[0, bid] == BC_DIRICHLET:
        rhs = ub0 - ((U[3, p] - wp.float64(1.0)) * dbx + U[4, p] * dby + U[5, p] * dbz)
        wp.atomic_add(bc_diag, 0, p, w2)
        wp.atomic_add(bc_rhs, 0, p, w2 * rhs)
    if force_all or component_kind_by_id[1, bid] == BC_DIRICHLET:
        rhs = ub1 - (U[6, p] * dbx + (U[7, p] - wp.float64(1.0)) * dby + U[8, p] * dbz)
        wp.atomic_add(bc_diag, 1, p, w2)
        wp.atomic_add(bc_rhs, 1, p, w2 * rhs)
    if force_all or component_kind_by_id[2, bid] == BC_DIRICHLET:
        rhs = ub2 - (U[9, p] * dbx + U[10, p] * dby + (U[11, p] - wp.float64(1.0)) * dbz)
        wp.atomic_add(bc_diag, 2, p, w2)
        wp.atomic_add(bc_rhs, 2, p, w2 * rhs)


@wp.kernel
def _sparse_local_displacement_sweep(
    U: wp.array2d(dtype=wp.float64),
    neighbor: wp.array2d(dtype=wp.int32),
    u: wp.array2d(dtype=wp.float64),
    u_next: wp.array2d(dtype=wp.float64),
    bc_diag: wp.array2d(dtype=wp.float64),
    bc_rhs: wp.array2d(dtype=wp.float64),
    dx: wp.float64,
    dy: wp.float64,
    dz: wp.float64,
    relax: wp.float64,
):
    p = wp.tid()
    d0 = bc_diag[0, p]
    d1 = bc_diag[1, p]
    d2 = bc_diag[2, p]
    r0 = bc_rhs[0, p]
    r1 = bc_rhs[1, p]
    r2 = bc_rhs[2, p]
    for q in range(6):
        pn = neighbor[q, p]
        if pn >= 0:
            ddx = wp.float64(_dx(q)) * dx
            ddy = wp.float64(_dy(q)) * dy
            ddz = wp.float64(_dz(q)) * dz
            g0 = wp.float64(0.5) * (
                (U[3, p] + U[3, pn] - wp.float64(2.0)) * ddx
                + (U[4, p] + U[4, pn]) * ddy
                + (U[5, p] + U[5, pn]) * ddz
            )
            g1 = wp.float64(0.5) * (
                (U[6, p] + U[6, pn]) * ddx
                + (U[7, p] + U[7, pn] - wp.float64(2.0)) * ddy
                + (U[8, p] + U[8, pn]) * ddz
            )
            g2 = wp.float64(0.5) * (
                (U[9, p] + U[9, pn]) * ddx
                + (U[10, p] + U[10, pn]) * ddy
                + (U[11, p] + U[11, pn] - wp.float64(2.0)) * ddz
            )
            r0 += u[0, pn] - g0
            r1 += u[1, pn] - g1
            r2 += u[2, pn] - g2
            d0 += wp.float64(1.0)
            d1 += wp.float64(1.0)
            d2 += wp.float64(1.0)
    u_next[0, p] = u[0, p]
    u_next[1, p] = u[1, p]
    u_next[2, p] = u[2, p]
    if d0 > wp.float64(0.0):
        u_next[0, p] = u[0, p] + relax * (r0 / d0 - u[0, p])
    if d1 > wp.float64(0.0):
        u_next[1, p] = u[1, p] + relax * (r1 / d1 - u[1, p])
    if d2 > wp.float64(0.0):
        u_next[2, p] = u[2, p] + relax * (r2 / d2 - u[2, p])


@wp.kernel
def _sparse_local_displacement_commit(
    u_next: wp.array2d(dtype=wp.float64),
    u: wp.array2d(dtype=wp.float64),
    u_star: wp.array2d(dtype=wp.float64),
):
    p = wp.tid()
    for comp in range(3):
        delta = u_next[comp, p] - u[comp, p]
        u[comp, p] = u_next[comp, p]
        u_star[comp, p] = u_star[comp, p] + delta


@wp.func
def _sparse_local_du_axis(
    u: wp.array2d(dtype=wp.float64),
    neighbor: wp.array2d(dtype=wp.int32),
    comp: int,
    p: int,
    q_plus: int,
    q_minus: int,
    h: wp.float64,
) -> wp.float64:
    pp = neighbor[q_plus, p]
    pm = neighbor[q_minus, p]
    if pp >= 0 and pm >= 0:
        return (u[comp, pp] - u[comp, pm]) / (wp.float64(2.0) * h)
    if pp >= 0:
        pp2 = neighbor[q_plus, pp]
        if pp2 >= 0:
            return (
                -wp.float64(3.0) * u[comp, p] + wp.float64(4.0) * u[comp, pp] - u[comp, pp2]
            ) / (wp.float64(2.0) * h)
        return (u[comp, pp] - u[comp, p]) / h
    if pm >= 0:
        pm2 = neighbor[q_minus, pm]
        if pm2 >= 0:
            return (wp.float64(3.0) * u[comp, p] - wp.float64(4.0) * u[comp, pm] + u[comp, pm2]) / (
                wp.float64(2.0) * h
            )
        return (u[comp, p] - u[comp, pm]) / h
    return wp.float64(0.0)


@wp.kernel
def _sparse_local_repair_F(
    f: wp.array2d(dtype=wp.float64),
    U: wp.array2d(dtype=wp.float64),
    neighbor: wp.array2d(dtype=wp.int32),
    u: wp.array2d(dtype=wp.float64),
    dx: wp.float64,
    dy: wp.float64,
    dz: wp.float64,
    blend: wp.float64,
    lattice_speed: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
):
    p = wp.tid()
    beta = wp.max(wp.float64(0.0), wp.min(wp.float64(1.0), blend))
    o11 = U[3, p]
    o12 = U[4, p]
    o13 = U[5, p]
    o21 = U[6, p]
    o22 = U[7, p]
    o23 = U[8, p]
    o31 = U[9, p]
    o32 = U[10, p]
    o33 = U[11, p]
    t11 = wp.float64(1.0) + _sparse_local_du_axis(u, neighbor, 0, p, 0, 3, dx)
    t12 = _sparse_local_du_axis(u, neighbor, 0, p, 1, 4, dy)
    t13 = _sparse_local_du_axis(u, neighbor, 0, p, 2, 5, dz)
    t21 = _sparse_local_du_axis(u, neighbor, 1, p, 0, 3, dx)
    t22 = wp.float64(1.0) + _sparse_local_du_axis(u, neighbor, 1, p, 1, 4, dy)
    t23 = _sparse_local_du_axis(u, neighbor, 1, p, 2, 5, dz)
    t31 = _sparse_local_du_axis(u, neighbor, 2, p, 0, 3, dx)
    t32 = _sparse_local_du_axis(u, neighbor, 2, p, 1, 4, dy)
    t33 = wp.float64(1.0) + _sparse_local_du_axis(u, neighbor, 2, p, 2, 5, dz)
    n11 = o11 + beta * (t11 - o11)
    n12 = o12 + beta * (t12 - o12)
    n13 = o13 + beta * (t13 - o13)
    n21 = o21 + beta * (t21 - o21)
    n22 = o22 + beta * (t22 - o22)
    n23 = o23 + beta * (t23 - o23)
    n31 = o31 + beta * (t31 - o31)
    n32 = o32 + beta * (t32 - o32)
    n33 = o33 + beta * (t33 - o33)
    if _det3_wp(n11, n12, n13, n21, n22, n23, n31, n32, n33) <= j_floor:
        return
    # Exact equilibrium-moment delta with only two constitutive evaluations.
    # The velocity state is unchanged by this repair; hence rows 0--2 receive
    # only the Piola-flux delta and rows 3--11 only delta(F)/6.
    oldP = _piola_mat(
        o11, o12, o13, o21, o22, o23, o31, o32, o33, model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor
    )
    newP = _piola_mat(
        n11, n12, n13, n21, n22, n23, n31, n32, n33, model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor
    )
    flux_factor = wp.float64(0.5) / lattice_speed
    sixth = wp.float64(1.0) / wp.float64(6.0)
    for q in range(6):
        sx = wp.float64(_dx(q))
        sy = wp.float64(_dy(q))
        sz = wp.float64(_dz(q))
        base = q * 12
        f[base + 0, p] = f[base + 0, p] - flux_factor * (
            sx * (newP[0, 0] - oldP[0, 0])
            + sy * (newP[0, 1] - oldP[0, 1])
            + sz * (newP[0, 2] - oldP[0, 2])
        )
        f[base + 1, p] = f[base + 1, p] - flux_factor * (
            sx * (newP[1, 0] - oldP[1, 0])
            + sy * (newP[1, 1] - oldP[1, 1])
            + sz * (newP[1, 2] - oldP[1, 2])
        )
        f[base + 2, p] = f[base + 2, p] - flux_factor * (
            sx * (newP[2, 0] - oldP[2, 0])
            + sy * (newP[2, 1] - oldP[2, 1])
            + sz * (newP[2, 2] - oldP[2, 2])
        )
        f[base + 3, p] = f[base + 3, p] + sixth * (n11 - o11)
        f[base + 4, p] = f[base + 4, p] + sixth * (n12 - o12)
        f[base + 5, p] = f[base + 5, p] + sixth * (n13 - o13)
        f[base + 6, p] = f[base + 6, p] + sixth * (n21 - o21)
        f[base + 7, p] = f[base + 7, p] + sixth * (n22 - o22)
        f[base + 8, p] = f[base + 8, p] + sixth * (n23 - o23)
        f[base + 9, p] = f[base + 9, p] + sixth * (n31 - o31)
        f[base + 10, p] = f[base + 10, p] + sixth * (n32 - o32)
        f[base + 11, p] = f[base + 11, p] + sixth * (n33 - o33)
    U[3, p] = n11
    U[4, p] = n12
    U[5, p] = n13
    U[6, p] = n21
    U[7, p] = n22
    U[8, p] = n23
    U[9, p] = n31
    U[10, p] = n32
    U[11, p] = n33


@wp.kernel
def _sparse_characteristic_gather(
    U: wp.array2d(dtype=wp.float64),
    indices: wp.array2d(dtype=wp.int32),
    weights: wp.array3d(dtype=wp.float64),
    local_state: wp.array3d(dtype=wp.float64),
):
    a, ell = wp.tid()
    if indices[0, ell] < 0:
        return
    value = wp.float64(0.0)
    gx = wp.float64(0.0)
    gy = wp.float64(0.0)
    gz = wp.float64(0.0)
    for m in range(MAX_CHARACTERISTIC_SAMPLES):
        p = indices[m, ell]
        if p >= 0:
            u = U[a, p]
            value += weights[0, m, ell] * u
            gx += weights[1, m, ell] * u
            gy += weights[2, m, ell] * u
            gz += weights[3, m, ell] * u
    local_state[0, a, ell] = value
    local_state[1, a, ell] = gx
    local_state[2, a, ell] = gy
    local_state[3, a, ell] = gz


@wp.func
def _state_piola3(
    U: State12,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    p0: wp.float64,
    p1: wp.float64,
    p2: wp.float64,
    floor: wp.float64,
):
    return _piola_mat(
        U[3], U[4], U[5], U[6], U[7], U[8], U[9], U[10], U[11], model, lam, mu, p0, p1, p2, floor
    )


@wp.func
def _state_det3(U: State12):
    return (
        U[3] * (U[7] * U[11] - U[8] * U[10])
        - U[4] * (U[6] * U[11] - U[8] * U[9])
        + U[5] * (U[6] * U[10] - U[7] * U[9])
    )


@wp.func
def _state_flux_jvp3(
    axis: int,
    U: State12,
    W: State12,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    p0: wp.float64,
    p1: wp.float64,
    p2: wp.float64,
    floor: wp.float64,
):
    # Same centered constitutive directional derivative as the D2Q4 closure.
    scale = wp.float64(1.0)
    for a in range(12):
        scale = wp.max(scale, wp.abs(W[a]))
    eps = wp.float64(1.0e-6) / scale
    Pp = _state_piola3(U + eps * W, model, lam, mu, p0, p1, p2, floor)
    Pm = _state_piola3(U - eps * W, model, lam, mu, p0, p1, p2, floor)
    out = State12()
    for i in range(3):
        out[i] = -(Pp[i, axis] - Pm[i, axis]) / (wp.float64(2.0) * eps)
        out[3 + 3 * i + axis] = -W[i]
    return out


@wp.func
def _local_characteristic_population3(
    Ub: State12,
    Gx: State12,
    Gy: State12,
    Gz: State12,
    d: wp.vec3d,
    eta: wp.float64,
    dt: wp.float64,
    c: wp.float64,
    Bparams: wp.array(dtype=wp.float64),
    source_mode: int,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    p0: wp.float64,
    p1: wp.float64,
    p2: wp.float64,
    floor: wp.float64,
    omega: wp.float64,
):
    shift = eta * dt * c * (d[0] * Gx + d[1] * Gy + d[2] * Gz)
    chi = wp.float64(1.0)
    for guard in range(10):
        if _state_det3(Ub + chi * shift) > wp.float64(10.0) * floor:
            break
        chi = wp.float64(0.5) * chi
    Uc = Ub + chi * shift
    B = State12()
    for a in range(12):
        B[a] = _source_comp(a, source_mode, Bparams, Uc[0], Uc[1], Uc[2])
    w = (
        B
        - _state_flux_jvp3(0, Uc, Gx, model, lam, mu, p0, p1, p2, floor)
        - _state_flux_jvp3(1, Uc, Gy, model, lam, mu, p0, p1, p2, floor)
        - _state_flux_jvp3(2, Uc, Gz, model, lam, mu, p0, p1, p2, floor)
        + c * (d[0] * Gx + d[1] * Gy + d[2] * Gz)
    )
    axis = int(0)
    if wp.abs(d[1]) > wp.float64(0.5):
        axis = 1
    if wp.abs(d[2]) > wp.float64(0.5):
        axis = 2
    Aw = _state_flux_jvp3(axis, Uc, w, model, lam, mu, p0, p1, p2, floor)
    P = _state_piola3(Uc, model, lam, mu, p0, p1, p2, floor)
    flux = State12()
    for i in range(3):
        flux[i] = -P[i, axis]
        flux[3 + 3 * i + axis] = -Uc[i]
    return (
        Uc
        + wp.float64(3.0) * d[axis] * flux / c
        - dt / omega * (w + wp.float64(3.0) * d[axis] * Aw / c)
        + dt * (wp.float64(1.0) / omega - wp.float64(0.5)) * B
    ) / wp.float64(6.0)


@wp.kernel
def _sparse_apply_characteristic(
    U: wp.array2d(dtype=wp.float64),
    f1: wp.array2d(dtype=wp.float64),
    local_state: wp.array3d(dtype=wp.float64),
    lp: wp.array(dtype=wp.int32),
    lq: wp.array(dtype=wp.int32),
    leta: wp.array(dtype=wp.float64),
    lkind: wp.array2d(dtype=wp.int32),
    lbv: wp.array2d(dtype=wp.float64),
    lFb: wp.array2d(dtype=wp.float64),
    Bparams: wp.array(dtype=wp.float64),
    source_mode: int,
    dt: wp.float64,
    c: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    p0: wp.float64,
    p1: wp.float64,
    p2: wp.float64,
    floor: wp.float64,
    omega: wp.float64,
    invalid: wp.array(dtype=wp.int32),
):
    ell = wp.tid()
    if lkind[0, ell] != BC_NEUMANN and lkind[1, ell] != BC_NEUMANN and lkind[2, ell] != BC_NEUMANN:
        return
    p = lp[ell]
    qi = _opp(lq[ell])
    Ub = State12()
    Gx = State12()
    Gy = State12()
    Gz = State12()
    for a in range(12):
        Gx[a] = local_state[1, a, ell]
        Gy[a] = local_state[2, a, ell]
        Gz[a] = local_state[3, a, ell]
        if a < 3:
            Ub[a] = U[a, p]
            if lkind[a, ell] == BC_DIRICHLET:
                Ub[a] = lbv[a, ell]
        else:
            Ub[a] = lFb[a - 3, ell]
    d = wp.vec3d(wp.float64(_dx(qi)), wp.float64(_dy(qi)), wp.float64(_dz(qi)))
    if _state_det3(Ub) <= floor:
        wp.atomic_add(invalid, 0, 1)
    incoming = _local_characteristic_population3(
        Ub,
        Gx,
        Gy,
        Gz,
        d,
        leta[ell],
        dt,
        c,
        Bparams,
        source_mode,
        model,
        lam,
        mu,
        p0,
        p1,
        p2,
        floor,
        omega,
    )
    for a in range(12):
        row = a
        if a >= 3:
            row = (a - 3) // 3
        if lkind[row, ell] == BC_NEUMANN:
            f1[qi * 12 + a, p] = incoming[a]


@wp.kernel
def _sparse_prepare_links_gpu(
    U: wp.array2d(dtype=wp.float64),
    local_state: wp.array3d(dtype=wp.float64),
    u: wp.array2d(dtype=wp.float64),
    lp: wp.array(dtype=wp.int32),
    lup: wp.array(dtype=wp.int32),
    lq: wp.array(dtype=wp.int32),
    lbid: wp.array(dtype=wp.int32),
    leta: wp.array(dtype=wp.float64),
    lxb: wp.array(dtype=wp.float64),
    lyb: wp.array(dtype=wp.float64),
    lzb: wp.array(dtype=wp.float64),
    lnx: wp.array(dtype=wp.float64),
    lny: wp.array(dtype=wp.float64),
    lnz: wp.array(dtype=wp.float64),
    lsq_p: wp.array2d(dtype=wp.int32),
    lsq_wt1: wp.array2d(dtype=wp.float64),
    lsq_wt2: wp.array2d(dtype=wp.float64),
    lsq_valid: wp.array(dtype=wp.int32),
    kind_by_id: wp.array(dtype=wp.int32),
    component_kind_by_id: wp.array2d(dtype=wp.int32),
    mode_by_id: wp.array(dtype=wp.int32),
    params_by_id: wp.array2d(dtype=wp.float64),
    dynamic_traction_enabled: wp.array(dtype=wp.int32),
    dynamic_traction: wp.array2d(dtype=wp.float64),
    lkind: wp.array(dtype=wp.int32),
    lcomponent_kind: wp.array2d(dtype=wp.int32),
    lbv: wp.array2d(dtype=wp.float64),
    lFb: wp.array2d(dtype=wp.float64),
    lPb: wp.array2d(dtype=wp.float64),
    nlinks: int,
    t: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
    recon: int,
    normal_jacobian: int,
):
    ell = wp.tid()
    if ell >= nlinks:
        return
    bid = lbid[ell]
    kind = kind_by_id[bid]
    mode = mode_by_id[bid]
    kind0 = component_kind_by_id[0, bid]
    kind1 = component_kind_by_id[1, bid]
    kind2 = component_kind_by_id[2, bid]
    x = lxb[ell]
    y = lyb[ell]
    z = lzb[ell]
    nx = lnx[ell]
    ny = lny[ell]
    nz = lnz[ell]
    tx = _curved_value_comp(mode, params_by_id, bid, 0, x, y, z, nx, ny, nz, t, j_floor)
    ty = _curved_value_comp(mode, params_by_id, bid, 1, x, y, z, nx, ny, nz, t, j_floor)
    tz = _curved_value_comp(mode, params_by_id, bid, 2, x, y, z, nx, ny, nz, t, j_floor)
    if dynamic_traction_enabled[ell] != 0:
        tx = dynamic_traction[0, ell]
        ty = dynamic_traction[1, ell]
        tz = dynamic_traction[2, ell]
    lkind[ell] = kind
    lcomponent_kind[0, ell] = kind0
    lcomponent_kind[1, ell] = kind1
    lcomponent_kind[2, ell] = kind2
    lbv[0, ell] = tx
    lbv[1, ell] = ty
    lbv[2, ell] = tz
    Fb = wp.mat33d(
        wp.float64(1.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(1.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(1.0),
    )
    Pb = wp.mat33d(
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
    )
    if kind0 == BC_NEUMANN or kind1 == BC_NEUMANN or kind2 == BC_NEUMANN:
        p = lp[ell]
        F11 = U[3, p]
        F12 = U[4, p]
        F13 = U[5, p]
        F21 = U[6, p]
        F22 = U[7, p]
        F23 = U[8, p]
        F31 = U[9, p]
        F32 = U[10, p]
        F33 = U[11, p]
        if recon == BOUNDARY_RECON_COMPAT_BFL or recon == BOUNDARY_RECON_LOCAL_F_BFL:
            upid = lup[ell]
            if upid >= 0:
                eta = leta[ell]
                F11 = U[3, p] + eta * (U[3, p] - U[3, upid])
                F12 = U[4, p] + eta * (U[4, p] - U[4, upid])
                F13 = U[5, p] + eta * (U[5, p] - U[5, upid])
                F21 = U[6, p] + eta * (U[6, p] - U[6, upid])
                F22 = U[7, p] + eta * (U[7, p] - U[7, upid])
                F23 = U[8, p] + eta * (U[8, p] - U[8, upid])
                F31 = U[9, p] + eta * (U[9, p] - U[9, upid])
                F32 = U[10, p] + eta * (U[10, p] - U[10, upid])
                F33 = U[11, p] + eta * (U[11, p] - U[11, upid])
            if recon == BOUNDARY_RECON_COMPAT_BFL and lsq_valid[ell] != 0:
                n = _normalize3(wp.vec3d(nx, ny, nz))
                ref = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
                if wp.abs(n[0]) >= wp.float64(0.75):
                    ref = wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0))
                t1 = _normalize3(_cross3(n, ref))
                t2 = _cross3(n, t1)
                du10 = wp.float64(0.0)
                du11 = wp.float64(0.0)
                du12 = wp.float64(0.0)
                du20 = wp.float64(0.0)
                du21 = wp.float64(0.0)
                du22 = wp.float64(0.0)
                wt_abs = wp.float64(0.0)
                for m in range(MAX_LSQ_STENCIL):
                    pp = lsq_p[ell, m]
                    if pp >= 0:
                        w1 = lsq_wt1[ell, m]
                        w2 = lsq_wt2[ell, m]
                        du10 += w1 * u[0, pp]
                        du11 += w1 * u[1, pp]
                        du12 += w1 * u[2, pp]
                        du20 += w2 * u[0, pp]
                        du21 += w2 * u[1, pp]
                        du22 += w2 * u[2, pp]
                        wt_abs += wp.abs(w1) + wp.abs(w2)
                if wt_abs > wp.float64(0.0):
                    h0 = F11 * n[0] + F12 * n[1] + F13 * n[2]
                    h1 = F21 * n[0] + F22 * n[1] + F23 * n[2]
                    h2 = F31 * n[0] + F32 * n[1] + F33 * n[2]
                    g10 = t1[0] + du10
                    g11 = t1[1] + du11
                    g12 = t1[2] + du12
                    g20 = t2[0] + du20
                    g21 = t2[1] + du21
                    g22 = t2[2] + du22
                    F11 = h0 * n[0] + g10 * t1[0] + g20 * t2[0]
                    F12 = h0 * n[1] + g10 * t1[1] + g20 * t2[1]
                    F13 = h0 * n[2] + g10 * t1[2] + g20 * t2[2]
                    F21 = h1 * n[0] + g11 * t1[0] + g21 * t2[0]
                    F22 = h1 * n[1] + g11 * t1[1] + g21 * t2[1]
                    F23 = h1 * n[2] + g11 * t1[2] + g21 * t2[2]
                    F31 = h2 * n[0] + g12 * t1[0] + g22 * t2[0]
                    F32 = h2 * n[1] + g12 * t1[1] + g22 * t2[1]
                    F33 = h2 * n[2] + g12 * t1[2] + g22 * t2[2]
        if recon == BOUNDARY_RECON_LOCAL_F:
            F11 = local_state[0, 3, ell]
            F12 = local_state[0, 4, ell]
            F13 = local_state[0, 5, ell]
            F21 = local_state[0, 6, ell]
            F22 = local_state[0, 7, ell]
            F23 = local_state[0, 8, ell]
            F31 = local_state[0, 9, ell]
            F32 = local_state[0, 10, ell]
            F33 = local_state[0, 11, ell]
        Fb = wp.mat33d(F11, F12, F13, F21, F22, F23, F31, F32, F33)
        if kind0 == BC_NEUMANN and kind1 == BC_NEUMANN and kind2 == BC_NEUMANN:
            Fb = _solve_Fb_arbitrary_normal(
                F11,
                F12,
                F13,
                F21,
                F22,
                F23,
                F31,
                F32,
                F33,
                nx,
                ny,
                nz,
                tx,
                ty,
                tz,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
                normal_jacobian,
            )
        elif kind0 == BC_NEUMANN or kind1 == BC_NEUMANN or kind2 == BC_NEUMANN:
            Fb = _solve_Fb_componentwise_normal(
                F11,
                F12,
                F13,
                F21,
                F22,
                F23,
                F31,
                F32,
                F33,
                nx,
                ny,
                nz,
                tx,
                ty,
                tz,
                kind0,
                kind1,
                kind2,
                model,
                lam,
                mu,
                mat_p0,
                mat_p1,
                mat_p2,
                j_floor,
                normal_jacobian,
            )
        Pb = _piola_mat(
            Fb[0, 0],
            Fb[0, 1],
            Fb[0, 2],
            Fb[1, 0],
            Fb[1, 1],
            Fb[1, 2],
            Fb[2, 0],
            Fb[2, 1],
            Fb[2, 2],
            model,
            lam,
            mu,
            mat_p0,
            mat_p1,
            mat_p2,
            j_floor,
        )
    lFb[0, ell] = Fb[0, 0]
    lFb[1, ell] = Fb[0, 1]
    lFb[2, ell] = Fb[0, 2]
    lFb[3, ell] = Fb[1, 0]
    lFb[4, ell] = Fb[1, 1]
    lFb[5, ell] = Fb[1, 2]
    lFb[6, ell] = Fb[2, 0]
    lFb[7, ell] = Fb[2, 1]
    lFb[8, ell] = Fb[2, 2]
    lPb[0, ell] = Pb[0, 0]
    lPb[1, ell] = Pb[0, 1]
    lPb[2, ell] = Pb[0, 2]
    lPb[3, ell] = Pb[1, 0]
    lPb[4, ell] = Pb[1, 1]
    lPb[5, ell] = Pb[1, 2]
    lPb[6, ell] = Pb[2, 0]
    lPb[7, ell] = Pb[2, 1]
    lPb[8, ell] = Pb[2, 2]


@wp.kernel
def _sparse_apply_links(
    fpost: wp.array2d(dtype=wp.float64),
    f1: wp.array2d(dtype=wp.float64),
    lp: wp.array(dtype=wp.int32),
    lup: wp.array(dtype=wp.int32),
    lq: wp.array(dtype=wp.int32),
    leta: wp.array(dtype=wp.float64),
    lkind: wp.array(dtype=wp.int32),
    lcomponent_kind: wp.array2d(dtype=wp.int32),
    lbv: wp.array2d(dtype=wp.float64),
    lFb: wp.array2d(dtype=wp.float64),
    lPb: wp.array2d(dtype=wp.float64),
    nlinks: int,
    c: wp.float64,
    recon: int,
):
    ell = wp.tid()
    if ell >= nlinks:
        return
    p = lp[ell]
    qo = lq[ell]
    qi = _opp(qo)
    eta = leta[ell]
    if recon == BOUNDARY_RECON_HALF_WAY:
        eta = wp.float64(0.5)
    dxin = wp.float64(_dx(qi))
    dyin = wp.float64(_dy(qi))
    dzin = wp.float64(_dz(qi))
    upid = lup[ell]
    v0 = lbv[0, ell]
    v1 = lbv[1, ell]
    v2 = lbv[2, ell]
    for a in range(12):
        row = a
        if a >= 3:
            row = (a - 3) // 3
        kind = lcomponent_kind[row, ell]
        D = wp.float64(1.0)
        S = wp.float64(0.0)
        if kind == BC_DIRICHLET:
            if a == 0:
                D = wp.float64(-1.0)
                S = v0 / wp.float64(3.0)
            elif a == 1:
                D = wp.float64(-1.0)
                S = v1 / wp.float64(3.0)
            elif a == 2:
                D = wp.float64(-1.0)
                S = v2 / wp.float64(3.0)
            else:
                row = (a - 3) // 3
                col = (a - 3) - row * 3
                vi = v0
                if row == 1:
                    vi = v1
                elif row == 2:
                    vi = v2
                dB = dxin
                if col == 1:
                    dB = dyin
                elif col == 2:
                    dB = dzin
                S = -dB * vi / c
        else:
            if a == 0:
                S = -(dxin * lPb[0, ell] + dyin * lPb[1, ell] + dzin * lPb[2, ell]) / c
            elif a == 1:
                S = -(dxin * lPb[3, ell] + dyin * lPb[4, ell] + dzin * lPb[5, ell]) / c
            elif a == 2:
                S = -(dxin * lPb[6, ell] + dyin * lPb[7, ell] + dzin * lPb[8, ell]) / c
            else:
                D = wp.float64(-1.0)
                S = lFb[a - 3, ell] / wp.float64(3.0)
        fo = fpost[qo * 12 + a, p]
        fi = fpost[qi * 12 + a, p]
        if eta >= wp.float64(0.5):
            alpha = wp.float64(1.0) / (wp.float64(2.0) * eta)
            val = alpha * (D * fo + S) + (wp.float64(1.0) - alpha) * fi
        else:
            up = fo
            if upid >= 0:
                up = fpost[qo * 12 + a, upid]
            val = (
                D * (wp.float64(2.0) * eta * fo + (wp.float64(1.0) - wp.float64(2.0) * eta) * up)
                + S
            )
        f1[qi * 12 + a, p] = val


class WarpCurvedBoundaryLBM3D:
    def __init__(
        self,
        geometry: CurvedBoundaryGeometry3D,
        *,
        material: WarpHyperelasticMaterial3D,
        boundary: CurvedBoundarySpec3D,
        lattice_speed: float = 5.0,
        collision_omega: float = 1.8,
        boundary_reconstruction: BoundaryReconstruction = "bfl",
        device: str | None = None,
        check_linearized_cfl: bool = True,
        jacobian_floor: float = 1e-12,
        source_mode: SourceName = "none",
        source_vector: tuple[float, ...] = (0.0,) * NCOMP,
        damping_gamma: float = 0.0,
        normal_jacobian: Literal["finite_difference", "acoustic"] | None = None,
        neumann_solver: Literal["newton", "energy"] | None = None,
    ):
        self.geometry = geometry
        self.nx = geometry.nx
        self.ny = geometry.ny
        self.nz = geometry.nz
        self.length_x = geometry.length_x
        self.length_y = geometry.length_y
        self.length_z = geometry.length_z
        self.dx = geometry.dx
        self.dy = geometry.dy
        self.dz = geometry.dz
        self.material = material
        self.boundary = boundary
        self.lattice_speed = float(lattice_speed)
        self.collision_omega = float(collision_omega)
        self.boundary_reconstruction = _recon_id(boundary_reconstruction)
        self.boundary_reconstruction_name = _recon_name(self.boundary_reconstruction)
        self.dt = self.dx / self.lattice_speed
        self.time = 0.0
        self.steps = 0
        self.device = device or default_device()
        self.shape = (self.nx, self.ny, self.nz)
        self.jacobian_floor = float(jacobian_floor)
        self.normal_jacobian = _normal_jacobian_for_material(material, normal_jacobian)
        self.normal_jacobian_name = normal_jacobian_name(self.normal_jacobian)
        self.neumann_solver_name = neumann_solver_name(neumann_solver, material)
        self._normal_solver_mode = (
            NORMAL_SOLVER_ENERGY if self.neumann_solver_name == "energy" else self.normal_jacobian
        )
        if check_linearized_cfl and material.linearized_stability_ratio(self.lattice_speed) >= 1.0:
            raise ValueError("linearized D3Q6 stability ratio must be below one")
        self.source_mode = source_id(source_mode)
        self.source_params = np.zeros(NCOMP)
        if self.source_mode == SOURCE_CONSTANT:
            for i, v in enumerate(tuple(source_vector)[:NCOMP]):
                self.source_params[i] = float(v)
        elif self.source_mode == SOURCE_DAMPING:
            self.source_params[0] = float(damping_gamma)
        xs = (np.arange(self.nx) + 0.5) * self.dx
        ys = (np.arange(self.ny) + 0.5) * self.dy
        zs = (np.arange(self.nz) + 0.5) * self.dz
        self.x, self.y, self.z = np.meshgrid(xs, ys, zs, indexing="ij")
        self.active_np = np.ascontiguousarray(geometry.active.astype(np.int32))
        self.active = _as_warp(self.active_np, dtype=wp.int32, device=self.device)
        self.link_i = _as_warp(geometry.link_i, dtype=wp.int32, device=self.device)
        self.link_j = _as_warp(geometry.link_j, dtype=wp.int32, device=self.device)
        self.link_k = _as_warp(geometry.link_k, dtype=wp.int32, device=self.device)
        self.link_q = _as_warp(geometry.link_q, dtype=wp.int32, device=self.device)
        self.link_boundary_id = _as_warp(
            geometry.link_boundary_id, dtype=wp.int32, device=self.device
        )
        self.link_eta = _as_warp(geometry.link_eta, dtype=wp.float64, device=self.device)
        self.link_xb = _as_warp(geometry.link_xb, dtype=wp.float64, device=self.device)
        self.link_yb = _as_warp(geometry.link_yb, dtype=wp.float64, device=self.device)
        self.link_zb = _as_warp(geometry.link_zb, dtype=wp.float64, device=self.device)
        self.link_nx = _as_warp(geometry.link_nx, dtype=wp.float64, device=self.device)
        self.link_ny = _as_warp(geometry.link_ny, dtype=wp.float64, device=self.device)
        self.link_nz = _as_warp(geometry.link_nz, dtype=wp.float64, device=self.device)
        self.kind_by_id, self.mode_by_id, self.params_by_id = boundary.tables(MAX_BOUNDARY_IDS)
        self.component_kind_by_id = boundary.component_kind_table(MAX_BOUNDARY_IDS)
        self.kind_by_id_wp = _as_warp(self.kind_by_id, dtype=wp.int32, device=self.device)
        self.component_kind_by_id_wp = _as_warp(
            self.component_kind_by_id, dtype=wp.int32, device=self.device
        )
        self.mode_by_id_wp = _as_warp(self.mode_by_id, dtype=wp.int32, device=self.device)
        self.params_by_id_wp = _as_warp(self.params_by_id, dtype=wp.float64, device=self.device)
        nl = geometry.n_links
        self.link_kind_np = np.zeros(nl, np.int32)
        self.link_component_kind_np = np.zeros((3, nl), np.int32)
        self.link_bv_np = np.zeros((3, nl))
        self.link_Fb_np = np.zeros((9, nl))
        self.link_Pb_np = np.zeros((9, nl))
        self.link_kind = _as_warp(self.link_kind_np, dtype=wp.int32, device=self.device)
        self.link_component_kind = _as_warp(
            self.link_component_kind_np, dtype=wp.int32, device=self.device
        )
        self.link_bv = _as_warp(self.link_bv_np, dtype=wp.float64, device=self.device)
        self.link_Fb = _as_warp(self.link_Fb_np, dtype=wp.float64, device=self.device)
        self.link_Pb = _as_warp(self.link_Pb_np, dtype=wp.float64, device=self.device)
        self.f0 = wp.zeros((FQ, self.nx, self.ny, self.nz), dtype=wp.float64, device=self.device)
        self.f1 = wp.zeros_like(self.f0)
        self.fpost = wp.zeros_like(self.f0)
        self.u = wp.zeros((3, self.nx, self.ny, self.nz), dtype=wp.float64, device=self.device)
        self.u_star = wp.zeros_like(self.u)
        self.U = wp.zeros((NCOMP, self.nx, self.ny, self.nz), dtype=wp.float64, device=self.device)
        self.P = wp.zeros((9, self.nx, self.ny, self.nz), dtype=wp.float64, device=self.device)
        self.sigma = wp.zeros_like(self.P)
        self.invalid = wp.zeros((1,), dtype=wp.int32, device=self.device)
        self.source_params_wp = _as_warp(self.source_params, dtype=wp.float64, device=self.device)
        self.initialize_identity()

    def _fill(self):
        wp.launch(_fill_inactive, dim=self.shape, inputs=[self.f0, self.active], device=self.device)

    def initialize_identity(self, *, init_order: int = 1):
        U = np.zeros((NCOMP, self.nx, self.ny, self.nz))
        U[3] = 1
        U[7] = 1
        U[11] = 1
        self.initialize_from_numpy(
            U=U, displacement=np.zeros((3, self.nx, self.ny, self.nz)), init_order=init_order
        )

    def initialize_from_numpy(
        self,
        *,
        U: np.ndarray,
        displacement: np.ndarray | None = None,
        init_order: int = 1,
        apply_initial_boundaries: bool = True,
        initial_boundary_time: float = 0.0,
    ):
        U = np.asarray(U, float)
        disp = (
            np.zeros((3, self.nx, self.ny, self.nz))
            if displacement is None
            else np.asarray(displacement, float)
        )
        mask = self.active_np.astype(bool)
        Uw = U.copy()
        Uw[:, ~mask] = 0
        Uw[3, ~mask] = 1
        Uw[7, ~mask] = 1
        Uw[11, ~mask] = 1
        if init_order == 1:
            f = _equilibrium_np(Uw, self.material, self.lattice_speed)
            f -= (
                0.5
                * self.dt
                / float(Q)
                * _source_np(Uw, self.source_mode, self.source_params)[None, ...]
            )
        else:
            f = _second_order_populations_np(
                Uw,
                self.material,
                self.lattice_speed,
                dx=self.dx,
                dy=self.dy,
                dz=self.dz,
                dt=self.dt,
                periodic_x=False,
                periodic_y=False,
                periodic_z=False,
                source_mode=self.source_mode,
                source_params=self.source_params,
            )
        disp = disp.copy()
        disp[:, ~mask] = 0
        us0 = disp - 0.5 * self.dt * Uw[:3]
        us0[:, ~mask] = 0
        wp.copy(self.f0, _as_warp(_flatten_pop(f), dtype=wp.float64, device=self.device))
        self._fill()
        wp.copy(
            self.f1, wp.zeros((FQ, self.nx, self.ny, self.nz), dtype=wp.float64, device=self.device)
        )
        wp.copy(self.fpost, wp.zeros_like(self.f1))
        wp.copy(self.u_star, _as_warp(us0, dtype=wp.float64, device=self.device))
        self.time = 0
        self.steps = 0
        self.refresh_current()
        if apply_initial_boundaries:
            self.apply_boundary_reconstruction(initial_boundary_time, in_place=True)
            self.refresh_current()
            wp.launch(
                _set_ustar,
                dim=self.shape,
                inputs=[
                    _as_warp(disp, dtype=wp.float64, device=self.device),
                    self.U,
                    self.u_star,
                    self.active,
                    self.dt,
                ],
                device=self.device,
            )
            self.refresh_current()

    def numpy_field(self, name):
        wp.synchronize_device(self.device)
        return np.asarray(getattr(self, name).numpy())

    def _prepare_links(self, t: float):
        wp.launch(
            _prepare_links_gpu,
            dim=self.geometry.n_links,
            inputs=[
                self.U,
                self.link_i,
                self.link_j,
                self.link_k,
                self.link_boundary_id,
                self.link_xb,
                self.link_yb,
                self.link_zb,
                self.link_nx,
                self.link_ny,
                self.link_nz,
                self.kind_by_id_wp,
                self.component_kind_by_id_wp,
                self.mode_by_id_wp,
                self.params_by_id_wp,
                self.link_kind,
                self.link_component_kind,
                self.link_bv,
                self.link_Fb,
                self.link_Pb,
                self.geometry.n_links,
                float(t),
                self.material.id,
                self.material.lam,
                self.material.mu,
                self.material.param0,
                self.material.param1,
                self.material.param2,
                self.jacobian_floor,
                self._normal_solver_mode,
            ],
            device=self.device,
        )

    def apply_boundary_reconstruction(self, t: float, *, in_place: bool = False):
        if self.geometry.n_links == 0:
            return
        self._prepare_links(t)
        if in_place:
            wp.copy(self.fpost, self.f0)
        src = self.fpost
        tgt = self.f0 if in_place else self.f1
        wp.launch(
            _apply_links,
            dim=self.geometry.n_links,
            inputs=[
                src,
                tgt,
                self.active,
                self.link_i,
                self.link_j,
                self.link_k,
                self.link_q,
                self.link_eta,
                self.link_kind,
                self.link_component_kind,
                self.link_bv,
                self.link_Fb,
                self.link_Pb,
                self.geometry.n_links,
                self.nx,
                self.ny,
                self.nz,
                self.lattice_speed,
                self.boundary_reconstruction,
            ],
            device=self.device,
        )

    def step(self):
        self._fill()
        wp.launch(_reset_invalid_kernel, dim=1, inputs=[self.invalid], device=self.device)
        wp.launch(
            _collide_kernel,
            dim=self.shape,
            inputs=[
                self.f0,
                self.fpost,
                self.u,
                self.u_star,
                self.U,
                self.P,
                self.sigma,
                self.invalid,
                self.source_params_wp,
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
            ],
            device=self.device,
        )
        wp.launch(
            _clear_kernel, dim=(FQ, self.nx, self.ny, self.nz), inputs=[self.f1], device=self.device
        )
        wp.launch(
            _stream_masked,
            dim=self.shape,
            inputs=[self.fpost, self.f1, self.active, self.nx, self.ny, self.nz],
            device=self.device,
        )
        self.apply_boundary_reconstruction(self.time + 0.5 * self.dt)
        self.f0, self.f1 = self.f1, self.f0
        self._fill()
        self.time += self.dt
        self.steps += 1

    def refresh_current(self):
        self._fill()
        wp.launch(_reset_invalid_kernel, dim=1, inputs=[self.invalid], device=self.device)
        wp.launch(
            _refresh_kernel,
            dim=self.shape,
            inputs=[
                self.f0,
                self.u,
                self.u_star,
                self.U,
                self.P,
                self.sigma,
                self.invalid,
                self.source_params_wp,
                self.dt,
                self.material.id,
                self.material.lam,
                self.material.mu,
                self.material.param0,
                self.material.param1,
                self.material.param2,
                self.jacobian_floor,
                self.source_mode,
            ],
            device=self.device,
        )

    def run_until(self, t_end: float):
        while self.time < float(t_end) - 0.5 * self.dt:
            self.step()
        self.refresh_current()
        wp.synchronize_device(self.device)

    def active_mask(self):
        return self.active_np.astype(bool)

    def system_state(self):
        return self.numpy_field("U")

    def displacement(self):
        return self.numpy_field("u")

    def deformation_gradient(self):
        return _state_to_F_np(self.system_state())

    def deformation_jacobian(self):
        return _det3_np(self.deformation_gradient())

    def invalid_state(self):
        wp.synchronize_device(self.device)
        U = self.system_state()
        m = self.active_mask()
        J = self.deformation_jacobian()
        return (
            int(np.asarray(self.invalid.numpy())[0]) != 0
            or not np.isfinite(U[:, m]).all()
            or not np.all(J[m] > 0)
        )

    def summary(self):
        self.refresh_current()
        m = self.active_mask()
        U = self.system_state()
        u = self.displacement()
        J = self.deformation_jacobian()
        return {
            "device": self.device,
            "geometry": self.geometry.label,
            "boundary_label": self.boundary.label,
            "model": self.material.model,
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "n_active": int(np.count_nonzero(m)),
            "n_cut_links": int(self.geometry.n_links),
            "steps": int(self.steps),
            "time": float(self.time),
            "dx": float(self.dx),
            "dt": float(self.dt),
            "lattice_speed": float(self.lattice_speed),
            "collision_omega": float(self.collision_omega),
            "boundary_reconstruction": self.boundary_reconstruction_name,
            "normal_jacobian": self.normal_jacobian_name,
            "neumann_solver": self.neumann_solver_name,
            "max_abs_u": float(np.max(np.sqrt(np.sum(u[:, m] * u[:, m], axis=0))))
            if np.any(m)
            else 0.0,
            "max_abs_v": float(np.max(np.sqrt(U[0, m] ** 2 + U[1, m] ** 2 + U[2, m] ** 2)))
            if np.any(m)
            else 0.0,
            "min_J": float(np.min(J[m])) if np.any(m) else 0.0,
            "max_J": float(np.max(J[m])) if np.any(m) else 0.0,
            "finite": bool(np.isfinite(U[:, m]).all() and np.all(J[m] > 0)) if np.any(m) else False,
        }

    def save_npz(self, path):
        self.refresh_current()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            x=self.x,
            y=self.y,
            z=self.z,
            active=self.active_np,
            u=self.displacement(),
            U=self.system_state(),
            F=self.deformation_gradient(),
            P=self.numpy_field("P"),
            sigma=self.numpy_field("sigma"),
            f=_unflatten_pop(self.numpy_field("f0")),
            time=np.array(self.time),
            metadata=np.array(json.dumps(self.summary())),
        )


WarpDenseCurvedBoundaryLBM3D = WarpCurvedBoundaryLBM3D


def _sparse_tables_from_geometry(geometry: CurvedBoundaryGeometry3D) -> dict[str, np.ndarray]:
    active_bool = np.asarray(geometry.active, dtype=bool)
    ai, aj, ak = np.nonzero(active_bool)
    ai = np.ascontiguousarray(ai.astype(np.int32))
    aj = np.ascontiguousarray(aj.astype(np.int32))
    ak = np.ascontiguousarray(ak.astype(np.int32))
    n_active = int(ai.size)
    active_id = np.full(active_bool.shape, -1, dtype=np.int32)
    active_id[ai, aj, ak] = np.arange(n_active, dtype=np.int32)
    neighbor = np.full((Q, n_active), -1, dtype=np.int32)
    nx, ny, nz = geometry.nx, geometry.ny, geometry.nz
    for q, (di, dj, dk) in enumerate(DIRS3):
        ii = ai + int(di)
        jj = aj + int(dj)
        kk = ak + int(dk)
        valid = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny) & (kk >= 0) & (kk < nz)
        ids = np.full(n_active, -1, dtype=np.int32)
        vv = np.where(valid)[0]
        if vv.size:
            ids[vv] = active_id[ii[vv], jj[vv], kk[vv]]
        neighbor[q] = ids
    link_p = active_id[geometry.link_i, geometry.link_j, geometry.link_k].astype(np.int32)
    link_upstream = np.full(geometry.n_links, -1, dtype=np.int32)
    for ell in range(geometry.n_links):
        q = int(geometry.link_q[ell])
        di, dj, dk = DIRS3[q]
        ii = int(geometry.link_i[ell]) - int(di)
        jj = int(geometry.link_j[ell]) - int(dj)
        kk = int(geometry.link_k[ell]) - int(dk)
        if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
            link_upstream[ell] = active_id[ii, jj, kk]
    return {
        "active_i": ai,
        "active_j": aj,
        "active_k": ak,
        "active_id": np.ascontiguousarray(active_id),
        "neighbor": np.ascontiguousarray(neighbor),
        "link_p": np.ascontiguousarray(link_p),
        "link_upstream": np.ascontiguousarray(link_upstream),
    }


def _build_sparse_tangent_lsq_weights(
    geometry: CurvedBoundaryGeometry3D,
    tables: dict[str, np.ndarray],
    *,
    max_stencil: int = MAX_LSQ_STENCIL,
) -> dict[str, np.ndarray]:
    """Build sparse active-node weights for two surface-tangent derivatives."""

    n_links = int(geometry.n_links)
    out_p = np.full((n_links, max_stencil), -1, dtype=np.int32)
    wt1 = np.zeros((n_links, max_stencil), dtype=np.float64)
    wt2 = np.zeros((n_links, max_stencil), dtype=np.float64)
    valid = np.zeros(n_links, dtype=np.int32)
    if n_links == 0:
        return {"lsq_p": out_p, "lsq_wt1": wt1, "lsq_wt2": wt2, "lsq_valid": valid}

    active = np.asarray(geometry.active, dtype=bool)
    active_id = np.asarray(tables["active_id"], dtype=np.int32)
    nx, ny, nz = geometry.nx, geometry.ny, geometry.nz
    dx, dy, dz = float(geometry.dx), float(geometry.dy), float(geometry.dz)
    xs = (np.arange(nx, dtype=np.float64) + 0.5) * dx
    ys = (np.arange(ny, dtype=np.float64) + 0.5) * dy
    zs = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    eye10 = np.eye(10, dtype=np.float64)

    def try_radius(ell: int, radius_cells: int):
        xb = float(geometry.link_xb[ell])
        yb = float(geometry.link_yb[ell])
        zb = float(geometry.link_zb[ell])
        nvec = np.array(
            [geometry.link_nx[ell], geometry.link_ny[ell], geometry.link_nz[ell]], dtype=np.float64
        )
        nrm = max(float(np.linalg.norm(nvec)), np.finfo(float).eps)
        nvec = nvec / nrm
        t1, t2 = _basis_np(nvec)
        i0 = int(geometry.link_i[ell])
        j0 = int(geometry.link_j[ell])
        k0 = int(geometry.link_k[ell])
        cand: list[tuple[float, int, float, float, float]] = []
        r2max = (float(radius_cells) + 0.25) ** 2
        for ii in range(max(0, i0 - radius_cells), min(nx, i0 + radius_cells + 1)):
            for jj in range(max(0, j0 - radius_cells), min(ny, j0 + radius_cells + 1)):
                for kk in range(max(0, k0 - radius_cells), min(nz, k0 + radius_cells + 1)):
                    if not active[ii, jj, kk]:
                        continue
                    p = int(active_id[ii, jj, kk])
                    if p < 0:
                        continue
                    rx = (float(xs[ii]) - xb) / dx
                    ry = (float(ys[jj]) - yb) / dy
                    rz = (float(zs[kk]) - zb) / dz
                    r2 = rx * rx + ry * ry + rz * rz
                    if r2 <= r2max:
                        cand.append((r2, p, rx, ry, rz))
        cand.sort(key=lambda item: item[0])
        if len(cand) < 10:
            return None
        cand = cand[:max_stencil]
        k = len(cand)
        A = np.empty((k, 10), dtype=np.float64)
        wd = np.empty(k, dtype=np.float64)
        for m, (r2, _p, rx, ry, rz) in enumerate(cand):
            A[m] = (
                1.0,
                rx,
                ry,
                rz,
                0.5 * rx * rx,
                rx * ry,
                rx * rz,
                0.5 * ry * ry,
                ry * rz,
                0.5 * rz * rz,
            )
            wd[m] = math.exp(-0.5 * r2 / (1.75 * 1.75)) + 1.0e-4
        if np.linalg.matrix_rank(A, tol=1.0e-10) < 10:
            return None
        Aw = A * wd[:, None]
        M = A.T @ Aw
        cond = np.linalg.cond(M)
        if not np.isfinite(cond) or cond > 1.0e9:
            return None
        try:
            coeff_to_values = np.linalg.solve(M + 1.0e-12 * eye10, A.T * wd[None, :])
        except np.linalg.LinAlgError:
            return None
        target1 = np.array(
            [0.0, t1[0] / dx, t1[1] / dy, t1[2] / dz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        target2 = np.array(
            [0.0, t2[0] / dx, t2[1] / dy, t2[2] / dz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        w1 = target1 @ coeff_to_values
        w2 = target2 @ coeff_to_values
        h = max(dx, dy, dz)
        if (not np.all(np.isfinite(w1))) or (not np.all(np.isfinite(w2))):
            return None
        if abs(float(np.sum(w1))) * h > 1.0e-6 or abs(float(np.sum(w2))) * h > 1.0e-6:
            return None
        if float(np.sum(np.abs(w1))) * h > 12.0 or float(np.sum(np.abs(w2))) * h > 12.0:
            return None
        return cand, w1, w2

    for ell in range(n_links):
        fit = None
        for radius in (3, 4, 5):
            fit = try_radius(ell, radius)
            if fit is not None:
                break
        if fit is None:
            continue
        cand, w1, w2 = fit
        for m, (_r2, p, _rx, _ry, _rz) in enumerate(cand):
            out_p[ell, m] = int(p)
            wt1[ell, m] = float(w1[m])
            wt2[ell, m] = float(w2[m])
        valid[ell] = 1
    return {
        "lsq_p": np.ascontiguousarray(out_p),
        "lsq_wt1": np.ascontiguousarray(wt1),
        "lsq_wt2": np.ascontiguousarray(wt2),
        "lsq_valid": np.ascontiguousarray(valid),
    }


class WarpSparseCurvedBoundaryLBM3D:
    """Active-node sparse curved 3-D solver.

    Runtime storage is O(n_active) for populations and fields.  Dense arrays are
    reconstructed only for compatibility methods such as ``system_state``.
    """

    def __init__(
        self,
        geometry: CurvedBoundaryGeometry3D,
        *,
        material: WarpHyperelasticMaterial3D,
        boundary: CurvedBoundarySpec3D,
        lattice_speed: float = 5.0,
        collision_omega: float = 1.8,
        boundary_reconstruction: BoundaryReconstruction = "bfl",
        device: str | None = None,
        check_linearized_cfl: bool = True,
        jacobian_floor: float = 1e-12,
        source_mode: SourceName = "none",
        source_vector: tuple[float, ...] = (0.0,) * NCOMP,
        damping_gamma: float = 0.0,
        compatibility_projection_interval: int = 0,
        compatibility_projection_iterations: int = 160,
        compatibility_projection_relax: float = 0.85,
        compatibility_projection_weight: float = 20.0,
        compatibility_projection_repair_F: bool = False,
        compatibility_projection_repair_F_band_width: int = -1,
        compatibility_projection_repair_F_blend: float = 1.0,
        compatibility_projection_boundary_id: int = -1,
        local_displacement_interval: int = 0,
        local_displacement_sweeps: int = 1,
        local_displacement_relax: float = 0.85,
        local_displacement_boundary_weight: float = 20.0,
        local_displacement_boundary_id: int = -1,
        local_compatibility_interval: int = 0,
        local_compatibility_blend: float = 0.1,
        normal_jacobian: Literal["finite_difference", "acoustic"] | None = None,
        neumann_solver: Literal["newton", "energy"] | None = None,
    ):
        self.geometry = geometry
        self.nx = geometry.nx
        self.ny = geometry.ny
        self.nz = geometry.nz
        self.length_x = geometry.length_x
        self.length_y = geometry.length_y
        self.length_z = geometry.length_z
        self.dx = geometry.dx
        self.dy = geometry.dy
        self.dz = geometry.dz
        self.material = material
        self.boundary = boundary
        self.lattice_speed = float(lattice_speed)
        self.collision_omega = float(collision_omega)
        if not (0.0 < self.collision_omega <= 2.0):
            raise ValueError("collision_omega must be in (0,2]")
        self.boundary_reconstruction = _recon_id(boundary_reconstruction)
        self.boundary_reconstruction_name = _recon_name(self.boundary_reconstruction)
        self.normal_jacobian = _normal_jacobian_for_material(material, normal_jacobian)
        self.normal_jacobian_name = normal_jacobian_name(self.normal_jacobian)
        self.neumann_solver_name = neumann_solver_name(neumann_solver, material)
        self._normal_solver_mode = (
            NORMAL_SOLVER_ENERGY if self.neumann_solver_name == "energy" else self.normal_jacobian
        )
        self.dt = self.dx / self.lattice_speed
        self.time = 0.0
        self.steps = 0
        self.device = device or default_device()
        self.shape = (self.nx, self.ny, self.nz)
        self.jacobian_floor = float(jacobian_floor)
        if check_linearized_cfl and material.linearized_stability_ratio(self.lattice_speed) >= 1.0:
            raise ValueError("linearized D3Q6 stability ratio must be below one")
        self.source_mode = source_id(source_mode)
        self.source_params = np.zeros(NCOMP)
        if self.source_mode == SOURCE_CONSTANT:
            for i, v in enumerate(tuple(source_vector)[:NCOMP]):
                self.source_params[i] = float(v)
        elif self.source_mode == SOURCE_DAMPING:
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
        self.local_displacement_interval = int(max(0, local_displacement_interval))
        self.local_displacement_sweeps = int(local_displacement_sweeps)
        self.local_displacement_relax = float(local_displacement_relax)
        self.local_displacement_boundary_weight = float(local_displacement_boundary_weight)
        self.local_displacement_boundary_id = int(local_displacement_boundary_id)
        self.local_compatibility_interval = int(max(0, local_compatibility_interval))
        self.local_compatibility_blend = float(local_compatibility_blend)
        if not (1 <= self.local_displacement_sweeps <= 8):
            raise ValueError("local_displacement_sweeps must be in [1,8]")
        if not (0.0 <= self.local_displacement_relax <= 1.0):
            raise ValueError("local_displacement_relax must be in [0,1]")
        if self.local_displacement_boundary_weight < 0.0:
            raise ValueError("local_displacement_boundary_weight must be non-negative")
        if not (0.0 <= self.local_compatibility_blend <= 1.0):
            raise ValueError("local_compatibility_blend must be in [0,1]")
        self._x_dense: np.ndarray | None = None
        self._y_dense: np.ndarray | None = None
        self._z_dense: np.ndarray | None = None
        self.active_np = np.ascontiguousarray(geometry.active.astype(np.int32))
        tables = _sparse_tables_from_geometry(geometry)
        self.active_i = tables["active_i"]
        self.active_j = tables["active_j"]
        self.active_k = tables["active_k"]
        self.active_id_np = tables["active_id"]
        self.n_active = int(self.active_i.size)
        self.n_nodes = self.n_active
        self.active_i_wp = _as_warp(self.active_i, dtype=wp.int32, device=self.device)
        self.active_j_wp = _as_warp(self.active_j, dtype=wp.int32, device=self.device)
        self.active_k_wp = _as_warp(self.active_k, dtype=wp.int32, device=self.device)
        self.neighbor = _as_warp(tables["neighbor"], dtype=wp.int32, device=self.device)
        self.link_p = _as_warp(tables["link_p"], dtype=wp.int32, device=self.device)
        self.link_upstream = _as_warp(tables["link_upstream"], dtype=wp.int32, device=self.device)
        self.link_q = _as_warp(geometry.link_q, dtype=wp.int32, device=self.device)
        self.link_boundary_id = _as_warp(
            geometry.link_boundary_id, dtype=wp.int32, device=self.device
        )
        self.link_eta = _as_warp(geometry.link_eta, dtype=wp.float64, device=self.device)
        self.link_xb = _as_warp(geometry.link_xb, dtype=wp.float64, device=self.device)
        self.link_yb = _as_warp(geometry.link_yb, dtype=wp.float64, device=self.device)
        self.link_zb = _as_warp(geometry.link_zb, dtype=wp.float64, device=self.device)
        self.link_nx = _as_warp(geometry.link_nx, dtype=wp.float64, device=self.device)
        self.link_ny = _as_warp(geometry.link_ny, dtype=wp.float64, device=self.device)
        self.link_nz = _as_warp(geometry.link_nz, dtype=wp.float64, device=self.device)
        # Tangential displacement LSQ belongs only to the historical
        # compat-BFL closure.  The F-only local closure never reads it, so do
        # not perform that sizeable CPU preprocessing step for local runs.
        self.tangent_lsq_active = self.boundary_reconstruction == BOUNDARY_RECON_COMPAT_BFL
        if self.tangent_lsq_active:
            lsq = _build_sparse_tangent_lsq_weights(geometry, tables)
            self.lsq_p = _as_warp(lsq["lsq_p"], dtype=wp.int32, device=self.device)
            self.lsq_wt1 = _as_warp(lsq["lsq_wt1"], dtype=wp.float64, device=self.device)
            self.lsq_wt2 = _as_warp(lsq["lsq_wt2"], dtype=wp.float64, device=self.device)
            self.lsq_valid = _as_warp(lsq["lsq_valid"], dtype=wp.int32, device=self.device)
            self.lsq_valid_np = lsq["lsq_valid"]
        else:
            nl = int(geometry.n_links)
            self.lsq_p = wp.zeros((nl, MAX_LSQ_STENCIL), dtype=wp.int32, device=self.device)
            self.lsq_wt1 = wp.zeros((nl, MAX_LSQ_STENCIL), dtype=wp.float64, device=self.device)
            self.lsq_wt2 = wp.zeros((nl, MAX_LSQ_STENCIL), dtype=wp.float64, device=self.device)
            self.lsq_valid = wp.zeros(nl, dtype=wp.int32, device=self.device)
            self.lsq_valid_np = np.zeros(nl, dtype=np.int32)
        self.kind_by_id, self.mode_by_id, self.params_by_id = boundary.tables(MAX_BOUNDARY_IDS)
        self.component_kind_by_id = boundary.component_kind_table(MAX_BOUNDARY_IDS)
        link_ids = np.asarray(geometry.link_boundary_id, dtype=np.int32)
        clipped_link_ids = np.clip(link_ids, 0, MAX_BOUNDARY_IDS - 1)
        self._has_neumann_cut_links = bool(
            link_ids.size and np.any(self.component_kind_by_id[:, clipped_link_ids] == BC_NEUMANN)
        )
        self.characteristic_stencil = None
        if self.boundary_reconstruction == BOUNDARY_RECON_LOCAL_F and self._has_neumann_cut_links:
            self.characteristic_stencil = build_characteristic_weights(
                geometry,
                tables,
                link_mask=np.any(
                    self.component_kind_by_id[:, clipped_link_ids] == BC_NEUMANN, axis=0
                ),
            )
            self.char_indices = _as_warp(
                self.characteristic_stencil["indices"], dtype=wp.int32, device=self.device
            )
            self.char_weights = _as_warp(
                self.characteristic_stencil["weights"], dtype=wp.float64, device=self.device
            )
            self.char_state = wp.zeros(
                (4, NCOMP, geometry.n_links), dtype=wp.float64, device=self.device
            )
        else:
            self.char_state = wp.zeros((1, 1, 1), dtype=wp.float64, device=self.device)
        self.kind_by_id_wp = _as_warp(self.kind_by_id, dtype=wp.int32, device=self.device)
        self.component_kind_by_id_wp = _as_warp(
            self.component_kind_by_id, dtype=wp.int32, device=self.device
        )
        self.mode_by_id_wp = _as_warp(self.mode_by_id, dtype=wp.int32, device=self.device)
        self.params_by_id_wp = _as_warp(self.params_by_id, dtype=wp.float64, device=self.device)
        nl = geometry.n_links
        self.link_kind_np = np.zeros(nl, np.int32)
        self.link_component_kind_np = np.zeros((3, nl), np.int32)
        self.link_bv_np = np.zeros((3, nl))
        self.link_Fb_np = np.zeros((9, nl))
        self.link_Pb_np = np.zeros((9, nl))
        self.link_kind = _as_warp(self.link_kind_np, dtype=wp.int32, device=self.device)
        self.link_component_kind = _as_warp(
            self.link_component_kind_np, dtype=wp.int32, device=self.device
        )
        self.link_bv = _as_warp(self.link_bv_np, dtype=wp.float64, device=self.device)
        self.link_Fb = _as_warp(self.link_Fb_np, dtype=wp.float64, device=self.device)
        self.link_Pb = _as_warp(self.link_Pb_np, dtype=wp.float64, device=self.device)
        # Optional device-resident per-link traction data.  A zero enable mask
        # preserves the historical boundary tables exactly.  FSI adapters may
        # bind live device arrays once, update their contents every step, and
        # still use the native compat-BFL upstream/eta/tangent-LSQ preparation.
        self.dynamic_traction_override_enabled = wp.zeros(nl, dtype=wp.int32, device=self.device)
        self.dynamic_traction_override = wp.zeros((3, nl), dtype=wp.float64, device=self.device)
        self.f0 = wp.zeros((FQ, self.n_active), dtype=wp.float64, device=self.device)
        self.f1 = wp.zeros_like(self.f0)
        self.u = wp.zeros((3, self.n_active), dtype=wp.float64, device=self.device)
        self.u_star = wp.zeros_like(self.u)
        self.U = wp.zeros((NCOMP, self.n_active), dtype=wp.float64, device=self.device)
        self.u_local_next = wp.zeros_like(self.u)
        self.local_bc_diag = wp.zeros_like(self.u)
        self.local_bc_rhs = wp.zeros_like(self.u)
        self.invalid = wp.zeros((1,), dtype=wp.int32, device=self.device)
        self.source_params_wp = _as_warp(self.source_params, dtype=wp.float64, device=self.device)
        self.initialize_identity()

    def _build_dense_coordinates(self) -> None:
        if self._x_dense is None or self._y_dense is None or self._z_dense is None:
            xs = (np.arange(self.nx) + 0.5) * self.dx
            ys = (np.arange(self.ny) + 0.5) * self.dy
            zs = (np.arange(self.nz) + 0.5) * self.dz
            self._x_dense, self._y_dense, self._z_dense = np.meshgrid(xs, ys, zs, indexing="ij")

    @property
    def x(self) -> np.ndarray:
        self._build_dense_coordinates()
        return self._x_dense

    @property
    def y(self) -> np.ndarray:
        self._build_dense_coordinates()
        return self._y_dense

    @property
    def z(self) -> np.ndarray:
        self._build_dense_coordinates()
        return self._z_dense

    def active_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.active_i.copy(), self.active_j.copy(), self.active_k.copy()

    def active_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            (self.active_i.astype(np.float64) + 0.5) * self.dx,
            (self.active_j.astype(np.float64) + 0.5) * self.dy,
            (self.active_k.astype(np.float64) + 0.5) * self.dz,
        )

    def memory_estimate_bytes(self) -> dict[str, int]:
        return {
            "populations": int(2 * FQ * self.n_active * 8),
            "fields": int((NCOMP + 3 + 3 + 3 + 3 + 3) * self.n_active * 8),
            "connectivity": int((Q * self.n_active + 2 * self.geometry.n_links) * 4),
            "link_data": int(
                (1 + 3 + 9 + 9 + 3) * self.geometry.n_links * 8 + self.geometry.n_links * 4
            ),
        }

    def _validate_dynamic_traction_override(self, enabled: wp.array, traction: wp.array) -> None:
        """Validate externally owned buffers before binding them."""
        nl = self.geometry.n_links
        if tuple(enabled.shape) != (nl,):
            raise ValueError(f"dynamic traction enable mask must have shape {(nl,)}")
        if tuple(traction.shape) != (3, nl):
            raise ValueError(f"dynamic traction values must have shape {(3, nl)}")
        if enabled.dtype != wp.int32:
            raise TypeError("dynamic traction enable mask must use wp.int32")
        if traction.dtype != wp.float64:
            raise TypeError("dynamic traction values must use wp.float64")
        device = wp.get_device(self.device)
        if enabled.device != device or traction.device != device:
            raise ValueError(f"dynamic traction buffers must reside on {device}")

    def bind_dynamic_traction_override(self, enabled: wp.array, traction: wp.array) -> None:
        """Bind live per-link device buffers without copying them.

        ``enabled[ell] != 0`` replaces the boundary-table traction by
        ``traction[:, ell]``.  Boundary kind and all reconstruction choices
        remain native to this solver.
        """
        self._validate_dynamic_traction_override(enabled, traction)
        self.dynamic_traction_override_enabled = enabled
        self.dynamic_traction_override = traction

    def update_dynamic_traction_override(
        self,
        *,
        enabled: np.ndarray | wp.array | None = None,
        traction: np.ndarray | wp.array | None = None,
    ) -> None:
        """Copy new values into the currently bound device buffers."""
        nl = self.geometry.n_links
        if enabled is not None:
            src = enabled
            if isinstance(enabled, np.ndarray):
                arr = np.ascontiguousarray(enabled, dtype=np.int32)
                if arr.shape != (nl,):
                    raise ValueError(f"dynamic traction enable mask must have shape {(nl,)}")
                src = _as_warp(arr, dtype=wp.int32, device=self.device)
            elif tuple(enabled.shape) != (nl,) or enabled.dtype != wp.int32:
                raise ValueError(
                    f"dynamic traction enable mask must be wp.int32 with shape {(nl,)}"
                )
            wp.copy(self.dynamic_traction_override_enabled, src)
        if traction is not None:
            src = traction
            if isinstance(traction, np.ndarray):
                arr = np.ascontiguousarray(traction, dtype=np.float64)
                if arr.shape != (3, nl):
                    raise ValueError(f"dynamic traction values must have shape {(3, nl)}")
                src = _as_warp(arr, dtype=wp.float64, device=self.device)
            elif tuple(traction.shape) != (3, nl) or traction.dtype != wp.float64:
                raise ValueError(f"dynamic traction values must be wp.float64 with shape {(3, nl)}")
            wp.copy(self.dynamic_traction_override, src)

    def clear_dynamic_traction_override(self) -> None:
        """Disable every override while retaining the bound buffers."""
        self.dynamic_traction_override_enabled.zero_()

    def _active_from_dense(self, arr: np.ndarray, ncomp: int, name: str) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape == (ncomp, self.n_active):
            return np.ascontiguousarray(arr)
        if arr.shape != (ncomp, self.nx, self.ny, self.nz):
            raise ValueError(
                f"{name} must have shape {(ncomp, self.nx, self.ny, self.nz)} or {(ncomp, self.n_active)}"
            )
        return np.ascontiguousarray(arr[:, self.active_np.astype(bool)])

    def _scatter_to_dense(
        self, active_values: np.ndarray, ncomp: int, *, identity_U: bool = False
    ) -> np.ndarray:
        out = np.zeros((ncomp, self.nx, self.ny, self.nz), dtype=np.float64)
        if identity_U and ncomp == NCOMP:
            out[3] = 1.0
            out[7] = 1.0
            out[11] = 1.0
        out[:, self.active_i, self.active_j, self.active_k] = active_values
        return out

    def _source_active_np(self, U: np.ndarray) -> np.ndarray:
        out = np.zeros_like(U)
        if self.source_mode == SOURCE_CONSTANT:
            out[:] = self.source_params[:NCOMP, None]
        elif self.source_mode == SOURCE_DAMPING:
            out[0] = -self.source_params[0] * U[0]
            out[1] = -self.source_params[0] * U[1]
            out[2] = -self.source_params[0] * U[2]
        return out

    def initialize_identity(self, *, init_order: int = 1):
        U = np.zeros((NCOMP, self.n_active), dtype=np.float64)
        U[3] = 1.0
        U[7] = 1.0
        U[11] = 1.0
        self._initialize_from_active(
            U, np.zeros((3, self.n_active), dtype=np.float64), init_order=init_order
        )

    def initialize_from_numpy(
        self,
        *,
        U: np.ndarray,
        displacement: np.ndarray | None = None,
        init_order: int = 1,
        apply_initial_boundaries: bool = True,
        initial_boundary_time: float = 0.0,
    ):
        U_active = self._active_from_dense(U, NCOMP, "U")
        if displacement is None:
            disp_active = np.zeros((3, self.n_active), dtype=np.float64)
        else:
            disp_active = self._active_from_dense(displacement, 3, "displacement")
        self._initialize_from_active(
            U_active,
            disp_active,
            init_order=init_order,
            apply_initial_boundaries=apply_initial_boundaries,
            initial_boundary_time=initial_boundary_time,
        )

    def _initialize_from_active(
        self,
        U_active: np.ndarray,
        disp_active: np.ndarray,
        *,
        init_order: int = 1,
        apply_initial_boundaries: bool = True,
        initial_boundary_time: float = 0.0,
    ):
        if init_order != 1:
            raise ValueError(
                "sparse 3-D curved solver currently supports init_order=1; use dense class for higher-order initialization"
            )
        f = _equilibrium_np(U_active, self.material, self.lattice_speed)
        f -= 0.5 * self.dt / float(Q) * self._source_active_np(U_active)[None, ...]
        wp.copy(
            self.f0,
            _as_warp(
                np.ascontiguousarray(f.reshape(FQ, self.n_active)),
                dtype=wp.float64,
                device=self.device,
            ),
        )
        wp.launch(_sparse_clear, dim=(FQ, self.n_active), inputs=[self.f1], device=self.device)
        us0 = np.ascontiguousarray(disp_active - 0.5 * self.dt * U_active[:3])
        wp.copy(self.u_star, _as_warp(us0, dtype=wp.float64, device=self.device))
        self.time = 0.0
        self.steps = 0
        self.refresh_current()
        if apply_initial_boundaries:
            self.apply_boundary_reconstruction(initial_boundary_time, in_place=True)
            self.refresh_current()
            disp_wp = _as_warp(
                np.ascontiguousarray(disp_active), dtype=wp.float64, device=self.device
            )
            wp.launch(
                _sparse_set_ustar,
                dim=self.n_active,
                inputs=[disp_wp, self.U, self.u_star, self.dt],
                device=self.device,
            )
            self.refresh_current()

    def _prepare_links(self, t: float):
        if self.geometry.n_links == 0:
            return
        if self.characteristic_stencil is not None:
            wp.launch(
                _sparse_characteristic_gather,
                dim=(NCOMP, self.geometry.n_links),
                inputs=[self.U, self.char_indices, self.char_weights, self.char_state],
                device=self.device,
            )
        wp.launch(
            _sparse_prepare_links_gpu,
            dim=self.geometry.n_links,
            inputs=[
                self.U,
                self.char_state,
                self.u,
                self.link_p,
                self.link_upstream,
                self.link_q,
                self.link_boundary_id,
                self.link_eta,
                self.link_xb,
                self.link_yb,
                self.link_zb,
                self.link_nx,
                self.link_ny,
                self.link_nz,
                self.lsq_p,
                self.lsq_wt1,
                self.lsq_wt2,
                self.lsq_valid,
                self.kind_by_id_wp,
                self.component_kind_by_id_wp,
                self.mode_by_id_wp,
                self.params_by_id_wp,
                self.dynamic_traction_override_enabled,
                self.dynamic_traction_override,
                self.link_kind,
                self.link_component_kind,
                self.link_bv,
                self.link_Fb,
                self.link_Pb,
                self.geometry.n_links,
                float(t),
                self.material.id,
                self.material.lam,
                self.material.mu,
                self.material.param0,
                self.material.param1,
                self.material.param2,
                self.jacobian_floor,
                self.boundary_reconstruction,
                self._normal_solver_mode,
            ],
            device=self.device,
        )

    def apply_boundary_reconstruction(self, t: float, *, in_place: bool = False):
        if self.geometry.n_links == 0:
            return
        self._prepare_links(t)
        if in_place:
            # f1 is scratch storage until the next streaming step.
            wp.copy(self.f1, self.f0)
        src = self.f1 if in_place else self.f0
        tgt = self.f0 if in_place else self.f1
        # Dirichlet rows retain their original BFL population values.
        wp.launch(
            _sparse_apply_links,
            dim=self.geometry.n_links,
            inputs=[
                src,
                tgt,
                self.link_p,
                self.link_upstream,
                self.link_q,
                self.link_eta,
                self.link_kind,
                self.link_component_kind,
                self.link_bv,
                self.link_Fb,
                self.link_Pb,
                self.geometry.n_links,
                self.lattice_speed,
                self.boundary_reconstruction,
            ],
            device=self.device,
        )
        if self.characteristic_stencil is not None:
            wp.launch(
                _sparse_apply_characteristic,
                dim=self.geometry.n_links,
                inputs=[
                    self.U,
                    tgt,
                    self.char_state,
                    self.link_p,
                    self.link_q,
                    self.link_eta,
                    self.link_component_kind,
                    self.link_bv,
                    self.link_Fb,
                    self.source_params_wp,
                    self.source_mode,
                    self.dt,
                    self.lattice_speed,
                    self.material.id,
                    self.material.lam,
                    self.material.mu,
                    self.material.param0,
                    self.material.param1,
                    self.material.param2,
                    self.jacobian_floor,
                    self.collision_omega,
                    self.invalid,
                ],
                device=self.device,
            )

    def refresh_current(self):
        wp.launch(_reset_invalid_kernel, dim=1, inputs=[self.invalid], device=self.device)
        wp.launch(
            _sparse_refresh_kernel,
            dim=self.n_active,
            inputs=[
                self.f0,
                self.u,
                self.u_star,
                self.U,
                self.invalid,
                self.source_params_wp,
                self.dt,
                self.material.id,
                self.material.lam,
                self.material.mu,
                self.material.param0,
                self.material.param1,
                self.material.param2,
                self.jacobian_floor,
                self.source_mode,
            ],
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
        boundary_id: int = -1,
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

    def _local_displacement_due(self) -> bool:
        return (
            self.local_displacement_interval > 0
            and self.steps % self.local_displacement_interval == 0
        )

    def _local_compatibility_due(self) -> bool:
        return (
            self.local_compatibility_interval > 0
            and self.steps % self.local_compatibility_interval == 0
        )

    def _apply_local_displacement(self, boundary_time: float) -> None:
        wp.launch(
            _sparse_local_bc_clear,
            dim=self.n_active,
            inputs=[self.local_bc_diag, self.local_bc_rhs],
            device=self.device,
        )
        if self.geometry.n_links > 0:
            wp.launch(
                _sparse_local_build_bc,
                dim=self.geometry.n_links,
                inputs=[
                    self.U,
                    self.active_i_wp,
                    self.active_j_wp,
                    self.active_k_wp,
                    self.link_p,
                    self.link_boundary_id,
                    self.link_xb,
                    self.link_yb,
                    self.link_zb,
                    self.component_kind_by_id_wp,
                    self.mode_by_id_wp,
                    self.params_by_id_wp,
                    self.geometry.n_links,
                    self.local_displacement_boundary_id,
                    boundary_time,
                    self.dx,
                    self.dy,
                    self.dz,
                    self.local_displacement_boundary_weight,
                    self.local_bc_diag,
                    self.local_bc_rhs,
                ],
                device=self.device,
            )
        for _ in range(self.local_displacement_sweeps):
            wp.launch(
                _sparse_local_displacement_sweep,
                dim=self.n_active,
                inputs=[
                    self.U,
                    self.neighbor,
                    self.u,
                    self.u_local_next,
                    self.local_bc_diag,
                    self.local_bc_rhs,
                    self.dx,
                    self.dy,
                    self.dz,
                    self.local_displacement_relax,
                ],
                device=self.device,
            )
            wp.launch(
                _sparse_local_displacement_commit,
                dim=self.n_active,
                inputs=[self.u_local_next, self.u, self.u_star],
                device=self.device,
            )

    def _apply_local_compatibility(self) -> None:
        wp.launch(
            _sparse_local_repair_F,
            dim=self.n_active,
            inputs=[
                self.f0,
                self.U,
                self.neighbor,
                self.u,
                self.dx,
                self.dy,
                self.dz,
                self.local_compatibility_blend,
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

    def step(self):
        wp.launch(_reset_invalid_kernel, dim=1, inputs=[self.invalid], device=self.device)
        wp.launch(
            _sparse_collide_inplace_kernel,
            dim=self.n_active,
            inputs=[
                self.f0,
                self.u,
                self.u_star,
                self.U,
                self.invalid,
                self.source_params_wp,
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
            ],
            device=self.device,
        )
        if self._local_displacement_due():
            self._apply_local_displacement(self.time + 0.5 * self.dt)
        if self._local_compatibility_due():
            self._apply_local_compatibility()
        wp.launch(_sparse_clear, dim=(FQ, self.n_active), inputs=[self.f1], device=self.device)
        wp.launch(
            _sparse_stream_kernel,
            dim=self.n_active,
            inputs=[self.f0, self.f1, self.neighbor],
            device=self.device,
        )
        self.apply_boundary_reconstruction(self.time + 0.5 * self.dt)
        self.f0, self.f1 = self.f1, self.f0
        self.time += self.dt
        self.steps += 1

    def run_until(self, t_end: float):
        while self.time < float(t_end) - 0.5 * self.dt:
            self.step()
        self.refresh_current()
        wp.synchronize_device(self.device)

    def active_mask(self):
        return self.active_np.astype(bool)

    def active_system_state(self) -> np.ndarray:
        wp.synchronize_device(self.device)
        return np.asarray(self.U.numpy())

    def active_displacement(self) -> np.ndarray:
        wp.synchronize_device(self.device)
        return np.asarray(self.u.numpy())

    def active_ustar(self) -> np.ndarray:
        wp.synchronize_device(self.device)
        return np.asarray(self.u_star.numpy())

    def active_populations(self) -> np.ndarray:
        wp.synchronize_device(self.device)
        return np.asarray(self.f0.numpy())

    def active_deformation_gradient(self) -> np.ndarray:
        return _state_to_F_np(self.active_system_state())

    def active_deformation_jacobian(self) -> np.ndarray:
        return _det3_np(self.active_deformation_gradient())

    def active_piola(self) -> np.ndarray:
        return _first_piola_np(self.active_deformation_gradient(), self.material)

    def active_sigma(self) -> np.ndarray:
        F = self.active_deformation_gradient()
        P = _first_piola_np(F, self.material)
        J = np.maximum(_det3_np(F), self.jacobian_floor)
        return np.einsum("ia...,ja...->ij...", P, F) / J

    def numpy_field(self, name):
        wp.synchronize_device(self.device)
        if name == "U":
            return self._scatter_to_dense(np.asarray(self.U.numpy()), NCOMP, identity_U=True)
        if name == "u":
            return self._scatter_to_dense(np.asarray(self.u.numpy()), 3)
        if name == "u_star":
            return self._scatter_to_dense(np.asarray(self.u_star.numpy()), 3)
        if name == "f0":
            return self._scatter_to_dense(np.asarray(self.f0.numpy()), FQ)
        if name == "f1":
            return self._scatter_to_dense(np.asarray(self.f1.numpy()), FQ)
        if name == "P" or name == "sigma":
            U = np.asarray(self.U.numpy())
            F = U[3:12].reshape(3, 3, self.n_active)
            P = _first_piola_np(F, self.material)
            if name == "P":
                return self._scatter_to_dense(P.reshape(9, self.n_active), 9)
            J = _det3_np(F)
            J = np.maximum(J, self.jacobian_floor)
            sigma = np.einsum("ia...,ja...->ij...", P, F) / J
            return self._scatter_to_dense(sigma.reshape(9, self.n_active), 9)
        raise AttributeError(name)

    def system_state(self):
        return self.numpy_field("U")

    def displacement(self):
        return self.numpy_field("u")

    def deformation_gradient(self):
        return _state_to_F_np(self.system_state())

    def deformation_jacobian(self):
        return _det3_np(self.deformation_gradient())

    def invalid_state(self):
        wp.synchronize_device(self.device)
        U = np.asarray(self.U.numpy())
        F = U[3:12].reshape(3, 3, self.n_active)
        J = _det3_np(F)
        return (
            int(np.asarray(self.invalid.numpy())[0]) != 0
            or not np.isfinite(U).all()
            or not np.all(J > 0)
        )

    def summary(self):
        self.refresh_current()
        wp.synchronize_device(self.device)
        U = np.asarray(self.U.numpy())
        u = np.asarray(self.u.numpy())
        F = U[3:12].reshape(3, 3, self.n_active)
        J = _det3_np(F)
        mem = self.memory_estimate_bytes()
        displacement_radius = (
            int(self.local_displacement_sweeps) if self.local_displacement_interval > 0 else 0
        )
        composed_radius = (
            (displacement_radius + 2)
            if self.local_compatibility_interval > 0
            else displacement_radius
        )
        return {
            "device": self.device,
            "storage": "sparse_active",
            "geometry": self.geometry.label,
            "boundary_label": self.boundary.label,
            "model": self.material.model,
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "n_active": int(self.n_active),
            "n_cut_links": int(self.geometry.n_links),
            "steps": int(self.steps),
            "time": float(self.time),
            "dx": float(self.dx),
            "dt": float(self.dt),
            "lattice_speed": float(self.lattice_speed),
            "collision_omega": float(self.collision_omega),
            "boundary_reconstruction": self.boundary_reconstruction_name,
            "population_reconstruction": "dirichlet_bfl_neumann_characteristic"
            if self.characteristic_stencil is not None
            else "bfl",
            "boundary_stencil_radius": 3 if self.characteristic_stencil is not None else 1,
            "characteristic_cut_links": int(
                np.count_nonzero(
                    np.any(
                        self.component_kind_by_id[
                            :, np.clip(self.geometry.link_boundary_id, 0, MAX_BOUNDARY_IDS - 1)
                        ]
                        == BC_NEUMANN,
                        axis=0,
                    )
                )
            )
            if self.characteristic_stencil is not None
            else 0,
            "bfl_cut_links": int(
                np.count_nonzero(
                    np.any(
                        self.component_kind_by_id[
                            :, np.clip(self.geometry.link_boundary_id, 0, MAX_BOUNDARY_IDS - 1)
                        ]
                        == BC_DIRICHLET,
                        axis=0,
                    )
                )
            )
            if self.characteristic_stencil is not None
            else int(self.geometry.n_links),
            "normal_jacobian": self.normal_jacobian_name,
            "neumann_solver": self.neumann_solver_name,
            "has_neumann_cut_links": bool(self._has_neumann_cut_links),
            "surface_tangent_lsq_active": bool(self.tangent_lsq_active),
            "compatibility_projection": {"interval": 0, "active_in_step": False},
            "local_method": {
                "displacement_interval": int(self.local_displacement_interval),
                "displacement_sweeps": int(self.local_displacement_sweeps),
                "displacement_relax": float(self.local_displacement_relax),
                "boundary_weight": float(self.local_displacement_boundary_weight),
                "boundary_id": int(self.local_displacement_boundary_id),
                "F_repair_interval": int(self.local_compatibility_interval),
                "F_repair_blend": float(self.local_compatibility_blend),
                "kernel_stencil_radius": 2
                if self.local_compatibility_interval > 0
                else (1 if displacement_radius > 0 else 0),
                "displacement_dependency_radius": displacement_radius,
                "composed_compatibility_radius": composed_radius,
            },
            "memory_estimate_MiB": {k: float(v / 1024**2) for k, v in mem.items()},
            "max_abs_u": float(np.max(np.sqrt(np.sum(u * u, axis=0)))) if self.n_active else 0.0,
            "max_abs_v": float(np.max(np.sqrt(U[0] * U[0] + U[1] * U[1] + U[2] * U[2])))
            if self.n_active
            else 0.0,
            "min_J": float(np.min(J)) if self.n_active else 0.0,
            "max_J": float(np.max(J)) if self.n_active else 0.0,
            "finite": bool(np.isfinite(U).all() and np.all(J > 0)) if self.n_active else False,
        }

    def save_npz(self, path):
        self.refresh_current()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        xa, ya, za = self.active_coordinates()
        U = self.active_system_state()
        u = self.active_displacement()
        F = _state_to_F_np(U)
        P = _first_piola_np(F, self.material)
        J = np.maximum(_det3_np(F), self.jacobian_floor)
        sigma = np.einsum("ia...,ja...->ij...", P, F) / J
        np.savez_compressed(
            path,
            x_active=xa,
            y_active=ya,
            z_active=za,
            active_i=self.active_i,
            active_j=self.active_j,
            active_k=self.active_k,
            shape=np.array([self.nx, self.ny, self.nz], dtype=np.int32),
            dx=np.array(self.dx),
            dy=np.array(self.dy),
            dz=np.array(self.dz),
            u=u,
            U=U,
            F=F,
            P=P,
            sigma=sigma,
            time=np.array(self.time),
            storage=np.array("sparse_active"),
            metadata=np.array(json.dumps(self.summary())),
        )


WarpCurvedBoundaryLBM3D = WarpSparseCurvedBoundaryLBM3D


def relative_l2_active(error, reference, mask):
    e = np.asarray(error)
    r = np.asarray(reference)
    m = np.asarray(mask, bool)
    return float(
        np.sqrt(np.sum(e[..., m] ** 2)) / max(np.sqrt(np.sum(r[..., m] ** 2)), np.finfo(float).eps)
    )


# Compatibility aliases for the notation used in DERIVATION_CURVED_BOUNDARY_3D.md.
_stream_masked_kernel = _stream_masked
_apply_curved_boundary_links_kernel = _apply_links
_solve_Fb_neumann_host = _solve_Fb_np


__all__ = [
    "WarpHyperelasticMaterial3D",
    "CurvedBoundarySpec3D",
    "CurvedBoundaryGeometry3D",
    "WarpCurvedBoundaryLBM3D",
    "WarpSparseCurvedBoundaryLBM3D",
    "WarpDenseCurvedBoundaryLBM3D",
    "constant_piola_params3d",
    "relative_l2_active",
    "CURVED3D_VALUE_ZERO",
    "CURVED3D_VALUE_CONSTANT_VECTOR",
    "CURVED3D_VALUE_AFFINE_VELOCITY",
    "CURVED3D_VALUE_RAMP_VECTOR",
    "CURVED3D_VALUE_SINE2_HOLD_VECTOR",
    "CURVED3D_VALUE_PIOLA_NORMAL",
    "CURVED3D_VALUE_NORMAL_TRACTION",
    "CURVED3D_VALUE_SINE2_NORMAL_TRACTION",
    "CURVED3D_VALUE_AFFINE_NEO_HOOKE_PIOLA_NORMAL",
    "CURVED3D_VALUE_Z_PULL_TORSION_TRACTION",
    "CURVED3D_VALUE_Z_STRETCH_EXACT_TWIST_VELOCITY",
    "CURVED_VALUE_ZERO",
    "CURVED_VALUE_CONSTANT_VECTOR",
    "CURVED_VALUE_AFFINE_VELOCITY",
    "CURVED_VALUE_RAMP_VECTOR",
    "CURVED_VALUE_SINE2_HOLD_VECTOR",
    "CURVED_VALUE_PIOLA_NORMAL",
    "CURVED_VALUE_NORMAL_TRACTION",
    "CURVED_VALUE_SINE2_NORMAL_TRACTION",
    "CURVED_VALUE_AFFINE_NEO_HOOKE_PIOLA_NORMAL",
    "CURVED_VALUE_Z_PULL_TORSION_TRACTION",
    "CURVED_VALUE_Z_STRETCH_EXACT_TWIST_VELOCITY",
    "BOUNDARY_RECON_BFL",
    "BOUNDARY_RECON_HALF_WAY",
    "BOUNDARY_RECON_COMPAT_BFL",
    "BOUNDARY_RECON_LOCAL_F",
    "BOUNDARY_ID_OUTER",
    "BOUNDARY_ID_INNER",
]

if __name__ == "__main__":
    mat = WarpHyperelasticMaterial3D.from_poisson("svk", 0.2, mu=1.0)
    geom = CurvedBoundaryGeometry3D.sphere(nx=8, ny=8, nz=8, radius=0.32)
    bc = CurvedBoundarySpec3D(kind="dirichlet", value_mode=CURVED3D_VALUE_ZERO)
    sim = WarpCurvedBoundaryLBM3D(
        geom,
        material=mat,
        boundary=bc,
        lattice_speed=10.0,
        collision_omega=2.0,
        device="cpu",
        check_linearized_cfl=False,
    )
    sim.step()
    print(json.dumps(sim.summary(), indent=2))
