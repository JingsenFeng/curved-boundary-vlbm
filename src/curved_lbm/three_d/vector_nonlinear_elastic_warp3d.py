"""Warp D3Q6x12 vectorial LBM for 3D finite-strain elastodynamics.

This module generalizes the total-Lagrangian vectorial LBM from the uploaded
2D D2Q4x6 prototype to a 3D D3Q6x12 scheme.  The state per lattice direction is

    U = [v_x, v_y, v_z,
         F_11, F_12, F_13,
         F_21, F_22, F_23,
         F_31, F_32, F_33].

The code is written for NVIDIA Warp.  It runs on Warp's CPU device when CUDA is
not available and uses the same kernels on CUDA devices.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import warp as wp


wp.init()

DIM = 3
Q = 6
NCOMP = 12
FQ = Q * NCOMP

MATERIAL_SVK = 0
MATERIAL_NEO_HOOKE = 1
MATERIAL_LOG_NEO_HOOKE = 2
MATERIAL_MOONEY_RIVLIN = 3
MATERIAL_YEOH = 4
MATERIAL_GENT = 5
MATERIAL_SPH_NEO_HOOKE = 6

BC_PERIODIC = 0
BC_DIRICHLET = 1
BC_NEUMANN = 2

SIDE_LEFT = 0  # X = 0, N=(-1,0,0)
SIDE_RIGHT = 1  # X = Lx, N=(+1,0,0)
SIDE_BOTTOM = 2  # Y = 0, N=(0,-1,0)
SIDE_TOP = 3  # Y = Ly, N=(0,+1,0)
SIDE_FRONT = 4  # Z = 0, N=(0,0,-1)
SIDE_BACK = 5  # Z = Lz, N=(0,0,+1)

VALUE_ZERO = 0
VALUE_CONSTANT = 1
VALUE_RAMP = 2
VALUE_SINE2_HOLD = 3
VALUE_AFFINE_VELOCITY = 4
VALUE_SHEAR_VELOCITY_X = 5

SOURCE_NONE = 0
SOURCE_CONSTANT = 1
SOURCE_DAMPING = 2

PI = wp.constant(wp.float64(3.141592653589793238462643383279502884))

_DIRS = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ],
    dtype=np.int32,
)

MaterialName = Literal[
    "svk",
    "neo_hooke",
    "log_neo_hooke",
    "mooney_rivlin",
    "yeoh",
    "gent",
    "sph_neo_hooke",
]
BoundaryKind = Literal["periodic", "dirichlet", "neumann"]
SideName = Literal["left", "right", "bottom", "top", "front", "back", "zmin", "zmax"]
SourceName = Literal["none", "constant", "damping"]


def default_device() -> str:
    requested = os.environ.get("WARP_DEVICE")
    if requested:
        return requested
    if wp.is_cuda_available() and wp.get_cuda_device_count() > 0:
        return "cuda:0"
    return "cpu"


def material_id(name: str) -> int:
    if name == "svk":
        return MATERIAL_SVK
    if name == "neo_hooke":
        return MATERIAL_NEO_HOOKE
    if name == "log_neo_hooke":
        return MATERIAL_LOG_NEO_HOOKE
    if name == "mooney_rivlin":
        return MATERIAL_MOONEY_RIVLIN
    if name == "yeoh":
        return MATERIAL_YEOH
    if name == "gent":
        return MATERIAL_GENT
    if name == "sph_neo_hooke":
        return MATERIAL_SPH_NEO_HOOKE
    raise ValueError(f"unsupported material model: {name}")


def material_name(mid: int) -> str:
    if mid == MATERIAL_SVK:
        return "svk"
    if mid == MATERIAL_NEO_HOOKE:
        return "neo_hooke"
    if mid == MATERIAL_LOG_NEO_HOOKE:
        return "log_neo_hooke"
    if mid == MATERIAL_MOONEY_RIVLIN:
        return "mooney_rivlin"
    if mid == MATERIAL_YEOH:
        return "yeoh"
    if mid == MATERIAL_GENT:
        return "gent"
    if mid == MATERIAL_SPH_NEO_HOOKE:
        return "sph_neo_hooke"
    return f"unknown_{mid}"


def side_id(side: str) -> int:
    if side == "left":
        return SIDE_LEFT
    if side == "right":
        return SIDE_RIGHT
    if side == "bottom":
        return SIDE_BOTTOM
    if side == "top":
        return SIDE_TOP
    if side in ("front", "zmin"):
        return SIDE_FRONT
    if side in ("back", "zmax"):
        return SIDE_BACK
    raise ValueError(f"unsupported side: {side}")


def side_name(sid: int) -> str:
    return ("left", "right", "bottom", "top", "front", "back")[int(sid)]


def kind_id(kind: str) -> int:
    if kind == "periodic":
        return BC_PERIODIC
    if kind == "dirichlet":
        return BC_DIRICHLET
    if kind == "neumann":
        return BC_NEUMANN
    raise ValueError(f"unsupported boundary kind: {kind}")


def source_id(name: str) -> int:
    if name == "none":
        return SOURCE_NONE
    if name == "constant":
        return SOURCE_CONSTANT
    if name == "damping":
        return SOURCE_DAMPING
    raise ValueError(f"unsupported source mode: {name}")


def source_name(sid: int) -> str:
    if sid == SOURCE_NONE:
        return "none"
    if sid == SOURCE_CONSTANT:
        return "constant"
    if sid == SOURCE_DAMPING:
        return "damping"
    return f"unknown_{sid}"


@dataclass(frozen=True)
class WarpHyperelasticMaterial3D:
    """Compressible 3D hyperelastic material used by the local Piola map."""

    model: MaterialName
    lam: float
    mu: float = 1.0
    param0: float = 0.0
    param1: float = 0.0
    param2: float = 0.0

    def __post_init__(self) -> None:
        if self.model == "mooney_rivlin" and self.param0 == 0.0 and self.param1 == 0.0:
            csum = 0.5 * self.mu
            object.__setattr__(self, "param0", 0.75 * csum)  # c10
            object.__setattr__(self, "param1", 0.25 * csum)  # c01
        elif self.model == "yeoh" and self.param0 == 0.0:
            object.__setattr__(self, "param0", 0.5 * self.mu)  # c1
        elif self.model == "gent" and self.param0 == 0.0:
            object.__setattr__(self, "param0", 50.0)  # Jm

    @classmethod
    def from_poisson(
        cls,
        model: MaterialName,
        poisson: float,
        *,
        mu: float = 1.0,
        mooney_c01_fraction: float = 0.25,
        yeoh_c2: float = 0.0,
        yeoh_c3: float = 0.0,
        gent_jm: float = 50.0,
    ) -> "WarpHyperelasticMaterial3D":
        if abs(1.0 - 2.0 * poisson) < 1e-14:
            raise ValueError("poisson ratio too close to 1/2")
        lame_lambda = 2.0 * mu * poisson / (1.0 - 2.0 * poisson)
        # In the section-5.3 SPH energy, the historical ``lam`` storage slot
        # carries the bulk modulus K rather than the infinitesimal first Lamé
        # parameter.  K = lambda + 2 mu / 3.
        lam = lame_lambda + (2.0 / 3.0) * mu if model == "sph_neo_hooke" else lame_lambda
        if model == "mooney_rivlin":
            if not (0.0 <= mooney_c01_fraction <= 1.0):
                raise ValueError("mooney_c01_fraction must be in [0, 1]")
            csum = 0.5 * mu
            return cls(
                model=model,
                lam=lam,
                mu=mu,
                param0=(1.0 - mooney_c01_fraction) * csum,
                param1=mooney_c01_fraction * csum,
            )
        if model == "yeoh":
            return cls(
                model=model,
                lam=lam,
                mu=mu,
                param0=0.5 * mu,
                param1=float(yeoh_c2),
                param2=float(yeoh_c3),
            )
        if model == "gent":
            if gent_jm <= 0.0:
                raise ValueError("gent_jm must be positive")
            return cls(model=model, lam=lam, mu=mu, param0=float(gent_jm))
        return cls(model=model, lam=lam, mu=mu)

    @property
    def id(self) -> int:
        return material_id(self.model)

    @property
    def poisson(self) -> float:
        linearized_lame = self.linearized_lame
        return linearized_lame / (2.0 * (linearized_lame + self.mu))

    @property
    def linearized_lame(self) -> float:
        """Infinitesimal first Lamé parameter of the finite-strain model."""
        if self.model == "sph_neo_hooke":
            return self.lam - (2.0 / 3.0) * self.mu
        return self.lam

    @property
    def bulk_modulus(self) -> float:
        """Infinitesimal bulk modulus."""
        if self.model == "sph_neo_hooke":
            return self.lam
        return self.lam + (2.0 / 3.0) * self.mu

    @property
    def params(self) -> tuple[float, float, float]:
        return float(self.param0), float(self.param1), float(self.param2)

    def linearized_stability_ratio(self, lattice_speed: float) -> float:
        """Conservative D3Q6 audit using the infinitesimal longitudinal speed."""
        longitudinal_modulus = self.linearized_lame + 2.0 * self.mu
        return 2.0 * math.sqrt(longitudinal_modulus) / float(lattice_speed)


@dataclass(frozen=True)
class BoxBoundarySpec:
    side: SideName
    kind: BoundaryKind
    value_mode: int = VALUE_ZERO
    params: tuple[float, ...] = ()
    label: str = ""


class BoxBoundarySet:
    """Compact boundary encoding for axis-aligned boxes."""

    def __init__(self, specs: tuple[BoxBoundarySpec, ...] = ()) -> None:
        self.side_kind = np.full(6, BC_PERIODIC, dtype=np.int32)
        self.value_mode = np.zeros(6, dtype=np.int32)
        self.params = np.zeros((6, 16), dtype=np.float64)
        self.labels = ["periodic"] * 6
        for spec in specs:
            self.set(spec)
        self.validate_periodic_pairs()

    def set(self, spec: BoxBoundarySpec) -> None:
        sid = side_id(spec.side)
        self.side_kind[sid] = kind_id(spec.kind)
        self.value_mode[sid] = int(spec.value_mode)
        self.params[sid, :] = 0.0
        for i, value in enumerate(spec.params[:16]):
            self.params[sid, i] = float(value)
        self.labels[sid] = spec.label or f"{spec.kind}:{spec.side}"

    def validate_periodic_pairs(self) -> None:
        if (self.side_kind[SIDE_LEFT] == BC_PERIODIC) != (
            self.side_kind[SIDE_RIGHT] == BC_PERIODIC
        ):
            raise ValueError("x-periodic boundaries must be set on both left and right")
        if (self.side_kind[SIDE_BOTTOM] == BC_PERIODIC) != (
            self.side_kind[SIDE_TOP] == BC_PERIODIC
        ):
            raise ValueError("y-periodic boundaries must be set on both bottom and top")
        if (self.side_kind[SIDE_FRONT] == BC_PERIODIC) != (
            self.side_kind[SIDE_BACK] == BC_PERIODIC
        ):
            raise ValueError("z-periodic boundaries must be set on both front and back")

    @property
    def periodic_x(self) -> bool:
        return bool(
            self.side_kind[SIDE_LEFT] == BC_PERIODIC and self.side_kind[SIDE_RIGHT] == BC_PERIODIC
        )

    @property
    def periodic_y(self) -> bool:
        return bool(
            self.side_kind[SIDE_BOTTOM] == BC_PERIODIC and self.side_kind[SIDE_TOP] == BC_PERIODIC
        )

    @property
    def periodic_z(self) -> bool:
        return bool(
            self.side_kind[SIDE_FRONT] == BC_PERIODIC and self.side_kind[SIDE_BACK] == BC_PERIODIC
        )

    @classmethod
    def periodic(cls) -> "BoxBoundarySet":
        return cls(())

    @classmethod
    def all_dirichlet_zero(cls) -> "BoxBoundarySet":
        return cls(
            tuple(
                BoxBoundarySpec(s, "dirichlet", VALUE_ZERO, label="zero velocity")
                for s in ("left", "right", "bottom", "top", "front", "back")
            )
        )

    @classmethod
    def all_neumann_zero(cls) -> "BoxBoundarySet":
        return cls(
            tuple(
                BoxBoundarySpec(s, "neumann", VALUE_ZERO, label="zero nominal traction")
                for s in ("left", "right", "bottom", "top", "front", "back")
            )
        )

    @classmethod
    def affine_velocity(
        cls,
        A: Sequence[Sequence[float]],
        b: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        active_time: float = 0.0,
    ) -> "BoxBoundarySet":
        A_arr = np.asarray(A, dtype=np.float64).reshape(3, 3)
        b_arr = np.asarray(b, dtype=np.float64).reshape(3)
        params = tuple(
            A_arr.reshape(-1).tolist() + b_arr.tolist() + [float(active_time), 0.0, 0.0, 0.0]
        )
        return cls(
            tuple(
                BoxBoundarySpec(s, "dirichlet", VALUE_AFFINE_VELOCITY, params, f"affine {s}")
                for s in ("left", "right", "bottom", "top", "front", "back")
            )
        )


def reference_grid(
    nx: int, ny: int, nz: int, length_x: float, length_y: float, length_z: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = float(length_x) / float(nx)
    dy = float(length_y) / float(ny)
    dz = float(length_z) / float(nz)
    x = (np.arange(nx, dtype=np.float64) + 0.5) * dx
    y = (np.arange(ny, dtype=np.float64) + 0.5) * dy
    z = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    return np.meshgrid(x, y, z, indexing="ij")


def _as_warp(arr: np.ndarray, *, dtype, device: str) -> wp.array:
    return wp.array(np.ascontiguousarray(arr), dtype=dtype, device=device)


def _cofactor_np(F: np.ndarray) -> np.ndarray:
    C = np.empty_like(F)
    F11, F12, F13 = F[0, 0], F[0, 1], F[0, 2]
    F21, F22, F23 = F[1, 0], F[1, 1], F[1, 2]
    F31, F32, F33 = F[2, 0], F[2, 1], F[2, 2]
    C[0, 0] = F22 * F33 - F23 * F32
    C[0, 1] = F23 * F31 - F21 * F33
    C[0, 2] = F21 * F32 - F22 * F31
    C[1, 0] = F13 * F32 - F12 * F33
    C[1, 1] = F11 * F33 - F13 * F31
    C[1, 2] = F12 * F31 - F11 * F32
    C[2, 0] = F12 * F23 - F13 * F22
    C[2, 1] = F13 * F21 - F11 * F23
    C[2, 2] = F11 * F22 - F12 * F21
    return C


def _det3_np(F: np.ndarray) -> np.ndarray:
    return (
        F[0, 0] * (F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1])
        - F[0, 1] * (F[1, 0] * F[2, 2] - F[1, 2] * F[2, 0])
        + F[0, 2] * (F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0])
    )


def _first_piola_np(F: np.ndarray, material: WarpHyperelasticMaterial3D) -> np.ndarray:
    F = np.asarray(F, dtype=np.float64)
    if F.shape[:2] != (3, 3):
        raise ValueError("F must have leading shape (3,3)")
    if material.model == "svk":
        C = np.einsum("ia...,ib...->ab...", F, F)
        E = 0.5 * C
        E[0, 0] -= 0.5
        E[1, 1] -= 0.5
        E[2, 2] -= 0.5
        trE = E[0, 0] + E[1, 1] + E[2, 2]
        S = 2.0 * material.mu * E
        S[0, 0] += material.lam * trE
        S[1, 1] += material.lam * trE
        S[2, 2] += material.lam * trE
        return np.einsum("ib...,ba...->ia...", F, S)

    J = _det3_np(F)
    if np.any(J <= 0.0):
        raise FloatingPointError(
            f"non-positive deformation Jacobian: min(J)={float(np.min(J)):.6e}"
        )
    cof = _cofactor_np(F)
    H = cof / J
    if material.model == "neo_hooke":
        volumetric = 0.5 * material.lam * (J * J - 1.0)
        return material.mu * (F - H) + volumetric * H
    if material.model == "log_neo_hooke":
        volumetric = material.lam * np.log(J)
        return material.mu * (F - H) + volumetric * H

    C = np.einsum("ia...,ib...->ab...", F, F)
    I1 = C[0, 0] + C[1, 1] + C[2, 2]
    logJ = np.log(J)
    jm23 = np.exp((-2.0 / 3.0) * logJ)
    G1 = jm23 * (2.0 * F - (2.0 / 3.0) * I1 * H)
    if material.model == "sph_neo_hooke":
        # sph.pdf Eq. (13):
        #   W = K/4 (J^2 - 1 - 2 ln J)
        #       + mu/2 (J^(-2/3) I1 - 3).
        # Here material.lam stores K.  G1 is the exact derivative of
        # J^(-2/3) I1, while H = F^(-T).
        volumetric = 0.5 * material.lam * (J * J - 1.0)
        return 0.5 * material.mu * G1 + volumetric * H
    trC2 = (
        C[0, 0] * C[0, 0]
        + C[1, 1] * C[1, 1]
        + C[2, 2] * C[2, 2]
        + 2.0 * (C[0, 1] * C[0, 1] + C[0, 2] * C[0, 2] + C[1, 2] * C[1, 2])
    )
    I2 = 0.5 * (I1 * I1 - trC2)
    jm43 = np.exp((-4.0 / 3.0) * logJ)
    FC = np.einsum("ib...,ba...->ia...", F, C)
    G2 = jm43 * (2.0 * (I1 * F - FC) - (4.0 / 3.0) * I2 * H)
    volumetric = material.lam * logJ * H
    if material.model == "mooney_rivlin":
        return material.param0 * G1 + material.param1 * G2 + volumetric
    if material.model == "yeoh":
        A = jm23 * I1 - 3.0
        scale = material.param0 + 2.0 * material.param1 * A + 3.0 * material.param2 * A * A
        return scale * G1 + volumetric
    if material.model == "gent":
        A = jm23 * I1 - 3.0
        denom = 1.0 - A / material.param0
        if np.any(denom <= 0.0):
            raise FloatingPointError(
                f"Gent chain limit exceeded: min denom={float(np.min(denom)):.6e}"
            )
        return (0.5 * material.mu / denom) * G1 + volumetric
    raise ValueError(f"unsupported material model: {material.model}")


def _state_to_F_np(U: np.ndarray) -> np.ndarray:
    F = np.empty((3, 3) + U.shape[1:], dtype=np.float64)
    F[0, 0], F[0, 1], F[0, 2] = U[3], U[4], U[5]
    F[1, 0], F[1, 1], F[1, 2] = U[6], U[7], U[8]
    F[2, 0], F[2, 1], F[2, 2] = U[9], U[10], U[11]
    return F


def _flux_np(U: np.ndarray, material: WarpHyperelasticMaterial3D, axis: int) -> np.ndarray:
    P = _first_piola_np(_state_to_F_np(U), material)
    out = np.zeros_like(U)
    out[0] = -P[0, axis]
    out[1] = -P[1, axis]
    out[2] = -P[2, axis]
    for i in range(3):
        out[3 + i * 3 + axis] = -U[i]
    return out


def _speed_tuple(lattice_speed: float | Sequence[float]) -> tuple[float, float, float]:
    if np.isscalar(lattice_speed):
        value = float(lattice_speed)
        return value, value, value
    values = tuple(float(value) for value in lattice_speed)
    if len(values) != 3:
        raise ValueError("lattice_speed must be a scalar or a length-three sequence")
    if min(values) <= 0.0:
        raise ValueError("all directional lattice speeds must be positive")
    return values


def _equilibrium_np(
    U: np.ndarray,
    material: WarpHyperelasticMaterial3D,
    lattice_speed: float | Sequence[float],
) -> np.ndarray:
    fluxes = [_flux_np(U, material, a) for a in range(3)]
    f = np.empty((Q, NCOMP) + U.shape[1:], dtype=np.float64)
    speeds = _speed_tuple(lattice_speed)
    for q, s in enumerate(_DIRS):
        directional_flux = sum(s[axis] * fluxes[axis] / speeds[axis] for axis in range(3))
        f[q] = (1.0 / 6.0) * (U + 3.0 * directional_flux)
    return f


def _source_np(U: np.ndarray, source_mode: int, source_params: np.ndarray) -> np.ndarray:
    out = np.zeros_like(U)
    params = np.asarray(source_params, dtype=np.float64)
    if source_mode == SOURCE_CONSTANT:
        out[:] = params[:NCOMP, None, None, None]
    elif source_mode == SOURCE_DAMPING:
        out[0] = -params[0] * U[0]
        out[1] = -params[0] * U[1]
        out[2] = -params[0] * U[2]
    elif source_mode != SOURCE_NONE:
        raise ValueError(f"unsupported source mode id: {source_mode}")
    return out


def _spatial_derivative(U: np.ndarray, spacing: float, axis: int, periodic: bool) -> np.ndarray:
    if periodic:
        return (np.roll(U, -1, axis=axis) - np.roll(U, 1, axis=axis)) / (2.0 * float(spacing))
    out = np.empty_like(U)
    n = U.shape[axis]
    if n < 3:
        out.fill(0.0)
        return out
    slc = [slice(None)] * U.ndim
    mid = list(slc)
    mid[axis] = slice(1, -1)
    p = list(slc)
    p[axis] = slice(2, None)
    m = list(slc)
    m[axis] = slice(None, -2)
    out[tuple(mid)] = (U[tuple(p)] - U[tuple(m)]) / (2.0 * float(spacing))
    s0 = list(slc)
    s1 = list(slc)
    s2 = list(slc)
    s0[axis] = 0
    s1[axis] = 1
    s2[axis] = 2
    out[tuple(s0)] = (-3.0 * U[tuple(s0)] + 4.0 * U[tuple(s1)] - U[tuple(s2)]) / (
        2.0 * float(spacing)
    )
    sn = list(slc)
    sn1 = list(slc)
    sn2 = list(slc)
    sn[axis] = -1
    sn1[axis] = -2
    sn2[axis] = -3
    out[tuple(sn)] = (3.0 * U[tuple(sn)] - 4.0 * U[tuple(sn1)] + U[tuple(sn2)]) / (
        2.0 * float(spacing)
    )
    return out


def _flux_jvp_np(
    U: np.ndarray, W: np.ndarray, material: WarpHyperelasticMaterial3D, axis: int
) -> np.ndarray:
    scale = float(np.max(np.abs(W)))
    if scale == 0.0 or not np.isfinite(scale):
        return np.zeros_like(U)
    eps = 1.0e-6 / max(1.0, scale)
    for _ in range(8):
        try:
            return (
                _flux_np(U + eps * W, material, axis) - _flux_np(U - eps * W, material, axis)
            ) / (2.0 * eps)
        except FloatingPointError:
            eps *= 0.25
    raise FloatingPointError("failed to compute finite-difference flux tangent")


def _second_order_populations_np(
    U: np.ndarray,
    material: WarpHyperelasticMaterial3D,
    lattice_speed: float | Sequence[float],
    *,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    periodic_x: bool,
    periodic_y: bool,
    periodic_z: bool,
    source_mode: int = SOURCE_NONE,
    source_params: np.ndarray | None = None,
) -> np.ndarray:
    if source_params is None:
        source_params = np.zeros(NCOMP, dtype=np.float64)
    Ux = _spatial_derivative(U, dx, axis=1, periodic=periodic_x)
    Uy = _spatial_derivative(U, dy, axis=2, periodic=periodic_y)
    Uz = _spatial_derivative(U, dz, axis=3, periodic=periodic_z)
    AxUx = _flux_jvp_np(U, Ux, material, axis=0)
    AyUy = _flux_jvp_np(U, Uy, material, axis=1)
    AzUz = _flux_jvp_np(U, Uz, material, axis=2)
    Ut = _source_np(U, source_mode, source_params) - (AxUx + AyUy + AzUz)
    feq = _equilibrium_np(U, material, lattice_speed)
    correction = np.empty_like(feq)
    speeds = _speed_tuple(lattice_speed)
    for q, s in enumerate(_DIRS):
        Wq = Ut + (s[0] * speeds[0] * Ux + s[1] * speeds[1] * Uy + s[2] * speeds[2] * Uz)
        AxW = _flux_jvp_np(U, Wq, material, axis=0)
        AyW = _flux_jvp_np(U, Wq, material, axis=1)
        AzW = _flux_jvp_np(U, Wq, material, axis=2)
        correction[q] = (1.0 / 6.0) * (
            Wq + 3.0 * (s[0] * AxW / speeds[0] + s[1] * AyW / speeds[1] + s[2] * AzW / speeds[2])
        )
    return feq - 0.5 * float(dt) * correction


def _flatten_pop(f: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(f.reshape((FQ,) + f.shape[2:]))


def _unflatten_pop(f: np.ndarray) -> np.ndarray:
    return f.reshape((Q, NCOMP) + f.shape[1:])


def _edge_derivative_along(values: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    if values.shape[axis] < 3:
        out.fill(0.0)
        return out
    slc = [slice(None)] * values.ndim
    mid = list(slc)
    mid[axis] = slice(1, -1)
    p = list(slc)
    p[axis] = slice(2, None)
    m = list(slc)
    m[axis] = slice(None, -2)
    out[tuple(mid)] = (values[tuple(p)] - values[tuple(m)]) / (2.0 * float(spacing))
    s0 = list(slc)
    s1 = list(slc)
    s2 = list(slc)
    s0[axis] = 0
    s1[axis] = 1
    s2[axis] = 2
    out[tuple(s0)] = (-3.0 * values[tuple(s0)] + 4.0 * values[tuple(s1)] - values[tuple(s2)]) / (
        2.0 * float(spacing)
    )
    sn = list(slc)
    sn1 = list(slc)
    sn2 = list(slc)
    sn[axis] = -1
    sn1[axis] = -2
    sn2[axis] = -3
    out[tuple(sn)] = (3.0 * values[tuple(sn)] - 4.0 * values[tuple(sn1)] + values[tuple(sn2)]) / (
        2.0 * float(spacing)
    )
    return out


def _enforce_dirichlet_tangent_columns(
    U: np.ndarray,
    displacement: np.ndarray,
    boundaries: BoxBoundarySet,
    *,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    out = np.array(U, dtype=np.float64, copy=True)
    u = np.asarray(displacement, dtype=np.float64)

    def set_face(
        face_u: np.ndarray,
        selectors: tuple,
        tangents: tuple[int, int],
        spacings: tuple[float, float],
    ) -> None:
        for loc_axis, B in enumerate(tangents):
            deriv = _edge_derivative_along(face_u, spacings[loc_axis], axis=loc_axis + 1)
            for i_comp in range(3):
                comp = 3 + i_comp * 3 + B
                target = deriv[i_comp] + (1.0 if i_comp == B else 0.0)
                out[(comp,) + selectors] = target

    if boundaries.side_kind[SIDE_LEFT] == BC_DIRICHLET:
        set_face(u[:, 0, :, :], (0, slice(None), slice(None)), (1, 2), (dy, dz))
    if boundaries.side_kind[SIDE_RIGHT] == BC_DIRICHLET:
        set_face(u[:, -1, :, :], (-1, slice(None), slice(None)), (1, 2), (dy, dz))
    if boundaries.side_kind[SIDE_BOTTOM] == BC_DIRICHLET:
        set_face(u[:, :, 0, :], (slice(None), 0, slice(None)), (0, 2), (dx, dz))
    if boundaries.side_kind[SIDE_TOP] == BC_DIRICHLET:
        set_face(u[:, :, -1, :], (slice(None), -1, slice(None)), (0, 2), (dx, dz))
    if boundaries.side_kind[SIDE_FRONT] == BC_DIRICHLET:
        set_face(u[:, :, :, 0], (slice(None), slice(None), 0), (0, 1), (dx, dy))
    if boundaries.side_kind[SIDE_BACK] == BC_DIRICHLET:
        set_face(u[:, :, :, -1], (slice(None), slice(None), -1), (0, 1), (dx, dy))
    return out


@wp.func
def _dir_x(q: int) -> int:
    if q == 0:
        return 1
    if q == 3:
        return -1
    return 0


@wp.func
def _dir_y(q: int) -> int:
    if q == 1:
        return 1
    if q == 4:
        return -1
    return 0


@wp.func
def _dir_z(q: int) -> int:
    if q == 2:
        return 1
    if q == 5:
        return -1
    return 0


@wp.func
def _clamp_j(j: wp.float64, floor: wp.float64) -> wp.float64:
    out = j
    if out < floor:
        out = floor
    return out


@wp.func
def _det3(
    F11: wp.float64,
    F12: wp.float64,
    F13: wp.float64,
    F21: wp.float64,
    F22: wp.float64,
    F23: wp.float64,
    F31: wp.float64,
    F32: wp.float64,
    F33: wp.float64,
) -> wp.float64:
    return (
        F11 * (F22 * F33 - F23 * F32)
        - F12 * (F21 * F33 - F23 * F31)
        + F13 * (F21 * F32 - F22 * F31)
    )


@wp.func
def _piola_mat(
    F11: wp.float64,
    F12: wp.float64,
    F13: wp.float64,
    F21: wp.float64,
    F22: wp.float64,
    F23: wp.float64,
    F31: wp.float64,
    F32: wp.float64,
    F33: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
) -> wp.mat33d:
    if model == MATERIAL_SVK:
        C00 = F11 * F11 + F21 * F21 + F31 * F31
        C01 = F11 * F12 + F21 * F22 + F31 * F32
        C02 = F11 * F13 + F21 * F23 + F31 * F33
        C11 = F12 * F12 + F22 * F22 + F32 * F32
        C12 = F12 * F13 + F22 * F23 + F32 * F33
        C22 = F13 * F13 + F23 * F23 + F33 * F33
        E00 = wp.float64(0.5) * (C00 - wp.float64(1.0))
        E01 = wp.float64(0.5) * C01
        E02 = wp.float64(0.5) * C02
        E11 = wp.float64(0.5) * (C11 - wp.float64(1.0))
        E12 = wp.float64(0.5) * C12
        E22 = wp.float64(0.5) * (C22 - wp.float64(1.0))
        trE = E00 + E11 + E22
        S00 = lam * trE + wp.float64(2.0) * mu * E00
        S01 = wp.float64(2.0) * mu * E01
        S02 = wp.float64(2.0) * mu * E02
        S11 = lam * trE + wp.float64(2.0) * mu * E11
        S12 = wp.float64(2.0) * mu * E12
        S22 = lam * trE + wp.float64(2.0) * mu * E22
        P11 = F11 * S00 + F12 * S01 + F13 * S02
        P12 = F11 * S01 + F12 * S11 + F13 * S12
        P13 = F11 * S02 + F12 * S12 + F13 * S22
        P21 = F21 * S00 + F22 * S01 + F23 * S02
        P22 = F21 * S01 + F22 * S11 + F23 * S12
        P23 = F21 * S02 + F22 * S12 + F23 * S22
        P31 = F31 * S00 + F32 * S01 + F33 * S02
        P32 = F31 * S01 + F32 * S11 + F33 * S12
        P33 = F31 * S02 + F32 * S12 + F33 * S22
        return wp.mat33d(P11, P12, P13, P21, P22, P23, P31, P32, P33)

    J = _clamp_j(_det3(F11, F12, F13, F21, F22, F23, F31, F32, F33), j_floor)
    C11c = F22 * F33 - F23 * F32
    C12c = F23 * F31 - F21 * F33
    C13c = F21 * F32 - F22 * F31
    C21c = F13 * F32 - F12 * F33
    C22c = F11 * F33 - F13 * F31
    C23c = F12 * F31 - F11 * F32
    C31c = F12 * F23 - F13 * F22
    C32c = F13 * F21 - F11 * F23
    C33c = F11 * F22 - F12 * F21
    H11 = C11c / J
    H12 = C12c / J
    H13 = C13c / J
    H21 = C21c / J
    H22 = C22c / J
    H23 = C23c / J
    H31 = C31c / J
    H32 = C32c / J
    H33 = C33c / J
    logJ = wp.log(J)

    if model == MATERIAL_NEO_HOOKE:
        volumetric = wp.float64(0.5) * lam * (J * J - wp.float64(1.0))
        return wp.mat33d(
            mu * (F11 - H11) + volumetric * H11,
            mu * (F12 - H12) + volumetric * H12,
            mu * (F13 - H13) + volumetric * H13,
            mu * (F21 - H21) + volumetric * H21,
            mu * (F22 - H22) + volumetric * H22,
            mu * (F23 - H23) + volumetric * H23,
            mu * (F31 - H31) + volumetric * H31,
            mu * (F32 - H32) + volumetric * H32,
            mu * (F33 - H33) + volumetric * H33,
        )
    if model == MATERIAL_LOG_NEO_HOOKE:
        volumetric = lam * logJ
        return wp.mat33d(
            mu * (F11 - H11) + volumetric * H11,
            mu * (F12 - H12) + volumetric * H12,
            mu * (F13 - H13) + volumetric * H13,
            mu * (F21 - H21) + volumetric * H21,
            mu * (F22 - H22) + volumetric * H22,
            mu * (F23 - H23) + volumetric * H23,
            mu * (F31 - H31) + volumetric * H31,
            mu * (F32 - H32) + volumetric * H32,
            mu * (F33 - H33) + volumetric * H33,
        )

    C00 = F11 * F11 + F21 * F21 + F31 * F31
    C01 = F11 * F12 + F21 * F22 + F31 * F32
    C02 = F11 * F13 + F21 * F23 + F31 * F33
    C11m = F12 * F12 + F22 * F22 + F32 * F32
    C12 = F12 * F13 + F22 * F23 + F32 * F33
    C22 = F13 * F13 + F23 * F23 + F33 * F33
    I1 = C00 + C11m + C22
    jm23 = wp.exp((-wp.float64(2.0) / wp.float64(3.0)) * logJ)

    G111 = jm23 * (wp.float64(2.0) * F11 - (wp.float64(2.0) / wp.float64(3.0)) * I1 * H11)
    G112 = jm23 * (wp.float64(2.0) * F12 - (wp.float64(2.0) / wp.float64(3.0)) * I1 * H12)
    G113 = jm23 * (wp.float64(2.0) * F13 - (wp.float64(2.0) / wp.float64(3.0)) * I1 * H13)
    G121 = jm23 * (wp.float64(2.0) * F21 - (wp.float64(2.0) / wp.float64(3.0)) * I1 * H21)
    G122 = jm23 * (wp.float64(2.0) * F22 - (wp.float64(2.0) / wp.float64(3.0)) * I1 * H22)
    G123 = jm23 * (wp.float64(2.0) * F23 - (wp.float64(2.0) / wp.float64(3.0)) * I1 * H23)
    G131 = jm23 * (wp.float64(2.0) * F31 - (wp.float64(2.0) / wp.float64(3.0)) * I1 * H31)
    G132 = jm23 * (wp.float64(2.0) * F32 - (wp.float64(2.0) / wp.float64(3.0)) * I1 * H32)
    G133 = jm23 * (wp.float64(2.0) * F33 - (wp.float64(2.0) / wp.float64(3.0)) * I1 * H33)

    if model == MATERIAL_SPH_NEO_HOOKE:
        # sph.pdf Eq. (13), with ``lam`` carrying the bulk modulus K.
        volumetric_sph = wp.float64(0.5) * lam * (J * J - wp.float64(1.0))
        scale_sph = wp.float64(0.5) * mu
        return wp.mat33d(
            scale_sph * G111 + volumetric_sph * H11,
            scale_sph * G112 + volumetric_sph * H12,
            scale_sph * G113 + volumetric_sph * H13,
            scale_sph * G121 + volumetric_sph * H21,
            scale_sph * G122 + volumetric_sph * H22,
            scale_sph * G123 + volumetric_sph * H23,
            scale_sph * G131 + volumetric_sph * H31,
            scale_sph * G132 + volumetric_sph * H32,
            scale_sph * G133 + volumetric_sph * H33,
        )

    trC2 = (
        C00 * C00 + C11m * C11m + C22 * C22 + wp.float64(2.0) * (C01 * C01 + C02 * C02 + C12 * C12)
    )
    I2 = wp.float64(0.5) * (I1 * I1 - trC2)
    jm43 = wp.exp((-wp.float64(4.0) / wp.float64(3.0)) * logJ)

    FC11 = F11 * C00 + F12 * C01 + F13 * C02
    FC12 = F11 * C01 + F12 * C11m + F13 * C12
    FC13 = F11 * C02 + F12 * C12 + F13 * C22
    FC21 = F21 * C00 + F22 * C01 + F23 * C02
    FC22 = F21 * C01 + F22 * C11m + F23 * C12
    FC23 = F21 * C02 + F22 * C12 + F23 * C22
    FC31 = F31 * C00 + F32 * C01 + F33 * C02
    FC32 = F31 * C01 + F32 * C11m + F33 * C12
    FC33 = F31 * C02 + F32 * C12 + F33 * C22

    volumetric = lam * logJ
    if model == MATERIAL_MOONEY_RIVLIN:
        G211 = jm43 * (
            wp.float64(2.0) * (I1 * F11 - FC11) - (wp.float64(4.0) / wp.float64(3.0)) * I2 * H11
        )
        G212 = jm43 * (
            wp.float64(2.0) * (I1 * F12 - FC12) - (wp.float64(4.0) / wp.float64(3.0)) * I2 * H12
        )
        G213 = jm43 * (
            wp.float64(2.0) * (I1 * F13 - FC13) - (wp.float64(4.0) / wp.float64(3.0)) * I2 * H13
        )
        G221 = jm43 * (
            wp.float64(2.0) * (I1 * F21 - FC21) - (wp.float64(4.0) / wp.float64(3.0)) * I2 * H21
        )
        G222 = jm43 * (
            wp.float64(2.0) * (I1 * F22 - FC22) - (wp.float64(4.0) / wp.float64(3.0)) * I2 * H22
        )
        G223 = jm43 * (
            wp.float64(2.0) * (I1 * F23 - FC23) - (wp.float64(4.0) / wp.float64(3.0)) * I2 * H23
        )
        G231 = jm43 * (
            wp.float64(2.0) * (I1 * F31 - FC31) - (wp.float64(4.0) / wp.float64(3.0)) * I2 * H31
        )
        G232 = jm43 * (
            wp.float64(2.0) * (I1 * F32 - FC32) - (wp.float64(4.0) / wp.float64(3.0)) * I2 * H32
        )
        G233 = jm43 * (
            wp.float64(2.0) * (I1 * F33 - FC33) - (wp.float64(4.0) / wp.float64(3.0)) * I2 * H33
        )
        return wp.mat33d(
            mat_p0 * G111 + mat_p1 * G211 + volumetric * H11,
            mat_p0 * G112 + mat_p1 * G212 + volumetric * H12,
            mat_p0 * G113 + mat_p1 * G213 + volumetric * H13,
            mat_p0 * G121 + mat_p1 * G221 + volumetric * H21,
            mat_p0 * G122 + mat_p1 * G222 + volumetric * H22,
            mat_p0 * G123 + mat_p1 * G223 + volumetric * H23,
            mat_p0 * G131 + mat_p1 * G231 + volumetric * H31,
            mat_p0 * G132 + mat_p1 * G232 + volumetric * H32,
            mat_p0 * G133 + mat_p1 * G233 + volumetric * H33,
        )
    if model == MATERIAL_YEOH:
        A = jm23 * I1 - wp.float64(3.0)
        scale = mat_p0 + wp.float64(2.0) * mat_p1 * A + wp.float64(3.0) * mat_p2 * A * A
        return wp.mat33d(
            scale * G111 + volumetric * H11,
            scale * G112 + volumetric * H12,
            scale * G113 + volumetric * H13,
            scale * G121 + volumetric * H21,
            scale * G122 + volumetric * H22,
            scale * G123 + volumetric * H23,
            scale * G131 + volumetric * H31,
            scale * G132 + volumetric * H32,
            scale * G133 + volumetric * H33,
        )
    if model == MATERIAL_GENT:
        A = jm23 * I1 - wp.float64(3.0)
        denom = wp.float64(1.0) - A / mat_p0
        if denom < j_floor:
            denom = j_floor
        scale = wp.float64(0.5) * mu / denom
        return wp.mat33d(
            scale * G111 + volumetric * H11,
            scale * G112 + volumetric * H12,
            scale * G113 + volumetric * H13,
            scale * G121 + volumetric * H21,
            scale * G122 + volumetric * H22,
            scale * G123 + volumetric * H23,
            scale * G131 + volumetric * H31,
            scale * G132 + volumetric * H32,
            scale * G133 + volumetric * H33,
        )
    return wp.mat33d(
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


@wp.func
def _state_comp(
    a: int,
    vx: wp.float64,
    vy: wp.float64,
    vz: wp.float64,
    F11: wp.float64,
    F12: wp.float64,
    F13: wp.float64,
    F21: wp.float64,
    F22: wp.float64,
    F23: wp.float64,
    F31: wp.float64,
    F32: wp.float64,
    F33: wp.float64,
) -> wp.float64:
    if a == 0:
        return vx
    if a == 1:
        return vy
    if a == 2:
        return vz
    if a == 3:
        return F11
    if a == 4:
        return F12
    if a == 5:
        return F13
    if a == 6:
        return F21
    if a == 7:
        return F22
    if a == 8:
        return F23
    if a == 9:
        return F31
    if a == 10:
        return F32
    return F33


@wp.func
def _flux_comp(
    axis: int,
    a: int,
    vx: wp.float64,
    vy: wp.float64,
    vz: wp.float64,
    F11: wp.float64,
    F12: wp.float64,
    F13: wp.float64,
    F21: wp.float64,
    F22: wp.float64,
    F23: wp.float64,
    F31: wp.float64,
    F32: wp.float64,
    F33: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
) -> wp.float64:
    P = _piola_mat(
        F11, F12, F13, F21, F22, F23, F31, F32, F33, model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor
    )
    if axis == 0:
        if a == 0:
            return -P[0, 0]
        if a == 1:
            return -P[1, 0]
        if a == 2:
            return -P[2, 0]
        if a == 3:
            return -vx
        if a == 6:
            return -vy
        if a == 9:
            return -vz
        return wp.float64(0.0)
    if axis == 1:
        if a == 0:
            return -P[0, 1]
        if a == 1:
            return -P[1, 1]
        if a == 2:
            return -P[2, 1]
        if a == 4:
            return -vx
        if a == 7:
            return -vy
        if a == 10:
            return -vz
        return wp.float64(0.0)
    if a == 0:
        return -P[0, 2]
    if a == 1:
        return -P[1, 2]
    if a == 2:
        return -P[2, 2]
    if a == 5:
        return -vx
    if a == 8:
        return -vy
    if a == 11:
        return -vz
    return wp.float64(0.0)


@wp.func
def _equilibrium_comp(
    q: int,
    a: int,
    vx: wp.float64,
    vy: wp.float64,
    vz: wp.float64,
    F11: wp.float64,
    F12: wp.float64,
    F13: wp.float64,
    F21: wp.float64,
    F22: wp.float64,
    F23: wp.float64,
    F31: wp.float64,
    F32: wp.float64,
    F33: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    lattice_speed: wp.float64,
    j_floor: wp.float64,
) -> wp.float64:
    sx = wp.float64(_dir_x(q))
    sy = wp.float64(_dir_y(q))
    sz = wp.float64(_dir_z(q))
    val = _state_comp(a, vx, vy, vz, F11, F12, F13, F21, F22, F23, F31, F32, F33)
    phix = _flux_comp(
        0,
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
        j_floor,
    )
    phiy = _flux_comp(
        1,
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
        j_floor,
    )
    phiz = _flux_comp(
        2,
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
        j_floor,
    )
    return (wp.float64(1.0) / wp.float64(6.0)) * (
        val + (wp.float64(3.0) / lattice_speed) * (sx * phix + sy * phiy + sz * phiz)
    )


@wp.func
def _equilibrium_anisotropic_comp(
    q: int,
    a: int,
    vx: wp.float64,
    vy: wp.float64,
    vz: wp.float64,
    F11: wp.float64,
    F12: wp.float64,
    F13: wp.float64,
    F21: wp.float64,
    F22: wp.float64,
    F23: wp.float64,
    F31: wp.float64,
    F32: wp.float64,
    F33: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    lattice_speed_x: wp.float64,
    lattice_speed_y: wp.float64,
    lattice_speed_z: wp.float64,
    j_floor: wp.float64,
) -> wp.float64:
    if lattice_speed_x == lattice_speed_y and lattice_speed_x == lattice_speed_z:
        return _equilibrium_comp(
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
            lattice_speed_x,
            j_floor,
        )
    sx = wp.float64(_dir_x(q))
    sy = wp.float64(_dir_y(q))
    sz = wp.float64(_dir_z(q))
    val = _state_comp(a, vx, vy, vz, F11, F12, F13, F21, F22, F23, F31, F32, F33)
    phix = _flux_comp(
        0,
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
        j_floor,
    )
    phiy = _flux_comp(
        1,
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
        j_floor,
    )
    phiz = _flux_comp(
        2,
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
        j_floor,
    )
    return (wp.float64(1.0) / wp.float64(6.0)) * (
        val
        + wp.float64(3.0)
        * (sx * phix / lattice_speed_x + sy * phiy / lattice_speed_y + sz * phiz / lattice_speed_z)
    )


@wp.func
def _source_comp(
    a: int,
    mode: int,
    params: wp.array(dtype=wp.float64),
    vx: wp.float64,
    vy: wp.float64,
    vz: wp.float64,
) -> wp.float64:
    if mode == SOURCE_CONSTANT:
        return params[a]
    if mode == SOURCE_DAMPING:
        if a == 0:
            return -params[0] * vx
        if a == 1:
            return -params[0] * vy
        if a == 2:
            return -params[0] * vz
    return wp.float64(0.0)


@wp.func
def _boundary_value_comp(
    comp: int,
    mode: int,
    params: wp.array2d(dtype=wp.float64),
    side: int,
    x: wp.float64,
    y: wp.float64,
    z: wp.float64,
    t: wp.float64,
) -> wp.float64:
    if mode == VALUE_ZERO:
        return wp.float64(0.0)
    if mode == VALUE_CONSTANT:
        if comp == 0:
            return params[side, 0]
        if comp == 1:
            return params[side, 1]
        return params[side, 2]
    if mode == VALUE_RAMP:
        ramp_time = params[side, 3]
        scale = wp.float64(1.0)
        if ramp_time > wp.float64(0.0) and t < ramp_time:
            s = wp.sin(PI * t / ramp_time)
            scale = s * s
        if comp == 0:
            return params[side, 0] * scale
        if comp == 1:
            return params[side, 1] * scale
        return params[side, 2] * scale
    if mode == VALUE_SINE2_HOLD:
        active_time = params[side, 3]
        denom = params[side, 4]
        scale = wp.float64(1.0)
        if active_time > wp.float64(0.0) and t < active_time:
            if denom <= wp.float64(0.0):
                denom = wp.float64(2.0) * active_time
            s = wp.sin(PI * t / denom)
            scale = s * s
        if comp == 0:
            return params[side, 0] * scale
        if comp == 1:
            return params[side, 1] * scale
        return params[side, 2] * scale
    if mode == VALUE_AFFINE_VELOCITY:
        active_time = params[side, 12]
        if active_time > wp.float64(0.0) and t > active_time:
            return wp.float64(0.0)
        if comp == 0:
            return params[side, 0] * x + params[side, 1] * y + params[side, 2] * z + params[side, 9]
        if comp == 1:
            return (
                params[side, 3] * x + params[side, 4] * y + params[side, 5] * z + params[side, 10]
            )
        return params[side, 6] * x + params[side, 7] * y + params[side, 8] * z + params[side, 11]
    if mode == VALUE_SHEAR_VELOCITY_X:
        if comp == 0 and t < params[side, 1]:
            return params[side, 0] * wp.sin(PI * t / params[side, 1])
        return wp.float64(0.0)
    return wp.float64(0.0)


@wp.func
def _normal_residual_from_col(
    F11_in: wp.float64,
    F12_in: wp.float64,
    F13_in: wp.float64,
    F21_in: wp.float64,
    F22_in: wp.float64,
    F23_in: wp.float64,
    F31_in: wp.float64,
    F32_in: wp.float64,
    F33_in: wp.float64,
    normal_axis: int,
    x0: wp.float64,
    x1: wp.float64,
    x2: wp.float64,
    sign: wp.float64,
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
) -> wp.vec3d:
    F11 = F11_in
    F12 = F12_in
    F13 = F13_in
    F21 = F21_in
    F22 = F22_in
    F23 = F23_in
    F31 = F31_in
    F32 = F32_in
    F33 = F33_in
    if normal_axis == 0:
        F11 = x0
        F21 = x1
        F31 = x2
    elif normal_axis == 1:
        F12 = x0
        F22 = x1
        F32 = x2
    else:
        F13 = x0
        F23 = x1
        F33 = x2
    P = _piola_mat(
        F11, F12, F13, F21, F22, F23, F31, F32, F33, model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor
    )
    return wp.vec3d(
        sign * P[0, normal_axis] - tx, sign * P[1, normal_axis] - ty, sign * P[2, normal_axis] - tz
    )


@wp.func
def _det_from_col(
    F11_in: wp.float64,
    F12_in: wp.float64,
    F13_in: wp.float64,
    F21_in: wp.float64,
    F22_in: wp.float64,
    F23_in: wp.float64,
    F31_in: wp.float64,
    F32_in: wp.float64,
    F33_in: wp.float64,
    normal_axis: int,
    x0: wp.float64,
    x1: wp.float64,
    x2: wp.float64,
) -> wp.float64:
    F11 = F11_in
    F12 = F12_in
    F13 = F13_in
    F21 = F21_in
    F22 = F22_in
    F23 = F23_in
    F31 = F31_in
    F32 = F32_in
    F33 = F33_in
    if normal_axis == 0:
        F11 = x0
        F21 = x1
        F31 = x2
    elif normal_axis == 1:
        F12 = x0
        F22 = x1
        F32 = x2
    else:
        F13 = x0
        F23 = x1
        F33 = x2
    return _det3(F11, F12, F13, F21, F22, F23, F31, F32, F33)


@wp.func
def _solve_normal_column(
    F11_in: wp.float64,
    F12_in: wp.float64,
    F13_in: wp.float64,
    F21_in: wp.float64,
    F22_in: wp.float64,
    F23_in: wp.float64,
    F31_in: wp.float64,
    F32_in: wp.float64,
    F33_in: wp.float64,
    normal_axis: int,
    sign: wp.float64,
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
) -> wp.mat33d:
    x0 = F11_in
    x1 = F21_in
    x2 = F31_in
    if normal_axis == 1:
        x0 = F12_in
        x1 = F22_in
        x2 = F32_in
    elif normal_axis == 2:
        x0 = F13_in
        x1 = F23_in
        x2 = F33_in
    eps = wp.float64(1.0e-6)
    for _it in range(16):
        r = _normal_residual_from_col(
            F11_in,
            F12_in,
            F13_in,
            F21_in,
            F22_in,
            F23_in,
            F31_in,
            F32_in,
            F33_in,
            normal_axis,
            x0,
            x1,
            x2,
            sign,
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
        )
        rnorm = wp.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
        if rnorm < wp.float64(1.0e-10):
            break
        rp0 = _normal_residual_from_col(
            F11_in,
            F12_in,
            F13_in,
            F21_in,
            F22_in,
            F23_in,
            F31_in,
            F32_in,
            F33_in,
            normal_axis,
            x0 + eps,
            x1,
            x2,
            sign,
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
        )
        rp1 = _normal_residual_from_col(
            F11_in,
            F12_in,
            F13_in,
            F21_in,
            F22_in,
            F23_in,
            F31_in,
            F32_in,
            F33_in,
            normal_axis,
            x0,
            x1 + eps,
            x2,
            sign,
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
        )
        rp2 = _normal_residual_from_col(
            F11_in,
            F12_in,
            F13_in,
            F21_in,
            F22_in,
            F23_in,
            F31_in,
            F32_in,
            F33_in,
            normal_axis,
            x0,
            x1,
            x2 + eps,
            sign,
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
        )
        a = (rp0[0] - r[0]) / eps
        b = (rp1[0] - r[0]) / eps
        c = (rp2[0] - r[0]) / eps
        d = (rp0[1] - r[1]) / eps
        e = (rp1[1] - r[1]) / eps
        f = (rp2[1] - r[1]) / eps
        g = (rp0[2] - r[2]) / eps
        h = (rp1[2] - r[2]) / eps
        ii = (rp2[2] - r[2]) / eps
        det = a * (e * ii - f * h) - b * (d * ii - f * g) + c * (d * h - e * g)
        if wp.abs(det) < wp.float64(1.0e-14):
            break
        inv00 = (e * ii - f * h) / det
        inv01 = (c * h - b * ii) / det
        inv02 = (b * f - c * e) / det
        inv10 = (f * g - d * ii) / det
        inv11 = (a * ii - c * g) / det
        inv12 = (c * d - a * f) / det
        inv20 = (d * h - e * g) / det
        inv21 = (b * g - a * h) / det
        inv22 = (a * e - b * d) / det
        delta0 = -(inv00 * r[0] + inv01 * r[1] + inv02 * r[2])
        delta1 = -(inv10 * r[0] + inv11 * r[1] + inv12 * r[2])
        delta2 = -(inv20 * r[0] + inv21 * r[1] + inv22 * r[2])
        step = wp.float64(1.0)
        for _ls in range(12):
            c0 = x0 + step * delta0
            c1 = x1 + step * delta1
            c2 = x2 + step * delta2
            J = _det_from_col(
                F11_in,
                F12_in,
                F13_in,
                F21_in,
                F22_in,
                F23_in,
                F31_in,
                F32_in,
                F33_in,
                normal_axis,
                c0,
                c1,
                c2,
            )
            if J > j_floor:
                x0 = c0
                x1 = c1
                x2 = c2
                break
            step = wp.float64(0.5) * step
    F11 = F11_in
    F12 = F12_in
    F13 = F13_in
    F21 = F21_in
    F22 = F22_in
    F23 = F23_in
    F31 = F31_in
    F32 = F32_in
    F33 = F33_in
    if normal_axis == 0:
        F11 = x0
        F21 = x1
        F31 = x2
    elif normal_axis == 1:
        F12 = x0
        F22 = x1
        F32 = x2
    else:
        F13 = x0
        F23 = x1
        F33 = x2
    return wp.mat33d(F11, F12, F13, F21, F22, F23, F31, F32, F33)


@wp.kernel
def _reset_invalid_kernel(invalid: wp.array(dtype=wp.int32)):
    invalid[0] = wp.int32(0)


@wp.kernel
def _clear_kernel(arr: wp.array4d(dtype=wp.float64)):
    a, i, j, k = wp.tid()
    arr[a, i, j, k] = wp.float64(0.0)


@wp.kernel
def _set_u_star_from_displacement_kernel(
    displacement: wp.array4d(dtype=wp.float64),
    U: wp.array4d(dtype=wp.float64),
    u_star: wp.array4d(dtype=wp.float64),
    dt: wp.float64,
):
    i, j, k = wp.tid()
    u_star[0, i, j, k] = displacement[0, i, j, k] - wp.float64(0.5) * dt * U[0, i, j, k]
    u_star[1, i, j, k] = displacement[1, i, j, k] - wp.float64(0.5) * dt * U[1, i, j, k]
    u_star[2, i, j, k] = displacement[2, i, j, k] - wp.float64(0.5) * dt * U[2, i, j, k]


@wp.kernel
def _collide_kernel(
    f0: wp.array4d(dtype=wp.float64),
    fpost: wp.array4d(dtype=wp.float64),
    u: wp.array4d(dtype=wp.float64),
    u_star: wp.array4d(dtype=wp.float64),
    U: wp.array4d(dtype=wp.float64),
    P_field: wp.array4d(dtype=wp.float64),
    sigma: wp.array4d(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
    source_params: wp.array(dtype=wp.float64),
    dt: wp.float64,
    omega: wp.float64,
    lattice_speed_x: wp.float64,
    lattice_speed_y: wp.float64,
    lattice_speed_z: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
    source_mode: int,
):
    i, j, k = wp.tid()
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
        vx += f0[q * 12 + 0, i, j, k]
        vy += f0[q * 12 + 1, i, j, k]
        vz += f0[q * 12 + 2, i, j, k]
        F11 += f0[q * 12 + 3, i, j, k]
        F12 += f0[q * 12 + 4, i, j, k]
        F13 += f0[q * 12 + 5, i, j, k]
        F21 += f0[q * 12 + 6, i, j, k]
        F22 += f0[q * 12 + 7, i, j, k]
        F23 += f0[q * 12 + 8, i, j, k]
        F31 += f0[q * 12 + 9, i, j, k]
        F32 += f0[q * 12 + 10, i, j, k]
        F33 += f0[q * 12 + 11, i, j, k]
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

    U[0, i, j, k] = vx
    U[1, i, j, k] = vy
    U[2, i, j, k] = vz
    U[3, i, j, k] = F11
    U[4, i, j, k] = F12
    U[5, i, j, k] = F13
    U[6, i, j, k] = F21
    U[7, i, j, k] = F22
    U[8, i, j, k] = F23
    U[9, i, j, k] = F31
    U[10, i, j, k] = F32
    U[11, i, j, k] = F33

    ux = u_star[0, i, j, k] + half_dt * vx
    uy = u_star[1, i, j, k] + half_dt * vy
    uz = u_star[2, i, j, k] + half_dt * vz
    u[0, i, j, k] = ux
    u[1, i, j, k] = uy
    u[2, i, j, k] = uz
    u_star[0, i, j, k] = ux + half_dt * vx
    u_star[1, i, j, k] = uy + half_dt * vy
    u_star[2, i, j, k] = uz + half_dt * vz

    J = _det3(F11, F12, F13, F21, F22, F23, F31, F32, F33)
    if J <= j_floor:
        invalid[0] = wp.int32(1)
    J_safe = _clamp_j(J, j_floor)
    P = _piola_mat(
        F11, F12, F13, F21, F22, F23, F31, F32, F33, model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor
    )
    for r in range(3):
        for ccol in range(3):
            P_field[r * 3 + ccol, i, j, k] = P[r, ccol]
    sigma[0, i, j, k] = (P[0, 0] * F11 + P[0, 1] * F12 + P[0, 2] * F13) / J_safe
    sigma[1, i, j, k] = (P[0, 0] * F21 + P[0, 1] * F22 + P[0, 2] * F23) / J_safe
    sigma[2, i, j, k] = (P[0, 0] * F31 + P[0, 1] * F32 + P[0, 2] * F33) / J_safe
    sigma[3, i, j, k] = (P[1, 0] * F11 + P[1, 1] * F12 + P[1, 2] * F13) / J_safe
    sigma[4, i, j, k] = (P[1, 0] * F21 + P[1, 1] * F22 + P[1, 2] * F23) / J_safe
    sigma[5, i, j, k] = (P[1, 0] * F31 + P[1, 1] * F32 + P[1, 2] * F33) / J_safe
    sigma[6, i, j, k] = (P[2, 0] * F11 + P[2, 1] * F12 + P[2, 2] * F13) / J_safe
    sigma[7, i, j, k] = (P[2, 0] * F21 + P[2, 1] * F22 + P[2, 2] * F23) / J_safe
    sigma[8, i, j, k] = (P[2, 0] * F31 + P[2, 1] * F32 + P[2, 2] * F33) / J_safe

    for q in range(6):
        for a in range(12):
            feq = _equilibrium_anisotropic_comp(
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
                lattice_speed_x,
                lattice_speed_y,
                lattice_speed_z,
                j_floor,
            )
            old = f0[q * 12 + a, i, j, k]
            source_term = _source_comp(a, source_mode, source_params, vx, vy, vz)
            fpost[q * 12 + a, i, j, k] = (
                old
                + omega * (feq - old)
                + dt
                * (wp.float64(2.0) - omega)
                * (wp.float64(1.0) / wp.float64(12.0))
                * source_term
            )


@wp.kernel
def _stream_kernel(
    fpost: wp.array4d(dtype=wp.float64),
    f1: wp.array4d(dtype=wp.float64),
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
):
    i, j, k = wp.tid()
    for q in range(6):
        ii = i + _dir_x(q)
        jj = j + _dir_y(q)
        kk = k + _dir_z(q)
        valid = True
        if ii >= nx:
            if periodic_x != 0:
                ii = 0
            else:
                valid = False
        elif ii < 0:
            if periodic_x != 0:
                ii = nx - 1
            else:
                valid = False
        if jj >= ny:
            if periodic_y != 0:
                jj = 0
            else:
                valid = False
        elif jj < 0:
            if periodic_y != 0:
                jj = ny - 1
            else:
                valid = False
        if kk >= nz:
            if periodic_z != 0:
                kk = 0
            else:
                valid = False
        elif kk < 0:
            if periodic_z != 0:
                kk = nz - 1
            else:
                valid = False
        if valid:
            for a in range(12):
                f1[q * 12 + a, ii, jj, kk] = fpost[q * 12 + a, i, j, k]


@wp.kernel
def _apply_boundary_side_kernel(
    fpost: wp.array4d(dtype=wp.float64),
    f1: wp.array4d(dtype=wp.float64),
    U: wp.array4d(dtype=wp.float64),
    side_kind: wp.array(dtype=wp.int32),
    value_mode: wp.array(dtype=wp.int32),
    value_params: wp.array2d(dtype=wp.float64),
    nx: int,
    ny: int,
    nz: int,
    length_x: wp.float64,
    length_y: wp.float64,
    length_z: wp.float64,
    time_mid: wp.float64,
    lattice_speed_x: wp.float64,
    lattice_speed_y: wp.float64,
    lattice_speed_z: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
    side: int,
):
    s = wp.tid()
    if side_kind[side] == BC_PERIODIC:
        return
    n_nodes = ny * nz
    if side == SIDE_BOTTOM or side == SIDE_TOP:
        n_nodes = nx * nz
    elif side == SIDE_FRONT or side == SIDE_BACK:
        n_nodes = nx * ny
    if s >= n_nodes:
        return

    i = wp.int32(0)
    j = wp.int32(0)
    k = wp.int32(0)
    n0 = wp.int32(-1)
    n1 = wp.int32(0)
    n2 = wp.int32(0)
    q_in = wp.int32(0)
    q_out = wp.int32(3)
    normal_axis = wp.int32(0)
    sign = wp.float64(-1.0)
    if side == SIDE_LEFT:
        j = s % ny
        k = s // ny
    elif side == SIDE_RIGHT:
        i = nx - 1
        j = s % ny
        k = s // ny
        n0 = 1
        q_in = 3
        q_out = 0
        sign = wp.float64(1.0)
    elif side == SIDE_BOTTOM:
        i = s % nx
        k = s // nx
        n0 = 0
        n1 = -1
        n2 = 0
        q_in = 1
        q_out = 4
        normal_axis = 1
        sign = wp.float64(-1.0)
    elif side == SIDE_TOP:
        i = s % nx
        j = ny - 1
        k = s // nx
        n0 = 0
        n1 = 1
        n2 = 0
        q_in = 4
        q_out = 1
        normal_axis = 1
        sign = wp.float64(1.0)
    elif side == SIDE_FRONT:
        i = s % nx
        j = s // nx
        n0 = 0
        n1 = 0
        n2 = -1
        q_in = 2
        q_out = 5
        normal_axis = 2
        sign = wp.float64(-1.0)
    else:
        i = s % nx
        j = s // nx
        k = nz - 1
        n0 = 0
        n1 = 0
        n2 = 1
        q_in = 5
        q_out = 2
        normal_axis = 2
        sign = wp.float64(1.0)

    dx = length_x / wp.float64(nx)
    dy = length_y / wp.float64(ny)
    dz = length_z / wp.float64(nz)
    x = (wp.float64(i) + wp.float64(0.5)) * dx
    y = (wp.float64(j) + wp.float64(0.5)) * dy
    z = (wp.float64(k) + wp.float64(0.5)) * dz
    x_bc = x
    y_bc = y
    z_bc = z
    if side == SIDE_LEFT:
        x_bc = wp.float64(0.0)
    elif side == SIDE_RIGHT:
        x_bc = length_x
    elif side == SIDE_BOTTOM:
        y_bc = wp.float64(0.0)
    elif side == SIDE_TOP:
        y_bc = length_y
    elif side == SIDE_FRONT:
        z_bc = wp.float64(0.0)
    elif side == SIDE_BACK:
        z_bc = length_z

    mode = value_mode[side]
    normal_speed = lattice_speed_x
    if normal_axis == 1:
        normal_speed = lattice_speed_y
    elif normal_axis == 2:
        normal_speed = lattice_speed_z
    if side_kind[side] == BC_DIRICHLET:
        vdx = _boundary_value_comp(0, mode, value_params, side, x_bc, y_bc, z_bc, time_mid)
        vdy = _boundary_value_comp(1, mode, value_params, side, x_bc, y_bc, z_bc, time_mid)
        vdz = _boundary_value_comp(2, mode, value_params, side, x_bc, y_bc, z_bc, time_mid)
        f1[q_in * 12 + 0, i, j, k] = -fpost[q_out * 12 + 0, i, j, k] + vdx / wp.float64(3.0)
        f1[q_in * 12 + 1, i, j, k] = -fpost[q_out * 12 + 1, i, j, k] + vdy / wp.float64(3.0)
        f1[q_in * 12 + 2, i, j, k] = -fpost[q_out * 12 + 2, i, j, k] + vdz / wp.float64(3.0)
        f1[q_in * 12 + 3, i, j, k] = (
            fpost[q_out * 12 + 3, i, j, k] + wp.float64(n0) * vdx / normal_speed
        )
        f1[q_in * 12 + 4, i, j, k] = (
            fpost[q_out * 12 + 4, i, j, k] + wp.float64(n1) * vdx / normal_speed
        )
        f1[q_in * 12 + 5, i, j, k] = (
            fpost[q_out * 12 + 5, i, j, k] + wp.float64(n2) * vdx / normal_speed
        )
        f1[q_in * 12 + 6, i, j, k] = (
            fpost[q_out * 12 + 6, i, j, k] + wp.float64(n0) * vdy / normal_speed
        )
        f1[q_in * 12 + 7, i, j, k] = (
            fpost[q_out * 12 + 7, i, j, k] + wp.float64(n1) * vdy / normal_speed
        )
        f1[q_in * 12 + 8, i, j, k] = (
            fpost[q_out * 12 + 8, i, j, k] + wp.float64(n2) * vdy / normal_speed
        )
        f1[q_in * 12 + 9, i, j, k] = (
            fpost[q_out * 12 + 9, i, j, k] + wp.float64(n0) * vdz / normal_speed
        )
        f1[q_in * 12 + 10, i, j, k] = (
            fpost[q_out * 12 + 10, i, j, k] + wp.float64(n1) * vdz / normal_speed
        )
        f1[q_in * 12 + 11, i, j, k] = (
            fpost[q_out * 12 + 11, i, j, k] + wp.float64(n2) * vdz / normal_speed
        )
    elif side_kind[side] == BC_NEUMANN:
        tx = _boundary_value_comp(0, mode, value_params, side, x_bc, y_bc, z_bc, time_mid)
        ty = _boundary_value_comp(1, mode, value_params, side, x_bc, y_bc, z_bc, time_mid)
        tz = _boundary_value_comp(2, mode, value_params, side, x_bc, y_bc, z_bc, time_mid)
        Fb = _solve_normal_column(
            U[3, i, j, k],
            U[4, i, j, k],
            U[5, i, j, k],
            U[6, i, j, k],
            U[7, i, j, k],
            U[8, i, j, k],
            U[9, i, j, k],
            U[10, i, j, k],
            U[11, i, j, k],
            normal_axis,
            sign,
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
        )
        f1[q_in * 12 + 0, i, j, k] = fpost[q_out * 12 + 0, i, j, k] + tx / normal_speed
        f1[q_in * 12 + 1, i, j, k] = fpost[q_out * 12 + 1, i, j, k] + ty / normal_speed
        f1[q_in * 12 + 2, i, j, k] = fpost[q_out * 12 + 2, i, j, k] + tz / normal_speed
        f1[q_in * 12 + 3, i, j, k] = -fpost[q_out * 12 + 3, i, j, k] + Fb[0, 0] / wp.float64(3.0)
        f1[q_in * 12 + 4, i, j, k] = -fpost[q_out * 12 + 4, i, j, k] + Fb[0, 1] / wp.float64(3.0)
        f1[q_in * 12 + 5, i, j, k] = -fpost[q_out * 12 + 5, i, j, k] + Fb[0, 2] / wp.float64(3.0)
        f1[q_in * 12 + 6, i, j, k] = -fpost[q_out * 12 + 6, i, j, k] + Fb[1, 0] / wp.float64(3.0)
        f1[q_in * 12 + 7, i, j, k] = -fpost[q_out * 12 + 7, i, j, k] + Fb[1, 1] / wp.float64(3.0)
        f1[q_in * 12 + 8, i, j, k] = -fpost[q_out * 12 + 8, i, j, k] + Fb[1, 2] / wp.float64(3.0)
        f1[q_in * 12 + 9, i, j, k] = -fpost[q_out * 12 + 9, i, j, k] + Fb[2, 0] / wp.float64(3.0)
        f1[q_in * 12 + 10, i, j, k] = -fpost[q_out * 12 + 10, i, j, k] + Fb[2, 1] / wp.float64(3.0)
        f1[q_in * 12 + 11, i, j, k] = -fpost[q_out * 12 + 11, i, j, k] + Fb[2, 2] / wp.float64(3.0)


@wp.kernel
def _refresh_kernel(
    f0: wp.array4d(dtype=wp.float64),
    u: wp.array4d(dtype=wp.float64),
    u_star: wp.array4d(dtype=wp.float64),
    U: wp.array4d(dtype=wp.float64),
    P_field: wp.array4d(dtype=wp.float64),
    sigma: wp.array4d(dtype=wp.float64),
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
    i, j, k = wp.tid()
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
        vx += f0[q * 12 + 0, i, j, k]
        vy += f0[q * 12 + 1, i, j, k]
        vz += f0[q * 12 + 2, i, j, k]
        F11 += f0[q * 12 + 3, i, j, k]
        F12 += f0[q * 12 + 4, i, j, k]
        F13 += f0[q * 12 + 5, i, j, k]
        F21 += f0[q * 12 + 6, i, j, k]
        F22 += f0[q * 12 + 7, i, j, k]
        F23 += f0[q * 12 + 8, i, j, k]
        F31 += f0[q * 12 + 9, i, j, k]
        F32 += f0[q * 12 + 10, i, j, k]
        F33 += f0[q * 12 + 11, i, j, k]
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

    U[0, i, j, k] = vx
    U[1, i, j, k] = vy
    U[2, i, j, k] = vz
    U[3, i, j, k] = F11
    U[4, i, j, k] = F12
    U[5, i, j, k] = F13
    U[6, i, j, k] = F21
    U[7, i, j, k] = F22
    U[8, i, j, k] = F23
    U[9, i, j, k] = F31
    U[10, i, j, k] = F32
    U[11, i, j, k] = F33
    u[0, i, j, k] = u_star[0, i, j, k] + half_dt * vx
    u[1, i, j, k] = u_star[1, i, j, k] + half_dt * vy
    u[2, i, j, k] = u_star[2, i, j, k] + half_dt * vz
    J = _det3(F11, F12, F13, F21, F22, F23, F31, F32, F33)
    if J <= j_floor:
        invalid[0] = wp.int32(1)
    J_safe = _clamp_j(J, j_floor)
    P = _piola_mat(
        F11, F12, F13, F21, F22, F23, F31, F32, F33, model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor
    )
    for r in range(3):
        for ccol in range(3):
            P_field[r * 3 + ccol, i, j, k] = P[r, ccol]
    sigma[0, i, j, k] = (P[0, 0] * F11 + P[0, 1] * F12 + P[0, 2] * F13) / J_safe
    sigma[1, i, j, k] = (P[0, 0] * F21 + P[0, 1] * F22 + P[0, 2] * F23) / J_safe
    sigma[2, i, j, k] = (P[0, 0] * F31 + P[0, 1] * F32 + P[0, 2] * F33) / J_safe
    sigma[3, i, j, k] = (P[1, 0] * F11 + P[1, 1] * F12 + P[1, 2] * F13) / J_safe
    sigma[4, i, j, k] = (P[1, 0] * F21 + P[1, 1] * F22 + P[1, 2] * F23) / J_safe
    sigma[5, i, j, k] = (P[1, 0] * F31 + P[1, 1] * F32 + P[1, 2] * F33) / J_safe
    sigma[6, i, j, k] = (P[2, 0] * F11 + P[2, 1] * F12 + P[2, 2] * F13) / J_safe
    sigma[7, i, j, k] = (P[2, 0] * F21 + P[2, 1] * F22 + P[2, 2] * F23) / J_safe
    sigma[8, i, j, k] = (P[2, 0] * F31 + P[2, 1] * F32 + P[2, 2] * F33) / J_safe


class WarpNonlinearElasticLBM3D:
    """Warp-backed D3Q6x12 nonlinear elastic LBM on a cell-centered box."""

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        *,
        material: WarpHyperelasticMaterial3D,
        boundaries: BoxBoundarySet | None = None,
        length_x: float = 1.0,
        length_y: float = 1.0,
        length_z: float = 1.0,
        lattice_speed: float | Sequence[float] = 5.0,
        collision_omega: float = 1.8,
        device: str | None = None,
        check_linearized_cfl: bool = True,
        jacobian_floor: float = 1.0e-12,
        source_mode: SourceName = "none",
        source_vector: tuple[float, ...] = (0.0,) * NCOMP,
        damping_gamma: float = 0.0,
    ) -> None:
        if nx < 4 or ny < 4 or nz < 4:
            raise ValueError("nx, ny, and nz must be at least 4")
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.length_x = float(length_x)
        self.length_y = float(length_y)
        self.length_z = float(length_z)
        self.dx = self.length_x / float(self.nx)
        self.dy = self.length_y / float(self.ny)
        self.dz = self.length_z / float(self.nz)
        self.material = material
        self.boundaries = boundaries or BoxBoundarySet.periodic()
        self.boundaries.validate_periodic_pairs()
        speeds = _speed_tuple(lattice_speed)
        self.lattice_speed_x, self.lattice_speed_y, self.lattice_speed_z = speeds
        # Keep the historical scalar attribute as the limiting kinetic speed.
        self.lattice_speed = min(speeds)
        self.lattice_speeds = speeds
        self.collision_omega = float(collision_omega)
        if not (0.0 < self.collision_omega <= 2.0):
            raise ValueError("collision_omega must be in (0, 2]")
        if (
            check_linearized_cfl
            and self.material.linearized_stability_ratio(self.lattice_speed) >= 1.0
        ):
            raise ValueError("linearized D3Q6 stability ratio must be below one")
        directional_dt = (
            self.dx / self.lattice_speed_x,
            self.dy / self.lattice_speed_y,
            self.dz / self.lattice_speed_z,
        )
        self.dt = directional_dt[0]
        if max(abs(value - self.dt) for value in directional_dt[1:]) > 1.0e-12 * max(
            1.0, abs(self.dt)
        ):
            raise ValueError("directional spacings and lattice speeds must define one common dt")
        self.time = 0.0
        self.steps = 0
        self.device = device or default_device()
        self.shape = (self.nx, self.ny, self.nz)
        self.jacobian_floor = float(jacobian_floor)
        self.source_mode = source_id(source_mode)
        self.source_params = np.zeros(NCOMP, dtype=np.float64)
        if self.source_mode == SOURCE_CONSTANT:
            for idx, value in enumerate(tuple(source_vector)[:NCOMP]):
                self.source_params[idx] = float(value)
        elif self.source_mode == SOURCE_DAMPING:
            if damping_gamma < 0.0:
                raise ValueError("damping_gamma must be non-negative")
            self.source_params[0] = float(damping_gamma)

        self.x, self.y, self.z = reference_grid(
            self.nx, self.ny, self.nz, self.length_x, self.length_y, self.length_z
        )
        self.f0 = wp.zeros((FQ, self.nx, self.ny, self.nz), dtype=wp.float64, device=self.device)
        self.f1 = wp.zeros((FQ, self.nx, self.ny, self.nz), dtype=wp.float64, device=self.device)
        self.fpost = wp.zeros((FQ, self.nx, self.ny, self.nz), dtype=wp.float64, device=self.device)
        self.u = wp.zeros((3, self.nx, self.ny, self.nz), dtype=wp.float64, device=self.device)
        self.u_star = wp.zeros((3, self.nx, self.ny, self.nz), dtype=wp.float64, device=self.device)
        self.U = wp.zeros((NCOMP, self.nx, self.ny, self.nz), dtype=wp.float64, device=self.device)
        self.P = wp.zeros((9, self.nx, self.ny, self.nz), dtype=wp.float64, device=self.device)
        self.sigma = wp.zeros((9, self.nx, self.ny, self.nz), dtype=wp.float64, device=self.device)
        self.invalid = wp.zeros((1,), dtype=wp.int32, device=self.device)
        self.side_kind = _as_warp(self.boundaries.side_kind, dtype=wp.int32, device=self.device)
        self.value_mode = _as_warp(self.boundaries.value_mode, dtype=wp.int32, device=self.device)
        self.value_params = _as_warp(self.boundaries.params, dtype=wp.float64, device=self.device)
        self.source_params_wp = _as_warp(self.source_params, dtype=wp.float64, device=self.device)
        self.initialize_identity()

    def set_boundaries(self, boundaries: BoxBoundarySet) -> None:
        boundaries.validate_periodic_pairs()
        self.boundaries = boundaries
        wp.copy(self.side_kind, _as_warp(boundaries.side_kind, dtype=wp.int32, device=self.device))
        wp.copy(
            self.value_mode, _as_warp(boundaries.value_mode, dtype=wp.int32, device=self.device)
        )
        wp.copy(
            self.value_params, _as_warp(boundaries.params, dtype=wp.float64, device=self.device)
        )

    def initialize_identity(self, *, init_order: int = 1) -> None:
        U = np.zeros((NCOMP, self.nx, self.ny, self.nz), dtype=np.float64)
        U[3] = 1.0
        U[7] = 1.0
        U[11] = 1.0
        displacement = np.zeros((3, self.nx, self.ny, self.nz), dtype=np.float64)
        self.initialize_from_numpy(U=U, displacement=displacement, init_order=init_order)

    def initialize_from_numpy(
        self,
        *,
        U: np.ndarray,
        displacement: np.ndarray | None = None,
        init_order: int = 1,
        enforce_dirichlet_tangent: bool = True,
        apply_initial_boundaries: bool = True,
        initial_boundary_time: float = 0.0,
    ) -> None:
        U = np.asarray(U, dtype=np.float64)
        if U.shape != (NCOMP, self.nx, self.ny, self.nz):
            raise ValueError(f"U must have shape {(NCOMP, self.nx, self.ny, self.nz)}")
        if displacement is None:
            displacement = np.zeros((3, self.nx, self.ny, self.nz), dtype=np.float64)
        displacement = np.asarray(displacement, dtype=np.float64)
        if displacement.shape != (3, self.nx, self.ny, self.nz):
            raise ValueError(f"displacement must have shape {(3, self.nx, self.ny, self.nz)}")
        if enforce_dirichlet_tangent:
            U = _enforce_dirichlet_tangent_columns(
                U, displacement, self.boundaries, dx=self.dx, dy=self.dy, dz=self.dz
            )
        if init_order == 1:
            f = _equilibrium_np(U, self.material, self.lattice_speeds)
            source = _source_np(U, self.source_mode, self.source_params)
            f -= (0.5 * self.dt / float(Q)) * source[None, ...]
        elif init_order == 2:
            f = _second_order_populations_np(
                U,
                self.material,
                self.lattice_speeds,
                dx=self.dx,
                dy=self.dy,
                dz=self.dz,
                dt=self.dt,
                periodic_x=self.boundaries.periodic_x,
                periodic_y=self.boundaries.periodic_y,
                periodic_z=self.boundaries.periodic_z,
                source_mode=self.source_mode,
                source_params=self.source_params,
            )
        else:
            raise ValueError("init_order must be 1 or 2")
        u_star0 = displacement - 0.5 * self.dt * U[:3]
        displacement_wp = _as_warp(displacement, dtype=wp.float64, device=self.device)
        wp.copy(self.f0, _as_warp(_flatten_pop(f), dtype=wp.float64, device=self.device))
        wp.copy(
            self.f1, wp.zeros((FQ, self.nx, self.ny, self.nz), dtype=wp.float64, device=self.device)
        )
        wp.copy(
            self.fpost,
            wp.zeros((FQ, self.nx, self.ny, self.nz), dtype=wp.float64, device=self.device),
        )
        wp.copy(self.u_star, _as_warp(u_star0, dtype=wp.float64, device=self.device))
        self.time = 0.0
        self.steps = 0
        self.refresh_current()
        if apply_initial_boundaries:
            self.apply_boundary_reconstruction(float(initial_boundary_time))
            self.refresh_current()
            wp.launch(
                _set_u_star_from_displacement_kernel,
                dim=self.shape,
                inputs=[displacement_wp, self.U, self.u_star, self.dt],
                device=self.device,
            )
            self.refresh_current()

    def apply_boundary_reconstruction(self, boundary_time: float) -> None:
        max_face_nodes = max(self.ny * self.nz, self.nx * self.nz, self.nx * self.ny)
        for side in (SIDE_LEFT, SIDE_RIGHT, SIDE_BOTTOM, SIDE_TOP, SIDE_FRONT, SIDE_BACK):
            if self.boundaries.side_kind[side] != BC_PERIODIC:
                wp.launch(
                    _apply_boundary_side_kernel,
                    dim=max_face_nodes,
                    inputs=[
                        self.f0,
                        self.f0,
                        self.U,
                        self.side_kind,
                        self.value_mode,
                        self.value_params,
                        self.nx,
                        self.ny,
                        self.nz,
                        self.length_x,
                        self.length_y,
                        self.length_z,
                        float(boundary_time),
                        self.lattice_speed_x,
                        self.lattice_speed_y,
                        self.lattice_speed_z,
                        self.material.id,
                        self.material.lam,
                        self.material.mu,
                        self.material.param0,
                        self.material.param1,
                        self.material.param2,
                        self.jacobian_floor,
                        side,
                    ],
                    device=self.device,
                )

    def step(self) -> None:
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
                self.lattice_speed_x,
                self.lattice_speed_y,
                self.lattice_speed_z,
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
            _stream_kernel,
            dim=self.shape,
            inputs=[
                self.fpost,
                self.f1,
                self.nx,
                self.ny,
                self.nz,
                int(self.boundaries.periodic_x),
                int(self.boundaries.periodic_y),
                int(self.boundaries.periodic_z),
            ],
            device=self.device,
        )
        max_face_nodes = max(self.ny * self.nz, self.nx * self.nz, self.nx * self.ny)
        for side in (SIDE_LEFT, SIDE_RIGHT, SIDE_BOTTOM, SIDE_TOP, SIDE_FRONT, SIDE_BACK):
            if self.boundaries.side_kind[side] != BC_PERIODIC:
                wp.launch(
                    _apply_boundary_side_kernel,
                    dim=max_face_nodes,
                    inputs=[
                        self.fpost,
                        self.f1,
                        self.U,
                        self.side_kind,
                        self.value_mode,
                        self.value_params,
                        self.nx,
                        self.ny,
                        self.nz,
                        self.length_x,
                        self.length_y,
                        self.length_z,
                        self.time + 0.5 * self.dt,
                        self.lattice_speed_x,
                        self.lattice_speed_y,
                        self.lattice_speed_z,
                        self.material.id,
                        self.material.lam,
                        self.material.mu,
                        self.material.param0,
                        self.material.param1,
                        self.material.param2,
                        self.jacobian_floor,
                        side,
                    ],
                    device=self.device,
                )
        self.f0, self.f1 = self.f1, self.f0
        self.time += self.dt
        self.steps += 1

    def refresh_current(self) -> None:
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

    def run_until(
        self, t_end: float, *, stop_on_invalid: bool = True, check_interval: int = 20
    ) -> None:
        while self.time < float(t_end) - 0.5 * self.dt:
            self.step()
            if stop_on_invalid and self.steps % int(check_interval) == 0:
                if self.invalid_state():
                    break
        self.refresh_current()
        wp.synchronize_device(self.device)

    def invalid_state(self) -> bool:
        wp.synchronize_device(self.device)
        if int(np.asarray(self.invalid.numpy())[0]) != 0:
            return True
        U = self.numpy_field("U")
        return not np.isfinite(U).all()

    def numpy_field(
        self, name: Literal["u", "u_star", "U", "P", "sigma", "f0", "fpost"]
    ) -> np.ndarray:
        field = getattr(self, name)
        wp.synchronize_device(self.device)
        return np.asarray(field.numpy())

    def displacement(self) -> np.ndarray:
        return self.numpy_field("u")

    def system_state(self) -> np.ndarray:
        return self.numpy_field("U")

    def deformation_gradient(self) -> np.ndarray:
        return _state_to_F_np(self.system_state())

    def deformation_jacobian(self) -> np.ndarray:
        return _det3_np(self.deformation_gradient())

    def summary(self) -> dict[str, float | int | str | bool]:
        self.refresh_current()
        wp.synchronize_device(self.device)
        u = self.displacement()
        U = self.system_state()
        J = self.deformation_jacobian()
        return {
            "device": self.device,
            "model": self.material.model,
            "lambda": float(self.material.lam),
            "mu": float(self.material.mu),
            "material_param0": float(self.material.param0),
            "material_param1": float(self.material.param1),
            "material_param2": float(self.material.param2),
            "poisson": float(self.material.poisson),
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "steps": self.steps,
            "time": float(self.time),
            "dx": float(self.dx),
            "dy": float(self.dy),
            "dz": float(self.dz),
            "dt": float(self.dt),
            "lattice_speed": float(self.lattice_speed),
            "lattice_speed_x": float(self.lattice_speed_x),
            "lattice_speed_y": float(self.lattice_speed_y),
            "lattice_speed_z": float(self.lattice_speed_z),
            "collision_omega": float(self.collision_omega),
            "source_mode": source_name(self.source_mode),
            "linearized_stability_ratio": float(
                self.material.linearized_stability_ratio(self.lattice_speed)
            ),
            "periodic_x": bool(self.boundaries.periodic_x),
            "periodic_y": bool(self.boundaries.periodic_y),
            "periodic_z": bool(self.boundaries.periodic_z),
            "max_abs_u": float(np.max(np.sqrt(np.sum(u * u, axis=0)))),
            "max_abs_v": float(np.max(np.sqrt(U[0] * U[0] + U[1] * U[1] + U[2] * U[2]))),
            "min_J": float(np.min(J)),
            "max_J": float(np.max(J)),
            "finite": bool(np.isfinite(u).all() and np.isfinite(U).all() and np.all(J > 0.0)),
        }

    def save_npz(self, path: str | Path) -> None:
        self.refresh_current()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            x=self.x,
            y=self.y,
            z=self.z,
            u=self.displacement(),
            U=self.system_state(),
            F=self.deformation_gradient(),
            P=self.numpy_field("P"),
            sigma=self.numpy_field("sigma"),
            f=_unflatten_pop(self.numpy_field("f0")),
            side_kind=self.boundaries.side_kind,
            value_mode=self.boundaries.value_mode,
            value_params=self.boundaries.params,
            time=np.array(self.time),
            dx=np.array(self.dx),
            dt=np.array(self.dt),
            lattice_speed=np.asarray(self.lattice_speeds),
            collision_omega=np.array(self.collision_omega),
            source_mode=np.array(source_name(self.source_mode)),
            source_params=self.source_params,
            model=np.array(self.material.model),
            lam=np.array(self.material.lam),
            mu=np.array(self.material.mu),
            material_params=np.asarray(self.material.params, dtype=np.float64),
        )
