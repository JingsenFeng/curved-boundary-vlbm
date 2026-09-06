"""Warp D2Q4x6 vectorial LBM for 2D finite-strain elastodynamics.

This module is a GPU/CPU Warp counterpart of the current NumPy
``VectorHyperelasticBoundaryLBM2D`` prototype.  It keeps the core pieces
explicit and local:

* D2Q4 populations with six components per direction,
  ``U = [v_x, v_y, F_11, F_12, F_21, F_22]``.
* Hyperelastic first Piola stress selected by an integer material id.
* Periodic streaming per coordinate direction.
* Axis-aligned half-way Dirichlet and Neumann boundary reconstruction.

The boundary-value callbacks used in the NumPy prototype cannot be called from
Warp kernels, so boundary data are encoded by small value-model enums and
parameter arrays.  New constitutive laws or boundary value models should be
added as new enum branches in the local ``wp.func`` routines below.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Literal

import numpy as np
import warp as wp


wp.init()

Q = 4
NCOMP = 6
FQ = Q * NCOMP

MATERIAL_SVK = 0
MATERIAL_NEO_HOOKE = 1
MATERIAL_LOG_NEO_HOOKE = 2
MATERIAL_MOONEY_RIVLIN = 3
MATERIAL_YEOH = 4
MATERIAL_GENT = 5
MATERIAL_SUGIYAMA_SVK = 6

BC_PERIODIC = 0
BC_DIRICHLET = 1
BC_NEUMANN = 2

SIDE_LEFT = 0
SIDE_RIGHT = 1
SIDE_BOTTOM = 2
SIDE_TOP = 3

VALUE_ZERO = 0
VALUE_CONSTANT = 1
VALUE_RAMP = 2
VALUE_SHEAR_VELOCITY_X = 3
VALUE_WAVE_BEAM_RICKER_Y = 4
VALUE_SINE2_HOLD = 5
VALUE_AFFINE_VELOCITY = 6

SOURCE_NONE = 0
SOURCE_CONSTANT = 1
SOURCE_DAMPING = 2

PI = wp.constant(wp.float64(3.141592653589793238462643383279502884))

_CX = np.array([1, 0, -1, 0], dtype=np.int32)
_CY = np.array([0, 1, 0, -1], dtype=np.int32)
CX_I = wp.constant(wp.vec4i(1, 0, -1, 0))
CY_I = wp.constant(wp.vec4i(0, 1, 0, -1))


MaterialName = Literal[
    "svk",
    "neo_hooke",
    "log_neo_hooke",
    "mooney_rivlin",
    "yeoh",
    "gent",
    "sugiyama_svk",
]
BoundaryKind = Literal["periodic", "dirichlet", "neumann"]
SideName = Literal["left", "right", "bottom", "top"]
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
    if name == "sugiyama_svk":
        return MATERIAL_SUGIYAMA_SVK
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
    if mid == MATERIAL_SUGIYAMA_SVK:
        return "sugiyama_svk"
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
    raise ValueError(f"unsupported side: {side}")


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
class WarpHyperelasticMaterial:
    model: MaterialName
    lam: float
    mu: float = 1.0
    param0: float = 0.0
    param1: float = 0.0
    param2: float = 0.0

    def __post_init__(self) -> None:
        if self.model == "mooney_rivlin" and self.param0 == 0.0 and self.param1 == 0.0:
            csum = 0.5 * self.mu
            object.__setattr__(self, "param0", 0.75 * csum)
            object.__setattr__(self, "param1", 0.25 * csum)
        elif self.model == "yeoh" and self.param0 == 0.0:
            object.__setattr__(self, "param0", 0.5 * self.mu)
        elif self.model == "gent" and self.param0 == 0.0:
            object.__setattr__(self, "param0", 50.0)

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
    ) -> "WarpHyperelasticMaterial":
        if abs(1.0 - 2.0 * poisson) < 1e-14:
            raise ValueError("poisson ratio too close to 1/2")
        lam = 2.0 * mu * poisson / (1.0 - 2.0 * poisson)
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
        return self.lam / (2.0 * (self.lam + self.mu))

    def linearized_stability_ratio(self, lattice_speed: float) -> float:
        return 2.0 * math.sqrt(self.lam + 2.0 * self.mu) / float(lattice_speed)

    @property
    def params(self) -> tuple[float, float, float]:
        return float(self.param0), float(self.param1), float(self.param2)


@dataclass(frozen=True)
class AxisBoundarySpec:
    side: SideName
    kind: BoundaryKind
    value_mode: int = VALUE_ZERO
    params: tuple[float, ...] = ()
    label: str = ""


def smoothstep5_np(eta: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(eta, dtype=np.float64), 0.0, 1.0)
    return z * z * z * (10.0 - 15.0 * z + 6.0 * z * z)


def wave_beam_window_mean(ny: int, length_y: float, taper_nodes: float) -> float:
    if taper_nodes <= 0.0:
        return 1.0
    spacing = float(length_y) / float(ny)
    y = (np.arange(ny, dtype=np.float64) + 0.5) * spacing
    layer = max(float(taper_nodes) * spacing, np.finfo(float).eps)
    window = smoothstep5_np(y / layer) * smoothstep5_np((float(length_y) - y) / layer)
    mean = float(np.mean(window))
    return mean if mean > 1.0e-14 else 1.0


class RectangularBoundarySet:
    """Compact rectangular boundary encoding for Warp kernels."""

    def __init__(self, specs: tuple[AxisBoundarySpec, ...] = ()) -> None:
        self.side_kind = np.full(4, BC_PERIODIC, dtype=np.int32)
        self.value_mode = np.zeros(4, dtype=np.int32)
        self.params = np.zeros((4, 8), dtype=np.float64)
        self.labels = ["periodic", "periodic", "periodic", "periodic"]
        for spec in specs:
            self.set(spec)
        self.validate_periodic_pairs()

    def set(self, spec: AxisBoundarySpec) -> None:
        sid = side_id(spec.side)
        self.side_kind[sid] = kind_id(spec.kind)
        self.value_mode[sid] = int(spec.value_mode)
        self.params[sid, :] = 0.0
        for i, value in enumerate(spec.params[:8]):
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

    @classmethod
    def periodic(cls) -> "RectangularBoundarySet":
        return cls(())

    @classmethod
    def uniaxial_tension(
        cls,
        *,
        alpha: float,
        active_time: float = 2.0,
        sine_denominator: float = 4.0,
    ) -> "RectangularBoundarySet":
        """Muller et al. Fig. 2 traction benchmark.

        The top and bottom faces receive opposite first-Piola tractions
        ``T_y = +/- alpha sin^2(pi t / 4)`` until ``t=2``, then the traction
        is held at ``+/- alpha``.  The lateral faces are traction free.
        """

        return cls(
            (
                AxisBoundarySpec("left", "neumann", VALUE_ZERO, label="free left"),
                AxisBoundarySpec("right", "neumann", VALUE_ZERO, label="free right"),
                AxisBoundarySpec(
                    "bottom",
                    "neumann",
                    VALUE_SINE2_HOLD,
                    (0.0, -float(alpha), float(active_time), float(sine_denominator)),
                    "loaded bottom",
                ),
                AxisBoundarySpec(
                    "top",
                    "neumann",
                    VALUE_SINE2_HOLD,
                    (0.0, float(alpha), float(active_time), float(sine_denominator)),
                    "loaded top",
                ),
            )
        )

    @classmethod
    def wave_beam(
        cls,
        *,
        ny: int,
        length_y: float,
        load_wave: float = 0.15,
        ricker_width: float = 0.215,
        ricker_t0: float = 1.0,
        source_end: float = 2.0,
        corner_taper_nodes: float = 8.0,
    ) -> "RectangularBoundarySet":
        spacing = float(length_y) / float(ny)
        mean_w = wave_beam_window_mean(ny, length_y, corner_taper_nodes)
        return cls(
            (
                AxisBoundarySpec("left", "dirichlet", VALUE_ZERO, label="clamped left"),
                AxisBoundarySpec(
                    "right",
                    "neumann",
                    VALUE_WAVE_BEAM_RICKER_Y,
                    (
                        load_wave,
                        ricker_width,
                        ricker_t0,
                        source_end,
                        length_y,
                        spacing,
                        corner_taper_nodes,
                        mean_w,
                    ),
                    "corner-compatible Ricker shear",
                ),
                AxisBoundarySpec("bottom", "neumann", VALUE_ZERO, label="free bottom"),
                AxisBoundarySpec("top", "neumann", VALUE_ZERO, label="free top"),
            )
        )

    @classmethod
    def simple_shear_velocity(
        cls,
        *,
        amplitude: float = 0.03,
        active_time: float = 2.0,
    ) -> "RectangularBoundarySet":
        return cls(
            (
                AxisBoundarySpec("left", "periodic"),
                AxisBoundarySpec("right", "periodic"),
                AxisBoundarySpec("bottom", "dirichlet", VALUE_ZERO, label="fixed bottom"),
                AxisBoundarySpec(
                    "top",
                    "dirichlet",
                    VALUE_SHEAR_VELOCITY_X,
                    (amplitude, active_time),
                    "top shear velocity",
                ),
            )
        )

    @classmethod
    def simple_shear_benchmark(
        cls,
        *,
        amplitude: float = 0.03,
        active_time: float = 2.0,
    ) -> "RectangularBoundarySet":
        """Muller et al. Fig. 5 simple-shear benchmark.

        The bottom face is clamped, the top face is prescribed with
        ``v_x = alpha sin(pi t/2)`` for ``t<2`` and zero afterward, and the
        left/right faces are first-Piola traction free.
        """

        return cls(
            (
                AxisBoundarySpec("left", "neumann", VALUE_ZERO, label="free left"),
                AxisBoundarySpec("right", "neumann", VALUE_ZERO, label="free right"),
                AxisBoundarySpec("bottom", "dirichlet", VALUE_ZERO, label="fixed bottom"),
                AxisBoundarySpec(
                    "top",
                    "dirichlet",
                    VALUE_SHEAR_VELOCITY_X,
                    (amplitude, active_time),
                    "top shear velocity",
                ),
            )
        )

    @classmethod
    def affine_velocity(
        cls,
        *,
        axx: float,
        axy: float,
        ayx: float,
        ayy: float,
        bx: float = 0.0,
        by: float = 0.0,
        active_time: float = 0.0,
    ) -> "RectangularBoundarySet":
        """Prescribe the same affine velocity on all four rectangle sides."""

        params = (
            float(axx),
            float(axy),
            float(bx),
            float(ayx),
            float(ayy),
            float(by),
            float(active_time),
            0.0,
        )
        return cls(
            (
                AxisBoundarySpec("left", "dirichlet", VALUE_AFFINE_VELOCITY, params, "affine left"),
                AxisBoundarySpec(
                    "right", "dirichlet", VALUE_AFFINE_VELOCITY, params, "affine right"
                ),
                AxisBoundarySpec(
                    "bottom", "dirichlet", VALUE_AFFINE_VELOCITY, params, "affine bottom"
                ),
                AxisBoundarySpec("top", "dirichlet", VALUE_AFFINE_VELOCITY, params, "affine top"),
            )
        )


def _first_piola_np(F: np.ndarray, material: WarpHyperelasticMaterial) -> np.ndarray:
    if material.model == "svk":
        C00 = F[0, 0] * F[0, 0] + F[1, 0] * F[1, 0]
        C01 = F[0, 0] * F[0, 1] + F[1, 0] * F[1, 1]
        C11 = F[0, 1] * F[0, 1] + F[1, 1] * F[1, 1]
        E00 = 0.5 * (C00 - 1.0)
        E01 = 0.5 * C01
        E11 = 0.5 * (C11 - 1.0)
        trE = E00 + E11
        S00 = material.lam * trE + 2.0 * material.mu * E00
        S01 = 2.0 * material.mu * E01
        S11 = material.lam * trE + 2.0 * material.mu * E11
        P = np.empty_like(F)
        P[0, 0] = F[0, 0] * S00 + F[0, 1] * S01
        P[0, 1] = F[0, 0] * S01 + F[0, 1] * S11
        P[1, 0] = F[1, 0] * S00 + F[1, 1] * S01
        P[1, 1] = F[1, 0] * S01 + F[1, 1] * S11
        return P

    F11, F12, F21, F22 = F[0, 0], F[0, 1], F[1, 0], F[1, 1]
    J = F11 * F22 - F12 * F21
    if np.any(J <= 0.0):
        raise FloatingPointError(
            f"non-positive deformation Jacobian: min(J)={float(np.min(J)):.6e}"
        )
    H = np.empty_like(F)
    H[0, 0] = F22 / J
    H[0, 1] = -F21 / J
    H[1, 0] = -F12 / J
    H[1, 1] = F11 / J
    if material.model == "neo_hooke":
        volumetric = 0.5 * material.lam * (J * J - 1.0)
        return material.mu * (F - H) + volumetric * H
    if material.model == "log_neo_hooke":
        volumetric = material.lam * np.log(J)
        return material.mu * (F - H) + volumetric * H

    F2 = F11 * F11 + F12 * F12 + F21 * F21 + F22 * F22
    I1 = F2 + 1.0
    I2 = J * J + F2
    logJ = np.log(J)
    jm23 = np.exp((-2.0 / 3.0) * logJ)
    jm43 = np.exp((-4.0 / 3.0) * logJ)
    H_scale_1 = (-2.0 / 3.0) * I1
    H_scale_2 = 2.0 * J * J - (4.0 / 3.0) * I2
    grad_i1 = np.empty_like(F)
    grad_i2 = np.empty_like(F)
    grad_i1[0, 0] = jm23 * (2.0 * F11 + H_scale_1 * H[0, 0])
    grad_i1[0, 1] = jm23 * (2.0 * F12 + H_scale_1 * H[0, 1])
    grad_i1[1, 0] = jm23 * (2.0 * F21 + H_scale_1 * H[1, 0])
    grad_i1[1, 1] = jm23 * (2.0 * F22 + H_scale_1 * H[1, 1])
    grad_i2[0, 0] = jm43 * (2.0 * F11 + H_scale_2 * H[0, 0])
    grad_i2[0, 1] = jm43 * (2.0 * F12 + H_scale_2 * H[0, 1])
    grad_i2[1, 0] = jm43 * (2.0 * F21 + H_scale_2 * H[1, 0])
    grad_i2[1, 1] = jm43 * (2.0 * F22 + H_scale_2 * H[1, 1])
    volumetric = material.lam * logJ * H
    if material.model == "mooney_rivlin":
        return material.param0 * grad_i1 + material.param1 * grad_i2 + volumetric
    if material.model == "yeoh":
        A = jm23 * I1 - 3.0
        scale = material.param0 + 2.0 * material.param1 * A + 3.0 * material.param2 * A * A
        return scale * grad_i1 + volumetric
    if material.model == "gent":
        A = jm23 * I1 - 3.0
        denom = 1.0 - A / material.param0
        if np.any(denom <= 0.0):
            raise FloatingPointError(
                f"Gent chain limit exceeded: min(1-(I1bar-3)/Jm)={float(np.min(denom)):.6e}"
            )
        return (0.5 * material.mu / denom) * grad_i1 + volumetric
    if material.model == "sugiyama_svk":
        # Sugiyama et al. (2011), equations (15)--(20): c1=mu,
        # c2=-mu/2 and c3=(lambda_paper+2*mu)/8.  ``lam`` remains the
        # finite log(J) pressure penalty and ``param0`` stores lambda_paper.
        A = jm23 * I1 - 3.0
        c2 = -0.5 * material.mu
        c3 = (material.param0 + 2.0 * material.mu) / 8.0
        return (material.mu + 2.0 * c3 * A) * grad_i1 + c2 * grad_i2 + volumetric
    raise ValueError(f"unsupported material model: {material.model}")


def _flux_x_np(U: np.ndarray, material: WarpHyperelasticMaterial) -> np.ndarray:
    F = np.empty((2, 2) + U.shape[1:], dtype=np.float64)
    F[0, 0], F[0, 1], F[1, 0], F[1, 1] = U[2], U[3], U[4], U[5]
    P = _first_piola_np(F, material)
    phix = np.empty_like(U)
    phix[0] = -P[0, 0]
    phix[1] = -P[1, 0]
    phix[2] = -U[0]
    phix[3] = 0.0
    phix[4] = -U[1]
    phix[5] = 0.0
    return phix


def _flux_y_np(U: np.ndarray, material: WarpHyperelasticMaterial) -> np.ndarray:
    F = np.empty((2, 2) + U.shape[1:], dtype=np.float64)
    F[0, 0], F[0, 1], F[1, 0], F[1, 1] = U[2], U[3], U[4], U[5]
    P = _first_piola_np(F, material)
    phiy = np.empty_like(U)
    phiy[0] = -P[0, 1]
    phiy[1] = -P[1, 1]
    phiy[2] = 0.0
    phiy[3] = -U[0]
    phiy[4] = 0.0
    phiy[5] = -U[1]
    return phiy


def _equilibrium_np(
    U: np.ndarray, material: WarpHyperelasticMaterial, lattice_speed: float
) -> np.ndarray:
    phix = _flux_x_np(U, material)
    phiy = _flux_y_np(U, material)
    f = np.empty((Q, NCOMP) + U.shape[1:], dtype=np.float64)
    for q, (cx, cy) in enumerate(zip(_CX, _CY)):
        f[q] = 0.25 * (U + (2.0 / float(lattice_speed)) * (cx * phix + cy * phiy))
    return f


def _source_np(U: np.ndarray, source_mode: int, source_params: np.ndarray) -> np.ndarray:
    out = np.zeros_like(U)
    params = np.asarray(source_params, dtype=np.float64)
    if source_mode == SOURCE_CONSTANT:
        out[:] = params[:NCOMP, None, None]
    elif source_mode == SOURCE_DAMPING:
        gamma = float(params[0])
        out[0] = -gamma * U[0]
        out[1] = -gamma * U[1]
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
    slc_mid = list(slc)
    slc_mid[axis] = slice(1, -1)
    slc_p = list(slc)
    slc_p[axis] = slice(2, None)
    slc_m = list(slc)
    slc_m[axis] = slice(None, -2)
    out[tuple(slc_mid)] = (U[tuple(slc_p)] - U[tuple(slc_m)]) / (2.0 * float(spacing))

    slc0 = list(slc)
    slc1 = list(slc)
    slc2 = list(slc)
    slc0[axis] = 0
    slc1[axis] = 1
    slc2[axis] = 2
    out[tuple(slc0)] = (-3.0 * U[tuple(slc0)] + 4.0 * U[tuple(slc1)] - U[tuple(slc2)]) / (
        2.0 * float(spacing)
    )

    slcn = list(slc)
    slcn1 = list(slc)
    slcn2 = list(slc)
    slcn[axis] = -1
    slcn1[axis] = -2
    slcn2[axis] = -3
    out[tuple(slcn)] = (3.0 * U[tuple(slcn)] - 4.0 * U[tuple(slcn1)] + U[tuple(slcn2)]) / (
        2.0 * float(spacing)
    )
    return out


def _active_derivative_first_spatial(
    U: np.ndarray, active: np.ndarray, spacing: float, *, periodic: bool
) -> np.ndarray:
    out = np.zeros_like(U)
    active = np.asarray(active, dtype=bool)
    done = np.zeros_like(active)
    h = float(spacing)
    if U.shape[1] < 2:
        return out

    if periodic:
        up1 = np.roll(U, -1, axis=1)
        um1 = np.roll(U, 1, axis=1)
        up2 = np.roll(U, -2, axis=1)
        um2 = np.roll(U, 2, axis=1)
        ap1 = np.roll(active, -1, axis=0)
        am1 = np.roll(active, 1, axis=0)
        ap2 = np.roll(active, -2, axis=0)
        am2 = np.roll(active, 2, axis=0)

        m = active & ap1 & am1
        out[:, m] = ((up1 - um1) / (2.0 * h))[:, m]
        done |= m

        m = active & ap1 & ap2 & ~done
        out[:, m] = ((-3.0 * U + 4.0 * up1 - up2) / (2.0 * h))[:, m]
        done |= m

        m = active & am1 & am2 & ~done
        out[:, m] = ((3.0 * U - 4.0 * um1 + um2) / (2.0 * h))[:, m]
        done |= m

        m = active & ap1 & ~done
        out[:, m] = ((up1 - U) / h)[:, m]
        done |= m

        m = active & am1 & ~done
        out[:, m] = ((U - um1) / h)[:, m]
        return out

    if U.shape[1] >= 3:
        m = active[1:-1] & active[2:] & active[:-2]
        tmp = out[:, 1:-1]
        tmp[:, m] = ((U[:, 2:] - U[:, :-2]) / (2.0 * h))[:, m]
        out[:, 1:-1] = tmp
        done[1:-1] |= m

        m = active[:-2] & active[1:-1] & active[2:] & ~done[:-2]
        tmp = out[:, :-2]
        tmp[:, m] = ((-3.0 * U[:, :-2] + 4.0 * U[:, 1:-1] - U[:, 2:]) / (2.0 * h))[:, m]
        out[:, :-2] = tmp
        done[:-2] |= m

        m = active[2:] & active[1:-1] & active[:-2] & ~done[2:]
        tmp = out[:, 2:]
        tmp[:, m] = ((3.0 * U[:, 2:] - 4.0 * U[:, 1:-1] + U[:, :-2]) / (2.0 * h))[:, m]
        out[:, 2:] = tmp
        done[2:] |= m

    m = active[:-1] & active[1:] & ~done[:-1]
    tmp = out[:, :-1]
    tmp[:, m] = ((U[:, 1:] - U[:, :-1]) / h)[:, m]
    out[:, :-1] = tmp
    done[:-1] |= m

    m = active[1:] & active[:-1] & ~done[1:]
    tmp = out[:, 1:]
    tmp[:, m] = ((U[:, 1:] - U[:, :-1]) / h)[:, m]
    out[:, 1:] = tmp
    return out


def _masked_spatial_derivative(
    U: np.ndarray, active: np.ndarray, spacing: float, axis: int, periodic: bool
) -> np.ndarray:
    if axis == 1:
        return _active_derivative_first_spatial(U, active, spacing, periodic=periodic)
    if axis == 2:
        deriv = _active_derivative_first_spatial(
            np.swapaxes(U, 1, 2), active.T, spacing, periodic=periodic
        )
        return np.swapaxes(deriv, 1, 2)
    raise ValueError("masked derivative axis must be 1 or 2")


def _flux_jvp_np(
    U: np.ndarray, W: np.ndarray, material: WarpHyperelasticMaterial, axis: int
) -> np.ndarray:
    scale = float(np.max(np.abs(W)))
    if scale == 0.0 or not np.isfinite(scale):
        return np.zeros_like(U)
    eps = 1.0e-6 / max(1.0, scale)
    flux = _flux_x_np if axis == 1 else _flux_y_np
    for _ in range(8):
        try:
            return (flux(U + eps * W, material) - flux(U - eps * W, material)) / (2.0 * eps)
        except FloatingPointError:
            eps *= 0.25
    raise FloatingPointError(
        "failed to compute finite-difference flux tangent for second-order initialization"
    )


def _second_order_populations_from_derivatives_np(
    U: np.ndarray,
    Ux: np.ndarray,
    Uy: np.ndarray,
    material: WarpHyperelasticMaterial,
    lattice_speed: float,
    *,
    dt: float,
    source_mode: int = SOURCE_NONE,
    source_params: np.ndarray | None = None,
) -> np.ndarray:
    if source_params is None:
        source_params = np.zeros(NCOMP, dtype=np.float64)
    AxUx = _flux_jvp_np(U, Ux, material, axis=1)
    AyUy = _flux_jvp_np(U, Uy, material, axis=2)
    Ut = _source_np(U, source_mode, source_params) - (AxUx + AyUy)

    feq = _equilibrium_np(U, material, lattice_speed)
    correction = np.empty_like(feq)
    c = float(lattice_speed)
    for q, (cx, cy) in enumerate(zip(_CX, _CY)):
        Wq = Ut + c * (cx * Ux + cy * Uy)
        AxW = _flux_jvp_np(U, Wq, material, axis=1)
        AyW = _flux_jvp_np(U, Wq, material, axis=2)
        correction[q] = 0.25 * (Wq + (2.0 / c) * (cx * AxW + cy * AyW))
    return feq - 0.5 * float(dt) * correction


def _second_order_populations_np(
    U: np.ndarray,
    material: WarpHyperelasticMaterial,
    lattice_speed: float,
    *,
    dx: float,
    dy: float,
    dt: float,
    periodic_x: bool,
    periodic_y: bool,
    source_mode: int = SOURCE_NONE,
    source_params: np.ndarray | None = None,
) -> np.ndarray:
    Ux = _spatial_derivative(U, dx, axis=1, periodic=periodic_x)
    Uy = _spatial_derivative(U, dy, axis=2, periodic=periodic_y)
    return _second_order_populations_from_derivatives_np(
        U,
        Ux,
        Uy,
        material,
        lattice_speed,
        dt=dt,
        source_mode=source_mode,
        source_params=source_params,
    )


def _second_order_populations_masked_np(
    U: np.ndarray,
    active: np.ndarray,
    material: WarpHyperelasticMaterial,
    lattice_speed: float,
    *,
    dx: float,
    dy: float,
    dt: float,
    periodic_x: bool,
    periodic_y: bool,
    source_mode: int = SOURCE_NONE,
    source_params: np.ndarray | None = None,
) -> np.ndarray:
    active_bool = np.asarray(active, dtype=bool)
    U_work = np.array(U, dtype=np.float64, copy=True)
    U_work[:, ~active_bool] = 0.0
    U_work[2, ~active_bool] = 1.0
    U_work[5, ~active_bool] = 1.0
    Ux = _masked_spatial_derivative(U_work, active_bool, dx, axis=1, periodic=periodic_x)
    Uy = _masked_spatial_derivative(U_work, active_bool, dy, axis=2, periodic=periodic_y)
    return _second_order_populations_from_derivatives_np(
        U_work,
        Ux,
        Uy,
        material,
        lattice_speed,
        dt=dt,
        source_mode=source_mode,
        source_params=source_params,
    )


def _flatten_pop(f: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(f.reshape((FQ,) + f.shape[2:]))


def _unflatten_pop(f: np.ndarray) -> np.ndarray:
    return f.reshape((Q, NCOMP) + f.shape[1:])


def _edge_derivative(values: np.ndarray, spacing: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    if values.size < 3:
        out.fill(0.0)
        return out
    out[1:-1] = (values[2:] - values[:-2]) / (2.0 * float(spacing))
    out[0] = (-3.0 * values[0] + 4.0 * values[1] - values[2]) / (2.0 * float(spacing))
    out[-1] = (3.0 * values[-1] - 4.0 * values[-2] + values[-3]) / (2.0 * float(spacing))
    return out


def _enforce_dirichlet_tangent_columns(
    U: np.ndarray,
    displacement: np.ndarray,
    boundaries: RectangularBoundarySet,
    *,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Make Dirichlet boundary tangential F columns consistent with ``u0``.

    For a vertical side the tangent is Y, so this fills
    ``F[:,1] = e_y + d u / dY``.  For a horizontal side the tangent is X, so it
    fills ``F[:,0] = e_x + d u / dX``.  Normal columns are left untouched.
    """

    out = np.array(U, dtype=np.float64, copy=True)
    u = np.asarray(displacement, dtype=np.float64)
    if boundaries.side_kind[SIDE_LEFT] == BC_DIRICHLET:
        i = 0
        out[3, i, :] = _edge_derivative(u[0, i, :], dy)
        out[5, i, :] = 1.0 + _edge_derivative(u[1, i, :], dy)
    if boundaries.side_kind[SIDE_RIGHT] == BC_DIRICHLET:
        i = out.shape[1] - 1
        out[3, i, :] = _edge_derivative(u[0, i, :], dy)
        out[5, i, :] = 1.0 + _edge_derivative(u[1, i, :], dy)
    if boundaries.side_kind[SIDE_BOTTOM] == BC_DIRICHLET:
        j = 0
        out[2, :, j] = 1.0 + _edge_derivative(u[0, :, j], dx)
        out[4, :, j] = _edge_derivative(u[1, :, j], dx)
    if boundaries.side_kind[SIDE_TOP] == BC_DIRICHLET:
        j = out.shape[2] - 1
        out[2, :, j] = 1.0 + _edge_derivative(u[0, :, j], dx)
        out[4, :, j] = _edge_derivative(u[1, :, j], dx)
    return out


def reference_grid(
    nx: int, ny: int, length_x: float, length_y: float
) -> tuple[np.ndarray, np.ndarray]:
    dx = float(length_x) / float(nx)
    dy = float(length_y) / float(ny)
    x = (np.arange(nx, dtype=np.float64) + 0.5) * dx
    y = (np.arange(ny, dtype=np.float64) + 0.5) * dy
    return np.meshgrid(x, y, indexing="ij")


def _as_warp(arr: np.ndarray, *, dtype, device: str) -> wp.array:
    return wp.array(np.ascontiguousarray(arr), dtype=dtype, device=device)


@wp.func
def _clamp_j(j: wp.float64, floor: wp.float64) -> wp.float64:
    out = j
    if out < floor:
        out = floor
    return out


@wp.func
def _smoothstep5(eta: wp.float64) -> wp.float64:
    z = eta
    if z < wp.float64(0.0):
        z = wp.float64(0.0)
    if z > wp.float64(1.0):
        z = wp.float64(1.0)
    return z * z * z * (wp.float64(10.0) - wp.float64(15.0) * z + wp.float64(6.0) * z * z)


@wp.func
def _piola_vec(
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
    j_floor: wp.float64,
) -> wp.vec4d:
    if model == MATERIAL_SVK:
        C00 = F11 * F11 + F21 * F21
        C01 = F11 * F12 + F21 * F22
        C11 = F12 * F12 + F22 * F22
        E00 = wp.float64(0.5) * (C00 - wp.float64(1.0))
        E01 = wp.float64(0.5) * C01
        E11 = wp.float64(0.5) * (C11 - wp.float64(1.0))
        trE = E00 + E11
        S00 = lam * trE + wp.float64(2.0) * mu * E00
        S01 = wp.float64(2.0) * mu * E01
        S11 = lam * trE + wp.float64(2.0) * mu * E11
        return wp.vec4d(
            F11 * S00 + F12 * S01,
            F11 * S01 + F12 * S11,
            F21 * S00 + F22 * S01,
            F21 * S01 + F22 * S11,
        )

    J = _clamp_j(F11 * F22 - F12 * F21, j_floor)
    H00 = F22 / J
    H01 = -F21 / J
    H10 = -F12 / J
    H11 = F11 / J
    logJ = wp.log(J)
    volumetric = lam * logJ
    if model == MATERIAL_NEO_HOOKE:
        volumetric = wp.float64(0.5) * lam * (J * J - wp.float64(1.0))
        return wp.vec4d(
            mu * (F11 - H00) + volumetric * H00,
            mu * (F12 - H01) + volumetric * H01,
            mu * (F21 - H10) + volumetric * H10,
            mu * (F22 - H11) + volumetric * H11,
        )
    if model == MATERIAL_LOG_NEO_HOOKE:
        return wp.vec4d(
            mu * (F11 - H00) + volumetric * H00,
            mu * (F12 - H01) + volumetric * H01,
            mu * (F21 - H10) + volumetric * H10,
            mu * (F22 - H11) + volumetric * H11,
        )

    F2 = F11 * F11 + F12 * F12 + F21 * F21 + F22 * F22
    I1 = F2 + wp.float64(1.0)
    I2 = J * J + F2
    jm23 = wp.exp(wp.float64(-2.0 / 3.0) * logJ)
    jm43 = wp.exp(wp.float64(-4.0 / 3.0) * logJ)
    H_scale_1 = wp.float64(-2.0 / 3.0) * I1
    H_scale_2 = wp.float64(2.0) * J * J - wp.float64(4.0 / 3.0) * I2
    G100 = jm23 * (wp.float64(2.0) * F11 + H_scale_1 * H00)
    G101 = jm23 * (wp.float64(2.0) * F12 + H_scale_1 * H01)
    G110 = jm23 * (wp.float64(2.0) * F21 + H_scale_1 * H10)
    G111 = jm23 * (wp.float64(2.0) * F22 + H_scale_1 * H11)
    if model == MATERIAL_MOONEY_RIVLIN or model == MATERIAL_SUGIYAMA_SVK:
        G200 = jm43 * (wp.float64(2.0) * F11 + H_scale_2 * H00)
        G201 = jm43 * (wp.float64(2.0) * F12 + H_scale_2 * H01)
        G210 = jm43 * (wp.float64(2.0) * F21 + H_scale_2 * H10)
        G211 = jm43 * (wp.float64(2.0) * F22 + H_scale_2 * H11)
        if model == MATERIAL_SUGIYAMA_SVK:
            A = jm23 * I1 - wp.float64(3.0)
            c2 = -wp.float64(0.5) * mu
            c3 = (mat_p0 + wp.float64(2.0) * mu) / wp.float64(8.0)
            scale = mu + wp.float64(2.0) * c3 * A
            return wp.vec4d(
                scale * G100 + c2 * G200 + volumetric * H00,
                scale * G101 + c2 * G201 + volumetric * H01,
                scale * G110 + c2 * G210 + volumetric * H10,
                scale * G111 + c2 * G211 + volumetric * H11,
            )
        return wp.vec4d(
            mat_p0 * G100 + mat_p1 * G200 + volumetric * H00,
            mat_p0 * G101 + mat_p1 * G201 + volumetric * H01,
            mat_p0 * G110 + mat_p1 * G210 + volumetric * H10,
            mat_p0 * G111 + mat_p1 * G211 + volumetric * H11,
        )
    if model == MATERIAL_YEOH:
        A = jm23 * I1 - wp.float64(3.0)
        scale = mat_p0 + wp.float64(2.0) * mat_p1 * A + wp.float64(3.0) * mat_p2 * A * A
        return wp.vec4d(
            scale * G100 + volumetric * H00,
            scale * G101 + volumetric * H01,
            scale * G110 + volumetric * H10,
            scale * G111 + volumetric * H11,
        )
    if model == MATERIAL_GENT:
        A = jm23 * I1 - wp.float64(3.0)
        denom = wp.float64(1.0) - A / mat_p0
        if denom < j_floor:
            denom = j_floor
        scale = wp.float64(0.5) * mu / denom
        return wp.vec4d(
            scale * G100 + volumetric * H00,
            scale * G101 + volumetric * H01,
            scale * G110 + volumetric * H10,
            scale * G111 + volumetric * H11,
        )
    return wp.vec4d(
        mu * (F11 - H00) + volumetric * H00,
        mu * (F12 - H01) + volumetric * H01,
        mu * (F21 - H10) + volumetric * H10,
        mu * (F22 - H11) + volumetric * H11,
    )


@wp.func
def _state_comp(
    a: int,
    vx: wp.float64,
    vy: wp.float64,
    F11: wp.float64,
    F12: wp.float64,
    F21: wp.float64,
    F22: wp.float64,
) -> wp.float64:
    if a == 0:
        return vx
    if a == 1:
        return vy
    if a == 2:
        return F11
    if a == 3:
        return F12
    if a == 4:
        return F21
    return F22


@wp.func
def _flux_x_comp(
    a: int,
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
    j_floor: wp.float64,
) -> wp.float64:
    P = _piola_vec(F11, F12, F21, F22, model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor)
    if a == 0:
        return -P[0]
    if a == 1:
        return -P[2]
    if a == 2:
        return -vx
    if a == 3:
        return wp.float64(0.0)
    if a == 4:
        return -vy
    return wp.float64(0.0)


@wp.func
def _flux_y_comp(
    a: int,
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
    j_floor: wp.float64,
) -> wp.float64:
    P = _piola_vec(F11, F12, F21, F22, model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor)
    if a == 0:
        return -P[1]
    if a == 1:
        return -P[3]
    if a == 2:
        return wp.float64(0.0)
    if a == 3:
        return -vx
    if a == 4:
        return wp.float64(0.0)
    return -vy


@wp.func
def _equilibrium_comp(
    q: int,
    a: int,
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
    cx = wp.float64(CX_I[q])
    cy = wp.float64(CY_I[q])
    val = _state_comp(a, vx, vy, F11, F12, F21, F22)
    phix = _flux_x_comp(
        a, vx, vy, F11, F12, F21, F22, model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor
    )
    phiy = _flux_y_comp(
        a, vx, vy, F11, F12, F21, F22, model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor
    )
    return wp.float64(0.25) * (val + (wp.float64(2.0) / lattice_speed) * (cx * phix + cy * phiy))


@wp.func
def _ricker_wavelet(
    t: wp.float64, amplitude: wp.float64, width: wp.float64, t0: wp.float64
) -> wp.float64:
    z = (t - t0) / width
    pref = wp.float64(2.0) * amplitude / (wp.sqrt(wp.float64(3.0) * width) * wp.sqrt(wp.sqrt(PI)))
    return pref * (wp.float64(1.0) - z * z) * wp.exp(wp.float64(-0.5) * z * z)


@wp.func
def _boundary_value_comp(
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
    t: wp.float64,
) -> wp.float64:
    if mode == VALUE_ZERO:
        return wp.float64(0.0)
    if mode == VALUE_CONSTANT:
        if comp == 0:
            return p0
        return p1
    if mode == VALUE_RAMP:
        ramp_time = p2
        scale = wp.float64(1.0)
        if ramp_time > wp.float64(0.0) and t < ramp_time:
            s = wp.sin(PI * t / ramp_time)
            scale = s * s
        if comp == 0:
            return p0 * scale
        return p1 * scale
    if mode == VALUE_SINE2_HOLD:
        active_time = p2
        sine_denominator = p3
        scale = wp.float64(1.0)
        if active_time > wp.float64(0.0) and t < active_time:
            denom = sine_denominator
            if denom <= wp.float64(0.0):
                denom = wp.float64(2.0) * active_time
            s = wp.sin(PI * t / denom)
            scale = s * s
        if comp == 0:
            return p0 * scale
        return p1 * scale
    if mode == VALUE_SHEAR_VELOCITY_X:
        if comp == 0 and t < p1:
            return p0 * wp.sin(PI * t / p1)
        return wp.float64(0.0)
    if mode == VALUE_WAVE_BEAM_RICKER_Y:
        if comp != 1 or t > p3:
            return wp.float64(0.0)
        value = _ricker_wavelet(t, p0, p1, p2)
        if p6 > wp.float64(0.0):
            layer = p6 * p5
            if layer < wp.float64(1.0e-30):
                layer = wp.float64(1.0e-30)
            window = _smoothstep5(y / layer) * _smoothstep5((p4 - y) / layer)
            if p7 > wp.float64(1.0e-14):
                window = window / p7
            value = value * window
        return value
    if mode == VALUE_AFFINE_VELOCITY:
        if p6 > wp.float64(0.0) and t > p6:
            return wp.float64(0.0)
        if comp == 0:
            return p0 * x + p1 * y + p2
        return p3 * x + p4 * y + p5
    return wp.float64(0.0)


@wp.func
def _source_comp(
    a: int,
    mode: int,
    p0: wp.float64,
    p1: wp.float64,
    p2: wp.float64,
    p3: wp.float64,
    p4: wp.float64,
    p5: wp.float64,
    vx: wp.float64,
    vy: wp.float64,
) -> wp.float64:
    if mode == SOURCE_CONSTANT:
        if a == 0:
            return p0
        if a == 1:
            return p1
        if a == 2:
            return p2
        if a == 3:
            return p3
        if a == 4:
            return p4
        return p5
    if mode == SOURCE_DAMPING:
        if a == 0:
            return -p0 * vx
        if a == 1:
            return -p0 * vy
    return wp.float64(0.0)


@wp.func
def _normal_residual(
    F11: wp.float64,
    F12: wp.float64,
    F21: wp.float64,
    F22: wp.float64,
    normal_axis: int,
    sign: wp.float64,
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
    P = _piola_vec(F11, F12, F21, F22, model, lam, mu, mat_p0, mat_p1, mat_p2, j_floor)
    if normal_axis == 0:
        return wp.vec2d(sign * P[0] - tx, sign * P[2] - ty)
    return wp.vec2d(sign * P[1] - tx, sign * P[3] - ty)


@wp.func
def _solve_normal_column(
    F11_in: wp.float64,
    F12_in: wp.float64,
    F21_in: wp.float64,
    F22_in: wp.float64,
    normal_axis: int,
    sign: wp.float64,
    tx: wp.float64,
    ty: wp.float64,
    model: int,
    lam: wp.float64,
    mu: wp.float64,
    mat_p0: wp.float64,
    mat_p1: wp.float64,
    mat_p2: wp.float64,
    j_floor: wp.float64,
) -> wp.vec4d:
    x0 = F11_in
    x1 = F21_in
    if normal_axis == 1:
        x0 = F12_in
        x1 = F22_in

    F11 = F11_in
    F12 = F12_in
    F21 = F21_in
    F22 = F22_in
    eps = wp.float64(1.0e-6)

    for _it in range(12):
        if normal_axis == 0:
            F11 = x0
            F21 = x1
        else:
            F12 = x0
            F22 = x1

        r = _normal_residual(
            F11,
            F12,
            F21,
            F22,
            normal_axis,
            sign,
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
        rnorm = wp.sqrt(r[0] * r[0] + r[1] * r[1])
        if rnorm < wp.float64(1.0e-10):
            break

        if normal_axis == 0:
            rp0 = _normal_residual(
                F11 + eps,
                F12,
                F21,
                F22,
                normal_axis,
                sign,
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
            rp1 = _normal_residual(
                F11,
                F12,
                F21 + eps,
                F22,
                normal_axis,
                sign,
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
        else:
            rp0 = _normal_residual(
                F11,
                F12 + eps,
                F21,
                F22,
                normal_axis,
                sign,
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
            rp1 = _normal_residual(
                F11,
                F12,
                F21,
                F22 + eps,
                normal_axis,
                sign,
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
        for _ls in range(10):
            c0 = x0 + step * delta0
            c1 = x1 + step * delta1
            test_F11 = F11_in
            test_F12 = F12_in
            test_F21 = F21_in
            test_F22 = F22_in
            if normal_axis == 0:
                test_F11 = c0
                test_F21 = c1
            else:
                test_F12 = c0
                test_F22 = c1
            J = test_F11 * test_F22 - test_F12 * test_F21
            if J > j_floor:
                x0 = c0
                x1 = c1
                break
            step = wp.float64(0.5) * step

    if normal_axis == 0:
        F11 = x0
        F21 = x1
    else:
        F12 = x0
        F22 = x1
    return wp.vec4d(F11, F12, F21, F22)


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
def _stream_kernel(
    fpost: wp.array4d(dtype=wp.float64),
    f1: wp.array4d(dtype=wp.float64),
    nx: int,
    ny: int,
    periodic_x: int,
    periodic_y: int,
):
    i, j, k = wp.tid()
    for q in range(4):
        ii = i + CX_I[q]
        jj = j + CY_I[q]
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
        if valid:
            for a in range(6):
                f1[q * 6 + a, ii, jj, k] = fpost[q * 6 + a, i, j, k]


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
    length_x: wp.float64,
    length_y: wp.float64,
    time_mid: wp.float64,
    lattice_speed: wp.float64,
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

    n_nodes = ny
    if side == SIDE_BOTTOM or side == SIDE_TOP:
        n_nodes = nx
    if s >= n_nodes:
        return

    i = wp.int32(0)
    j = wp.int32(s)
    nx_n = wp.int32(-1)
    ny_n = wp.int32(0)
    q_in = wp.int32(0)
    q_out = wp.int32(2)
    normal_axis = wp.int32(0)
    sign = wp.float64(-1.0)
    if side == SIDE_RIGHT:
        i = nx - 1
        j = s
        nx_n = 1
        ny_n = 0
        q_in = 2
        q_out = 0
        normal_axis = 0
        sign = wp.float64(1.0)
    elif side == SIDE_BOTTOM:
        i = s
        j = 0
        nx_n = 0
        ny_n = -1
        q_in = 1
        q_out = 3
        normal_axis = 1
        sign = wp.float64(-1.0)
    elif side == SIDE_TOP:
        i = s
        j = ny - 1
        nx_n = 0
        ny_n = 1
        q_in = 3
        q_out = 1
        normal_axis = 1
        sign = wp.float64(1.0)

    dx = length_x / wp.float64(nx)
    dy = length_y / wp.float64(ny)
    x = (wp.float64(i) + wp.float64(0.5)) * dx
    y = (wp.float64(j) + wp.float64(0.5)) * dy
    x_bc = x
    y_bc = y
    if side == SIDE_LEFT:
        x_bc = wp.float64(0.0)
    elif side == SIDE_RIGHT:
        x_bc = length_x
    elif side == SIDE_BOTTOM:
        y_bc = wp.float64(0.0)
    elif side == SIDE_TOP:
        y_bc = length_y

    mode = value_mode[side]
    p0 = value_params[side, 0]
    p1 = value_params[side, 1]
    p2 = value_params[side, 2]
    p3 = value_params[side, 3]
    p4 = value_params[side, 4]
    p5 = value_params[side, 5]
    p6 = value_params[side, 6]
    p7 = value_params[side, 7]

    if side_kind[side] == BC_DIRICHLET:
        vdx = _boundary_value_comp(0, mode, p0, p1, p2, p3, p4, p5, p6, p7, x_bc, y_bc, time_mid)
        vdy = _boundary_value_comp(1, mode, p0, p1, p2, p3, p4, p5, p6, p7, x_bc, y_bc, time_mid)
        S0 = wp.float64(0.5) * vdx
        S1 = wp.float64(0.5) * vdy
        S2 = wp.float64(nx_n) * vdx / lattice_speed
        S3 = wp.float64(ny_n) * vdx / lattice_speed
        S4 = wp.float64(nx_n) * vdy / lattice_speed
        S5 = wp.float64(ny_n) * vdy / lattice_speed
        f1[q_in * 6 + 0, i, j, 0] = -fpost[q_out * 6 + 0, i, j, 0] + S0
        f1[q_in * 6 + 1, i, j, 0] = -fpost[q_out * 6 + 1, i, j, 0] + S1
        f1[q_in * 6 + 2, i, j, 0] = fpost[q_out * 6 + 2, i, j, 0] + S2
        f1[q_in * 6 + 3, i, j, 0] = fpost[q_out * 6 + 3, i, j, 0] + S3
        f1[q_in * 6 + 4, i, j, 0] = fpost[q_out * 6 + 4, i, j, 0] + S4
        f1[q_in * 6 + 5, i, j, 0] = fpost[q_out * 6 + 5, i, j, 0] + S5
    elif side_kind[side] == BC_NEUMANN:
        tx = _boundary_value_comp(0, mode, p0, p1, p2, p3, p4, p5, p6, p7, x_bc, y_bc, time_mid)
        ty = _boundary_value_comp(1, mode, p0, p1, p2, p3, p4, p5, p6, p7, x_bc, y_bc, time_mid)
        F_boundary = _solve_normal_column(
            U[2, i, j, 0],
            U[3, i, j, 0],
            U[4, i, j, 0],
            U[5, i, j, 0],
            normal_axis,
            sign,
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
        f1[q_in * 6 + 0, i, j, 0] = fpost[q_out * 6 + 0, i, j, 0] + tx / lattice_speed
        f1[q_in * 6 + 1, i, j, 0] = fpost[q_out * 6 + 1, i, j, 0] + ty / lattice_speed
        f1[q_in * 6 + 2, i, j, 0] = -fpost[q_out * 6 + 2, i, j, 0] + wp.float64(0.5) * F_boundary[0]
        f1[q_in * 6 + 3, i, j, 0] = -fpost[q_out * 6 + 3, i, j, 0] + wp.float64(0.5) * F_boundary[1]
        f1[q_in * 6 + 4, i, j, 0] = -fpost[q_out * 6 + 4, i, j, 0] + wp.float64(0.5) * F_boundary[2]
        f1[q_in * 6 + 5, i, j, 0] = -fpost[q_out * 6 + 5, i, j, 0] + wp.float64(0.5) * F_boundary[3]


@wp.kernel
def _refresh_kernel(
    f0: wp.array4d(dtype=wp.float64),
    u: wp.array4d(dtype=wp.float64),
    u_star: wp.array4d(dtype=wp.float64),
    U: wp.array4d(dtype=wp.float64),
    P_field: wp.array4d(dtype=wp.float64),
    sigma: wp.array4d(dtype=wp.float64),
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


class WarpNonlinearElasticLBM2D:
    """Warp-backed D2Q4x6 nonlinear elastic LBM on a cell-centered rectangle."""

    def __init__(
        self,
        nx: int,
        ny: int,
        *,
        material: WarpHyperelasticMaterial,
        boundaries: RectangularBoundarySet | None = None,
        length_x: float = 1.0,
        length_y: float = 1.0,
        lattice_speed: float = 5.0,
        collision_omega: float = 1.8,
        device: str | None = None,
        check_linearized_cfl: bool = True,
        jacobian_floor: float = 1.0e-12,
        source_mode: SourceName = "none",
        source_vector: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        damping_gamma: float = 0.0,
    ) -> None:
        if nx < 4 or ny < 4:
            raise ValueError("nx and ny must be at least 4")
        self.nx = int(nx)
        self.ny = int(ny)
        self.length_x = float(length_x)
        self.length_y = float(length_y)
        self.dx = self.length_x / float(self.nx)
        self.dy = self.length_y / float(self.ny)
        if abs(self.dx - self.dy) > 1e-14:
            raise ValueError("D2Q4 requires dx=dy")
        self.material = material
        self.boundaries = boundaries or RectangularBoundarySet.periodic()
        self.boundaries.validate_periodic_pairs()
        self.lattice_speed = float(lattice_speed)
        self.collision_omega = float(collision_omega)
        if not (0.0 < self.collision_omega <= 2.0):
            raise ValueError("collision_omega must be in (0, 2]")
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
        self.source_mode = source_id(source_mode)
        self.source_params = np.zeros(NCOMP, dtype=np.float64)
        if self.source_mode == SOURCE_CONSTANT:
            for idx, value in enumerate(tuple(source_vector)[:NCOMP]):
                self.source_params[idx] = float(value)
        elif self.source_mode == SOURCE_DAMPING:
            if damping_gamma < 0.0:
                raise ValueError("damping_gamma must be non-negative")
            self.source_params[0] = float(damping_gamma)

        self.x, self.y = reference_grid(self.nx, self.ny, self.length_x, self.length_y)
        self.f0 = wp.zeros((FQ, self.nx, self.ny, 1), dtype=wp.float64, device=self.device)
        self.f1 = wp.zeros((FQ, self.nx, self.ny, 1), dtype=wp.float64, device=self.device)
        self.fpost = wp.zeros((FQ, self.nx, self.ny, 1), dtype=wp.float64, device=self.device)
        self.u = wp.zeros((2, self.nx, self.ny, 1), dtype=wp.float64, device=self.device)
        self.u_star = wp.zeros((2, self.nx, self.ny, 1), dtype=wp.float64, device=self.device)
        self.U = wp.zeros((NCOMP, self.nx, self.ny, 1), dtype=wp.float64, device=self.device)
        self.P = wp.zeros((4, self.nx, self.ny, 1), dtype=wp.float64, device=self.device)
        self.sigma = wp.zeros((4, self.nx, self.ny, 1), dtype=wp.float64, device=self.device)
        self.invalid = wp.zeros((1,), dtype=wp.int32, device=self.device)
        self.side_kind = _as_warp(self.boundaries.side_kind, dtype=wp.int32, device=self.device)
        self.value_mode = _as_warp(self.boundaries.value_mode, dtype=wp.int32, device=self.device)
        self.value_params = _as_warp(self.boundaries.params, dtype=wp.float64, device=self.device)
        self.initialize_identity()

    def set_boundaries(self, boundaries: RectangularBoundarySet) -> None:
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
        U = np.zeros((NCOMP, self.nx, self.ny), dtype=np.float64)
        U[2] = 1.0
        U[5] = 1.0
        self.initialize_from_numpy(
            U=U,
            displacement=np.zeros((2, self.nx, self.ny), dtype=np.float64),
            init_order=init_order,
        )

    def initialize_sine_displacement(
        self,
        amplitude: float = 1.0e-3,
        mode_x: int = 1,
        mode_y: int = 1,
        *,
        init_order: int = 1,
    ) -> None:
        kx = 2.0 * math.pi * int(mode_x) / self.length_x
        ky = 2.0 * math.pi * int(mode_y) / self.length_y
        x, y = self.x, self.y
        ux = float(amplitude) * np.sin(kx * x) * np.cos(ky * y)
        uy = -float(amplitude) * np.cos(kx * x) * np.sin(ky * y)
        U = np.zeros((NCOMP, self.nx, self.ny), dtype=np.float64)
        U[2] = 1.0 + float(amplitude) * kx * np.cos(kx * x) * np.cos(ky * y)
        U[3] = -float(amplitude) * ky * np.sin(kx * x) * np.sin(ky * y)
        U[4] = float(amplitude) * kx * np.sin(kx * x) * np.sin(ky * y)
        U[5] = 1.0 - float(amplitude) * ky * np.cos(kx * x) * np.cos(ky * y)
        self.initialize_from_numpy(U=U, displacement=np.stack([ux, uy]), init_order=init_order)

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
        if U.shape != (NCOMP, self.nx, self.ny):
            raise ValueError(f"U must have shape {(NCOMP, self.nx, self.ny)}")
        displacement = (
            np.zeros((2, self.nx, self.ny), dtype=np.float64)
            if displacement is None
            else np.asarray(displacement, dtype=np.float64)
        )
        if displacement.shape != (2, self.nx, self.ny):
            raise ValueError(f"displacement must have shape {(2, self.nx, self.ny)}")
        if enforce_dirichlet_tangent:
            U = _enforce_dirichlet_tangent_columns(
                U,
                displacement,
                self.boundaries,
                dx=self.dx,
                dy=self.dy,
            )
        if init_order == 1:
            f = _equilibrium_np(U, self.material, self.lattice_speed)
            source = _source_np(U, self.source_mode, self.source_params)
            f -= (0.5 * self.dt / float(Q)) * source[None, ...]
        elif init_order == 2:
            f = _second_order_populations_np(
                U,
                self.material,
                self.lattice_speed,
                dx=self.dx,
                dy=self.dy,
                dt=self.dt,
                periodic_x=self.boundaries.periodic_x,
                periodic_y=self.boundaries.periodic_y,
                source_mode=self.source_mode,
                source_params=self.source_params,
            )
        else:
            raise ValueError("init_order must be 1 or 2")
        u_star0 = displacement - 0.5 * self.dt * U[:2]
        displacement_wp = _as_warp(displacement[..., None], dtype=wp.float64, device=self.device)
        wp.copy(self.f0, _as_warp(_flatten_pop(f)[..., None], dtype=wp.float64, device=self.device))
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
        max_side_nodes = max(self.nx, self.ny)
        for side in (SIDE_LEFT, SIDE_RIGHT, SIDE_BOTTOM, SIDE_TOP):
            if self.boundaries.side_kind[side] != BC_PERIODIC:
                wp.launch(
                    _apply_boundary_side_kernel,
                    dim=max_side_nodes,
                    inputs=[
                        self.f0,
                        self.f0,
                        self.U,
                        self.side_kind,
                        self.value_mode,
                        self.value_params,
                        self.nx,
                        self.ny,
                        self.length_x,
                        self.length_y,
                        float(boundary_time),
                        self.lattice_speed,
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
        wp.launch(
            _clear_kernel, dim=(FQ, self.nx, self.ny, 1), inputs=[self.f1], device=self.device
        )
        wp.launch(
            _stream_kernel,
            dim=self.shape,
            inputs=[
                self.fpost,
                self.f1,
                self.nx,
                self.ny,
                int(self.boundaries.periodic_x),
                int(self.boundaries.periodic_y),
            ],
            device=self.device,
        )
        max_side_nodes = max(self.nx, self.ny)
        for side in (SIDE_LEFT, SIDE_RIGHT, SIDE_BOTTOM, SIDE_TOP):
            if self.boundaries.side_kind[side] != BC_PERIODIC:
                wp.launch(
                    _apply_boundary_side_kernel,
                    dim=max_side_nodes,
                    inputs=[
                        self.fpost,
                        self.f1,
                        self.U,
                        self.side_kind,
                        self.value_mode,
                        self.value_params,
                        self.nx,
                        self.ny,
                        self.length_x,
                        self.length_y,
                        self.time + 0.5 * self.dt,
                        self.lattice_speed,
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
            "steps": self.steps,
            "time": float(self.time),
            "dx": float(self.dx),
            "dt": float(self.dt),
            "lattice_speed": float(self.lattice_speed),
            "collision_omega": float(self.collision_omega),
            "source_mode": source_name(self.source_mode),
            "source_p0": float(self.source_params[0]),
            "source_p1": float(self.source_params[1]),
            "source_p2": float(self.source_params[2]),
            "source_p3": float(self.source_params[3]),
            "source_p4": float(self.source_params[4]),
            "source_p5": float(self.source_params[5]),
            "linearized_stability_ratio": float(
                self.material.linearized_stability_ratio(self.lattice_speed)
            ),
            "periodic_x": bool(self.boundaries.periodic_x),
            "periodic_y": bool(self.boundaries.periodic_y),
            "max_abs_u": float(np.max(np.sqrt(np.sum(u * u, axis=0)))),
            "max_abs_v": float(np.max(np.sqrt(U[0] * U[0] + U[1] * U[1]))),
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
            lattice_speed=np.array(self.lattice_speed),
            collision_omega=np.array(self.collision_omega),
            source_mode=np.array(source_name(self.source_mode)),
            source_params=self.source_params,
            model=np.array(self.material.model),
            lam=np.array(self.material.lam),
            mu=np.array(self.material.mu),
            material_params=np.asarray(self.material.params, dtype=np.float64),
        )
