# Numerical method

The solver advances the material velocity and deformation gradient in reference
coordinates. A Cartesian lattice remains fixed while the material undergoes finite
deformation. The compressible neo-Hookean benchmarks use

\[
W(F)=\frac{\mu}{2}(F:F-D)
 -\left(\mu+\frac{\lambda}{2}\right)\log J
 +\frac{\lambda}{4}(J^2-1),\qquad J=\det F>0.
\]

The first Piola stress is `P = dW/dF`; reported Cauchy stresses use `sigma = P F^T / J`.
The benchmark material has `mu=1`, `rho0=1`, and Poisson ratio `0.2`.

## Bulk populations

With `U=(v,F)` and reference-coordinate flux `G_A(U)`, each axial lattice direction
stores a vector-valued population. The equilibrium is

\[
f^{eq}_{\pm A}(U)=\frac{U}{2D}\pm\frac{G_A(U)}{2c}.
\]

D2Q4×6 stores four six-component populations; D3Q6×12 stores six twelve-component
populations. The implementation evaluates constitutive stresses at material nodes
and advances the populations with explicit BGK collision and streaming.

## Curved boundaries

A level set defines the reference material by `phi(X) <= 0`. Geometry preprocessing
stores active nodes, cut-link fractions, physical intersection points, and reference
surface normals. These data remain fixed throughout a total-Lagrangian calculation.

Velocity Dirichlet links use the population-pair relation and distance-dependent
Bouzidi–Firdaouss–Lallemand interpolation. The interpolation retains an upstream-node
fallback when a cut stencil is truncated. Initialization reads a frozen population
buffer so all cut links evaluate the boundary rule from the same input state.

At a curved traction boundary, the surface normal generally differs from the lattice
direction. The prescribed nominal traction `P(F_b)n=T` therefore needs a complete
boundary deformation state to determine the lattice-direction stress flux. The
solver predicts tangential deformation from local interior data and solves for the
normal image of `F_b` using the acoustic-tensor Newton method.

A local quadratic stencil supplies state derivatives. Writing `U_c` for the boundary
state extrapolated to the adjacent material node and `w_d` for its characteristic
derivative, the incoming traction population is reconstructed as

\[
f_d=f^{eq}_d(U_c)
 -\frac{\Delta t}{\omega}(f^{eq}_d)'(U_c)w_d
 +\frac{\Delta t}{2D}\left(\frac{1}{\omega}-\frac12\right)B(U_c).
\]

The last term follows the BGK source discretization and vanishes at `omega=2`.
In mixed component conditions, prescribed velocity components retain their
Dirichlet reconstruction while traction components receive the characteristic update.

## Local kinematic update

Curved cut links and truncated stencils couple boundary reconstruction to the
discrete relation between the evolved deformation gradient and velocity-integrated
displacement. Each traction time step uses a nearest-neighbor displacement sweep,
then blends the evolved deformation gradient with `I + grad(u)`.

Where both neighboring nodes are active, the displacement gradient is centered.
When only one side is available, the second-order one-sided derivative is

\[
\partial_Au=\frac{s(-3u_0+4u_1-u_2)}{2h},\qquad s\in\{-1,+1\}.
\]

The sign selects the interior direction. If the second interior node is unavailable,
the implementation uses a first-order difference; if neither side is available, the
derivative is zero. The displacement sweep reads one lattice layer and the gradient
reads two, giving a combined dependency radius of three. Characteristic reconstruction
also uses a fixed radius-three stencil; six layers form a conservative dependency
bound for the complete ordered update.

The common traction settings are defined in
[traction_settings.py](../cases/traction_settings.py). All paper commands select the
Newton traction solver and apply one sweep and one gradient update per time step.

## Implementation map

| Component | Source |
|---|---|
| 2D bulk, equilibrium, stress and source | [vector_nonlinear_elastic_warp.py](../src/curved_lbm/two_d/vector_nonlinear_elastic_warp.py) |
| 2D embedded geometry and boundary solver | [curved_boundary_warp.py](../src/curved_lbm/two_d/curved_boundary_warp.py) |
| 3D bulk and constitutive operations | [vector_nonlinear_elastic_warp3d.py](../src/curved_lbm/three_d/vector_nonlinear_elastic_warp3d.py) |
| 3D active-node storage and boundaries | [curved_boundary_warp3d.py](../src/curved_lbm/three_d/curved_boundary_warp3d.py) |
| 3D characteristic stencil weights | [characteristic_stencil3d.py](../src/curved_lbm/three_d/characteristic_stencil3d.py) |

The kernels use double precision and local array operations. Geometry setup, reference
solutions, diagnostics, and plotting run on the host; time-stepping kernels run on the
selected Warp device.
