# Finite-element references and data

The repository contains the small reference files required for the manuscript LBM/FEM
comparisons. Their SHA-256 checksums and physical parameters are recorded in
[manifest.json](../reference_data/manifest.json).

| Reference | File | Content |
|---|---|---|
| Radial annulus | `annulus/annulus_traction_fem_n196.npz` | FEM fields sampled on the 196² comparison lattice |
| Superellipse shell | `superellipse/superellipse_mixed_fem_n384.npz` | FEM fields sampled on the 384² comparison lattice |
| Eccentric-hole pulse | `pulse/eccentric_pulse_fem_history_n128.npz` | Probe displacement, velocity, energy and maximum-displacement histories |
| Finite tube | `tube/fem_profiles.csv` | Axial profiles of displacement, rotation and volume ratio |

Paths in this table are relative to `reference_data/`. The tube table contains the
reference and kinematic columns; LBM columns are rebuilt from each simulation.
The planar files include active/valid masks to distinguish physical comparison nodes.

The affine annulus and ellipsoid use exact affine solutions computed by their case
scripts. The spherical-shell script solves its nonlinear radial BVP with SciPy.
Simulation datasets are archived on [Zenodo](https://doi.org/10.5281/zenodo.20572218).

## Regenerating planar FEM references

FEniCSx is optional and runs in a separate environment. The reference generators use
FEniCSx 0.10, Basix 0.10, UFL, Gmsh, mpi4py, and petsc4py. The local verification
environment used DOLFINx 0.10.0, Gmsh 4.15.2, and petsc4py 3.25.1.

With that environment activated, generate references directly:

```bash
python cases/radial_annulus.py --fem-only --results-dir results/fem-annulus
python cases/superellipse_shell.py --fem-only --results-dir results/fem-superellipse
python cases/eccentric_pulse.py --fem-only --results-dir results/fem-pulse
```

The planar generators run with one MPI rank. They are self-contained scripts and
do not import Warp while performing FEM calculations. To launch them from a Warp
environment, set `FENICSX_PYTHON` to the FEM environment's Python executable or pass
`--fenicsx-python` explicitly, together with `--force-fem`.

The bundled pulse history has fixed load, probe locations, averaging radius, material,
and time interval. A modified pulse requires a matching `--fem-reference`, a regenerated
reference, or `--no-fem` for an LBM-only calculation.

## Regenerating the tube reference

[cases/fem/tube_reference.py](../cases/fem/tube_reference.py) solves the quasi-static
stretched and twisted tube on a structured annular hexahedral mesh. The reference uses
6 radial, 48 circumferential, and 60 axial elements, degree 1, and 20 load increments:

```bash
mpiexec -n 8 python cases/fem/tube_reference.py \
  --target-strain 0.5 --target-twist-angle 0.7853981633974483 \
  --nr 6 --ntheta 48 --nz 60 --degree 1 \
  --load-steps 20 --newton-max-it 40 --results-dir results/fem-tube
```

Its sample grid uses 8 radial, 96 circumferential, and 120 axial samples and 79 profile
bins. The generated NPZ contains sampled displacements, deformation gradients, and
profiles; the accompanying JSON records mesh, material, loading, and convergence.

Convert the sampled fields into a table accepted by `cases/tube_profiles.py`:

```bash
python cases/fem/export_tube_profiles.py \
  --input results/fem-tube/tube_pull_torsion_coarse_fem_reference.npz \
  --output results/fem-tube/fem_profiles.csv
```

Pass this table with `--reference-profile` when assembling new tube comparisons.
