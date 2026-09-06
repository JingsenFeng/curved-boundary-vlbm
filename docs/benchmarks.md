# Reproducing the benchmarks

Run commands from the repository root after installation. The full reproduction driver
uses the configurations below and writes one log and execution record per case.

```bash
python cases/reproduce.py --list
python cases/reproduce.py --dry-run --device cuda:0
python cases/reproduce.py --device cuda:0
```

## Grids and loading

| Case | Grid sequence | Final nondimensional time | Loading |
|---|---|---:|---|
| Affine annulus | 64², 96², 128², 192², 256², 384² | 0.2 | Exact affine velocity at both walls |
| Radial annulus | 196² | 5 | Outer radial traction; ramp time 2 |
| Rotated superellipse | 384² | 20 | Outer nominal traction `(0, 0.125)`; ramp time 2 |
| Eccentric-hole pulse | 128² | 8 | Inner pressure amplitude 1.5; sine-squared pulse duration 16 |
| Rotated ellipsoid | 80³, 120³, 160³, 240³, 320³ | approximately 0.459907 | Exact affine velocity; target maximum displacement 0.1 |
| Spherical shell | 200³ | 5 | Radial nominal traction from the nonlinear BVP |
| Tube `r10` | 26×26×100 | 100 | 50% axial strain and 45° twist; ramp time 10 |
| Tube `r20` | 50×50×200 | 100 | Same loading |
| Tube `r30` | 76×76×300 | 100 | Same loading |

Final times are rounded to an integer number of lattice steps. The pulse calculation
covers the rising half of the pressure pulse. Annulus, superellipse, shell, and tube
calculations relax with `gamma=10`; the transient pulse uses `gamma=0`.

All traction cases use the constants in [traction_settings.py](../cases/traction_settings.py):

| Parameter | Value |
|---|---:|
| Lattice speed `c` | 10 |
| BGK relaxation `omega` | 1.8 |
| Displacement relaxation `alpha_u` | 0.85 |
| Deformation-gradient blend `theta_F` | 0.5 |
| Dirichlet anchor weight `w_D` | 100 |
| Displacement sweeps per step | 1 |
| Deformation-gradient updates per step | 1 |

The pure Dirichlet annulus uses `omega=2`, second-order initialization, and the
`compat_bfl` interpolation option. The ellipsoid uses `omega=1.8`, first-order
initialization, and `local_f` dispatch with Dirichlet data. Both omit the traction
kinematic corrections.

## Selecting calculations

```bash
python cases/reproduce.py --cases dirichlet --device cuda:0
python cases/reproduce.py --cases traction --device cuda:0
python cases/reproduce.py --cases annulus,pulse2d --device cuda:0
python cases/reproduce.py --cases tube10,tube20,tube30 --device cuda:0
python cases/reproduce.py --root results/experiment --device cuda:0 --no-render
```

Completed cases are reused only after their summaries and execution records are checked.
Use `--rerun` to replace selected results. `--render-only` loads completed outputs and
builds figures without advancing the solver.

The physical case scripts provide additional CLI parameters:

```bash
python cases/radial_annulus.py --help
python cases/superellipse_shell.py --help
python cases/eccentric_pulse.py --help
python cases/spherical_shell.py --help
python cases/tube_pull_torsion.py --help
```

For altered loading, material, or geometry, use a separate `--results-dir`/`--case-name`
and a matching reference solution. The packaged FEM files have fixed physical parameters
recorded in [reference_data/manifest.json](../reference_data/manifest.json).

## Output and units

`results/paper/` contains per-case `data/` directories with compressed NumPy fields,
JSON summaries, and CSV profiles, followed by PDF and PNG figures. The numerical files
use nondimensional variables. Figures use `L0=0.1 m` and stress scale `S0=10 MPa`, so
one displacement unit is 10 cm and one stress unit is 10 MPa. The density scale
`rho0=1000 kg/m³` gives time scale `T0=1 ms`.

Displacement and deformation-gradient errors use relative Euclidean norms on the stated
comparison nodes. Tube profile discrepancies are absolute RMS differences. Saved fields
and [validation/manuscript_metrics.json](../validation/manuscript_metrics.json) provide
the quantities needed to assess these measures.

## Computational requirements

The reported simulations use double precision on an NVIDIA GeForce RTX 5090 with 32 GB
of memory. CPU execution is supported for small examples and automated tests. The
full three-dimensional grids require substantially more memory and runtime than the
quick-start example. Kernel compilation and geometry setup add first-run overhead.
Execution records report actual elapsed times for the selected machine.
