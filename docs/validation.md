# Validation

The initial release was checked using Python 3.11, Warp 1.10.1, NumPy 2.3.5,
SciPy 1.16.3, and Matplotlib 3.10.7. CUDA checks used an NVIDIA GeForce RTX 5090;
the same numerical test suite also ran on the CPU backend.

## Automated numerical checks

The 27-test suite covers:

- Three-dimensional Dirichlet BFL populations after 25 steps against a stored
  regression fixture, including the absence of traction stencils in a pure Dirichlet case.
- Quadratic spatial derivatives, affine boundary prediction, cut-link selection,
  and the fixed spatial extent of the characteristic stencil.
- Nonlinear nominal-traction residuals and incoming populations against an independent
  flux/finite-difference calculation at general `omega`, including the source correction.
- Component-wise mixed conditions and preservation of the Dirichlet population history.
- Native two- and three-dimensional centered/one-sided displacement gradients on
  quadratic fields, and preservation of a zero-traction identity state.
- Common line sampling for FEM and LBM, treatment of holes, finite samples, and marker spacing.
- Reference-file checksums and rejection of mismatched reference-grid configurations.
- Agreement of in-place boundary initialization with a frozen-source evaluation in
  2D and both dense and sparse 3D storage, including closely spaced opposing cut links.

```bash
WARP_DEVICE=cpu python -m pytest -q
WARP_DEVICE=cuda:0 python -m pytest -q
python cases/smoke.py --device cpu
python cases/smoke.py --device cuda:0
```

Both device runs passed all 27 tests. The short examples advance a mixed-boundary
annulus and a traction-loaded sphere for 20 steps and check finite states and positive
deformation Jacobians. They provide an installation check; accuracy is assessed by the
exact-solution, FEM, and BVP benchmarks.

## Manuscript and release records

[manuscript_metrics.json](../validation/manuscript_metrics.json) records the benchmark
metrics and common numerical settings. [release_verification.json](../validation/release_verification.json)
records the release checks and comparison of the packaged calculations with those metrics.

The release completed all nine reproduction tasks: six affine-annulus grids, five
ellipsoid grids, and seven traction or mixed-boundary calculations, giving 18 individual
grid runs. All reached the prescribed final time within time-step rounding, with finite
saved fields and positive deformation Jacobians. Every stored error metric, including
the tube/FEM profile comparisons, agrees with the manuscript baseline to three significant
digits. The largest relative metric change is 0.0078% in the displacement error of the
160³ ellipsoid following the initialization fix described below.

The 202 top-level function/class definitions in the five solver modules were compared
with the manuscript source. Of these, 199 are unchanged. Three boundary-dispatch methods
now copy the initial populations into a scratch buffer before applying cut-link updates.
This removes a read/write dependency between opposing links during in-place initialization;
the constitutive, collision, streaming, and boundary-reconstruction kernels are unchanged.
The regression test fails on the aliased implementation and passes on CPU and CUDA
with the frozen-source update. Fingerprints and the specific change are recorded in
[initialization_buffer_fix.json](../validation/initialization_buffer_fix.json).

The import layout, documentation, output paths, and figure orchestration were organized
for an independent checkout. Source provenance is stored in `validation/`.

The figure pipeline is also checked after moving the datasets into a new directory
and replacing stored workstation paths with unavailable paths. This exercises the
bundled plot templates, local snapshot resolution, and FEM reference files.

The full benchmark commands record source hashes, elapsed times, completion status,
and the uniform traction settings. The tested configurations and their complete
commands are documented in [Benchmarks](benchmarks.md).
