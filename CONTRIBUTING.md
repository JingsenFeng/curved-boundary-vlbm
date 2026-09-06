# Contributing

For a bug report, include the command, Python/dependency versions, operating system,
Warp device, and a small reproducible case. Provide the relevant log and summary when
the issue concerns a numerical result.

Before opening a pull request:

```bash
python -m pip install -e '.[test,render]'
ruff check .
WARP_DEVICE=cpu python -m pytest -q
python cases/smoke.py --device cpu
```

Changes to boundary reconstruction or constitutive operations should include a
numerical test against an independent formula, exact solution, or recorded regression
fixture. Keep the manuscript settings in `cases/traction_settings.py` explicit.
Record changes to numerical conventions and benchmark behavior in the documentation.

Use `results/` for generated output. Commit source, documentation, tests, and small
reference fixtures; distribute large simulation fields through a data archive.
