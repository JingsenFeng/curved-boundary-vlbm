"""Portable case paths and validated, bundled finite-element references."""

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "paper"
REFERENCES = ROOT / "reference_data"
FENICSX_PYTHON = Path(os.environ.get("FENICSX_PYTHON", sys.executable)).expanduser()


def install_reference(case, args, destination, *, validate_only=False):
    """Use the bundled reference only for its recorded physical configuration.

    Customized static or dynamic FEM cases can generate their own reference
    through the case driver's --force-fem and --fenicsx-python options.
    """
    entry = json.loads((REFERENCES / "manifest.json").read_text())[case]
    differences = []
    for key, expected in entry["parameters"].items():
        actual = getattr(args, key, None)
        matches = actual == expected
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            matches = math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14)
        if not matches:
            differences.append(f"{key}={actual!r} (reference: {expected!r})")
    if differences:
        if validate_only:
            raise ValueError(
                "Bundled FEM reference does not match this case: "
                + "; ".join(differences)
                + ". Supply --fem-reference or regenerate with --force-fem."
            )
        return False
    source = REFERENCES / entry["file"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"Reference checksum mismatch: {source}")
    if not validate_only and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return True
