"""Reference reuse must match the physical problem and preserve the data."""

from argparse import Namespace
import hashlib
import json

import pytest

from _paths import REFERENCES, install_reference


@pytest.mark.parametrize("case", ["annulus", "superellipse", "pulse"])
def test_matching_reference_is_copied_without_changes(case, tmp_path):
    entry = json.loads((REFERENCES / "manifest.json").read_text())[case]
    destination = tmp_path / "reference.npz"
    assert install_reference(case, Namespace(**entry["parameters"]), destination)
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == entry["sha256"]


@pytest.mark.parametrize("case", ["annulus", "superellipse", "pulse"])
def test_changed_grid_does_not_reuse_reference(case, tmp_path):
    entry = json.loads((REFERENCES / "manifest.json").read_text())[case]
    parameters = {**entry["parameters"], "n": 17}
    destination = tmp_path / "reference.npz"
    assert not install_reference(case, Namespace(**parameters), destination)
    assert not destination.exists()
    with pytest.raises(ValueError, match="does not match"):
        install_reference(case, Namespace(**parameters), destination, validate_only=True)
