#!/usr/bin/env python3
"""Check the initial release's scientific definitions and local documentation links."""

import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def check():
    expected = json.loads((ROOT / "validation/scientific_definitions.json").read_text())
    fixes = json.loads((ROOT / "validation/initialization_buffer_fix.json").read_text())
    overrides = fixes["release_definition_hashes"]
    count = 0
    for filename, definitions in expected.items():
        tree = ast.parse((ROOT / filename).read_text())
        actual = {
            node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        for name, digest in definitions.items():
            digest = overrides.get(filename, {}).get(name, digest)
            if actual.get(name) != digest:
                raise ValueError(f"Scientific definition changed: {filename}:{name}")
            count += 1
    refs = json.loads((ROOT / "reference_data/manifest.json").read_text())
    for entry in refs.values():
        path = ROOT / "reference_data" / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Reference checksum mismatch: {path}")
    links = 0
    for path in [*ROOT.glob("*.md"), *(ROOT / "docs").glob("*.md")]:
        text = path.read_text()
        targets = re.findall(r"\]\(([^)]+)\)", text)
        targets += re.findall(r'src="([^"]+)"', text)
        for target in targets:
            if target.startswith(("http:", "https:", "#", "mailto:")):
                continue
            destination = (path.parent / target.split("#")[0]).resolve()
            if not destination.is_file():
                raise ValueError(f"Missing documentation target in {path.name}: {target}")
            links += 1
    print(
        json.dumps(
            {
                "definitions_checked": count,
                "definitions_unchanged": fixes["unmodified_top_level_definitions"],
                "initial_boundary_dispatch_fixes": len(fixes["changed_methods"]),
                "reference_files_verified": len(refs),
                "local_documentation_links_verified": links,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    check()
