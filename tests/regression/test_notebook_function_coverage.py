from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, cast

import nbformat

import distill_abm.legacy.compat as compat


def test_all_notebook_functions_are_accounted_for() -> None:
    notebook_root = Path("archive/legacy_repo/Code")
    notebook_functions = _extract_function_names(notebook_root)
    implemented = set(compat.__all__)
    allowed_unmigrated = {"main"}
    missing = sorted(notebook_functions - implemented - allowed_unmigrated)
    assert missing == []


def _extract_function_names(root: Path) -> set[str]:
    output: set[str] = set()
    for notebook in root.rglob("*.ipynb"):
        parsed = cast(Any, nbformat).read(notebook, as_version=4)
        for cell in parsed.cells:
            if cell.cell_type != "code":
                continue
            try:
                tree = ast.parse(cell.source)
            except SyntaxError:
                continue
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    output.add(node.name)
    return output
