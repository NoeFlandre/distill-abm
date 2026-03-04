"""Loads notebook-defined functions into a deterministic runtime registry."""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import FunctionType
from typing import Any, cast

import nbformat

NOTEBOOK_ROOT = Path("archive/legacy_repo/Code")
REQUIRED_NOTEBOOK_FUNCTIONS: tuple[str, ...] = (
    "analyze_factorial_contributions",
    "append_coverage_score",
    "append_faithfulness_score",
    "calculate_sst",
    "calculate_sums_and_sst",
    "clean_context_response",
    "clean_symbols",
    "compute_results",
    "create_factorial_design",
    "fill_faithfulness_scores",
    "read_and_parse_csv",
    "read_csv_to_df",
    "remove_evaluating_suffix",
    "return_csv",
    "return_csv_2",
    "update_structured_df",
)


@dataclass(frozen=True)
class NotebookFunction:
    """Stores one callable extracted from notebook code plus provenance."""

    name: str
    function: Callable[..., Any]
    notebook_path: Path


@lru_cache(maxsize=1)
def _build_registry() -> dict[str, NotebookFunction]:
    candidates = sorted(NOTEBOOK_ROOT.rglob("*.ipynb"), key=_path_priority)
    registry: dict[str, NotebookFunction] = {}
    for notebook_path in candidates:
        namespace = _load_notebook_namespace(notebook_path)
        for name, value in namespace.items():
            if not isinstance(value, FunctionType):
                continue
            if name.startswith("_"):
                continue
            current = registry.get(name)
            if current is None or _is_better_source(notebook_path, current.notebook_path):
                registry[name] = NotebookFunction(
                    name=name,
                    function=cast(Callable[..., Any], value),
                    notebook_path=notebook_path,
                )
    return registry


def available_function_names() -> list[str]:
    """Returns sorted function names extracted from archived notebooks."""
    return sorted(_build_registry().keys())


def missing_required_notebook_functions() -> list[str]:
    """Returns required notebook function names that are unavailable in the runtime registry."""
    available = set(available_function_names())
    return [name for name in REQUIRED_NOTEBOOK_FUNCTIONS if name not in available]


def required_notebook_function_sources() -> dict[str, Path]:
    """Returns notebook source paths for every required notebook function."""
    return {name: get_notebook_source_path(name) for name in REQUIRED_NOTEBOOK_FUNCTIONS}


def get_notebook_function(name: str) -> Callable[..., Any]:
    """Returns a callable that matches the selected notebook implementation."""
    try:
        return _build_registry()[name].function
    except KeyError as exc:
        raise KeyError(f"notebook function '{name}' not found") from exc


def get_notebook_source_path(name: str) -> Path:
    """Returns the notebook path that provided the selected function body."""
    try:
        return _build_registry()[name].notebook_path
    except KeyError as exc:
        raise KeyError(f"notebook function '{name}' not found") from exc


def _load_notebook_namespace(notebook_path: Path) -> dict[str, Any]:
    parsed = cast(Any, nbformat).read(notebook_path, as_version=4)
    namespace: dict[str, Any] = {}
    for cell in parsed.cells:
        if cell.cell_type != "code":
            continue
        source = str(cell.source)
        _execute_safe_nodes(source, namespace)
    return namespace


def _execute_safe_nodes(source: str, namespace: dict[str, Any]) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _safe_exec(node, namespace)
            continue
        if isinstance(node, ast.Assign) and _is_literal_like(node.value):
            _safe_exec(node, namespace)
            continue
        if isinstance(node, ast.AnnAssign) and node.value is not None and _is_literal_like(node.value):
            _safe_exec(node, namespace)
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _safe_exec(node, namespace)
            continue


def _safe_exec(node: ast.stmt, namespace: dict[str, Any]) -> None:
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    try:
        exec(compile(module, filename="<notebook_runtime>", mode="exec"), namespace)
    except Exception:
        return


def _is_literal_like(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_is_literal_like(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        keys_ok = all(key is None or _is_literal_like(key) for key in node.keys)
        vals_ok = all(_is_literal_like(value) for value in node.values)
        return keys_ok and vals_ok
    if isinstance(node, ast.UnaryOp):
        return _is_literal_like(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_literal_like(node.left) and _is_literal_like(node.right)
    return False


def _path_priority(path: Path) -> tuple[int, int, int, str]:
    text = str(path)
    lowered = text.lower()
    name_lower = path.name.lower()
    return (
        0 if "/archives/" in lowered else 1,
        0 if ".ipynb_checkpoints" in lowered else 1,
        0 if "copy" in name_lower else 1,
        lowered,
    )


def _is_better_source(candidate: Path, current: Path) -> bool:
    return _path_priority(candidate) > _path_priority(current)
