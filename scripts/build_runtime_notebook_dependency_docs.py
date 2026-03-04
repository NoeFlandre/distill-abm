"""Builds runtime-required notebook dependency reports from the loader registry."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distill_abm.legacy.notebook_loader import required_notebook_dependencies_by_path  # noqa: E402

JSON_OUT = ROOT / "docs/runtime_notebook_dependencies.json"
MD_OUT = ROOT / "docs/runtime_notebook_dependencies.md"


def build_rows() -> list[dict[str, object]]:
    grouped = required_notebook_dependencies_by_path()
    rows: list[dict[str, object]] = []
    for notebook_path, functions in grouped.items():
        rows.append(
            {
                "notebook_path": notebook_path.as_posix(),
                "required_functions": list(functions),
                "function_count": len(functions),
            }
        )
    return rows


def write_json(rows: list[dict[str, object]]) -> None:
    JSON_OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def write_markdown(rows: list[dict[str, object]]) -> None:
    lines = [
        "# Runtime Notebook Dependencies",
        "",
        f"- Required notebooks: {len(rows)}",
        f"- Required functions: {sum(int(row['function_count']) for row in rows)}",
        "",
        "| notebook_path | function_count | required_functions |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        functions = ", ".join(f"`{name}`" for name in row["required_functions"])
        lines.append(f"| `{row['notebook_path']}` | `{row['function_count']}` | {functions} |")
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_json(rows)
    write_markdown(rows)


if __name__ == "__main__":
    main()
