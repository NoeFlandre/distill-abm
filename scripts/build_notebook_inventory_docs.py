"""Renders human-readable markdown docs from notebook audit JSON artifacts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

AUDIT_JSON = Path("docs/notebook_function_audit.json")
COVERAGE_JSON = Path("docs/notebook_coverage_matrix.json")
INVENTORY_MD = Path("docs/notebook_function_inventory.md")
COVERAGE_MD = Path("docs/notebook_coverage_matrix.md")


def build_inventory_md() -> None:
    rows = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    by_path: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_path[row["path"]].append(row["qualname"])
    lines = [
        "# Notebook Function Inventory",
        "",
        f"- Total definitions: {len(rows)}",
        f"- Unique notebook files with definitions: {len(by_path)}",
        "",
        "## Functions by Notebook",
        "",
    ]
    for path in sorted(by_path):
        lines.append(f"### `{path}`")
        for name in sorted(set(by_path[path])):
            lines.append(f"- `{name}`")
        lines.append("")
    INVENTORY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_coverage_md() -> None:
    rows = json.loads(COVERAGE_JSON.read_text(encoding="utf-8"))
    class_counts = Counter("covered" if row["in_compat"] else "missing" for row in rows)
    lines = [
        "# Notebook Coverage Matrix",
        "",
        f"- Unique notebook function names: {len(rows)}",
        f"- Covered in `compat.compat`: {class_counts['covered']}",
        f"- Missing in `compat.compat`: {class_counts['missing']}",
        "",
        "| function | in_compat | in_src_name | src_modules |",
        "| --- | --- | --- | --- |",
    ]
    for row in sorted(rows, key=lambda item: item["name"]):
        modules = ", ".join(row["src_modules"]) if row["src_modules"] else "-"
        lines.append(f"| `{row['name']}` | `{row['in_compat']}` | `{row['in_src']}` | `{modules}` |")
    COVERAGE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    build_inventory_md()
    build_coverage_md()


if __name__ == "__main__":
    main()
