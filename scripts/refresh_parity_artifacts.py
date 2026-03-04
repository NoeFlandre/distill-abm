"""Refreshes all notebook parity and archive audit artifact files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMANDS = (
    [sys.executable, "scripts/archive_audit.py"],
    [sys.executable, "scripts/build_runtime_notebook_dependency_docs.py"],
    [sys.executable, "scripts/build_notebook_inventory_docs.py"],
)


def main() -> None:
    for command in COMMANDS:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
