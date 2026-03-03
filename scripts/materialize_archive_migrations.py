"""Copies archive files marked for migration into their mapped target paths."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

MANIFEST_PATH = Path("docs/archive_full_manifest.json")


def main() -> None:
    rows = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    copied = 0
    for row in rows:
        if row["action"] != "migrate":
            continue
        source = Path(row["path"])
        target_raw = row["target_path"]
        if target_raw is None:
            continue
        target = Path(target_raw)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    print(f"copied {copied} files")


if __name__ == "__main__":
    main()
