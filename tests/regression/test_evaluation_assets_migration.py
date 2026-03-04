from __future__ import annotations

import hashlib
import subprocess
import unicodedata
from pathlib import Path

REFERENCE_EVAL_ROOT = Path("archive/reference_repo/Code/Evaluation")
MIRRORED_EVAL_ROOT = Path("tests/fixtures/notebook_parity/evaluation_assets/Evaluation")


def _tracked_evaluation_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", str(REFERENCE_EVAL_ROOT)],
        check=True,
        capture_output=True,
        text=False,
    )
    raw_items = [item for item in result.stdout.split(b"\x00") if item]
    files = [Path(item.decode("utf-8", errors="surrogateescape")) for item in raw_items]
    return sorted(path for path in files if path.is_file())


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalized_relative(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return "/".join(unicodedata.normalize("NFC", part) for part in rel.parts)


def test_mirrored_evaluation_assets_cover_all_tracked_files() -> None:
    tracked_files = _tracked_evaluation_files()
    mirrored_files = sorted(
        path
        for path in MIRRORED_EVAL_ROOT.rglob("*")
        if path.is_file() and path.name != ".DS_Store" and ".ipynb_checkpoints" not in path.parts
    )
    observed = {_normalized_relative(path, MIRRORED_EVAL_ROOT) for path in mirrored_files}
    expected = {_normalized_relative(path, REFERENCE_EVAL_ROOT) for path in tracked_files}
    assert expected.issubset(observed)


def test_mirrored_evaluation_assets_are_byte_equivalent() -> None:
    mirrored_index = {
        _normalized_relative(path, MIRRORED_EVAL_ROOT): path
        for path in MIRRORED_EVAL_ROOT.rglob("*")
        if path.is_file() and path.name != ".DS_Store" and ".ipynb_checkpoints" not in path.parts
    }
    for reference_path in _tracked_evaluation_files():
        key = _normalized_relative(reference_path, REFERENCE_EVAL_ROOT)
        mirrored_path = mirrored_index.get(key)
        assert mirrored_path is not None
        assert _sha256(reference_path) == _sha256(mirrored_path)
