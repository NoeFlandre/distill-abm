from __future__ import annotations

import hashlib
from pathlib import Path

REFERENCE_MODELS_ROOT = Path("archive/reference_repo/Code/Models")
MIRRORED_MODELS_ROOT = Path("tests/fixtures/notebook_parity/model_assets/Models")
MIGRATED_SUBDIRS = {"CSV", "JSON", "TXT", "NetLogo", "Images"}


def _iter_reference_model_asset_files() -> list[Path]:
    if REFERENCE_MODELS_ROOT.exists():
        return sorted(_iter_from_root(REFERENCE_MODELS_ROOT))

    return sorted(_iter_from_root(MIRRORED_MODELS_ROOT))


def _iter_from_root(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name == ".DS_Store":
            continue
        if ".ipynb_checkpoints" in path.parts:
            continue
        relative = path.relative_to(root)
        if len(relative.parts) < 2:
            continue
        if relative.parts[1] not in MIGRATED_SUBDIRS:
            continue
        files.append(path)
    return sorted(files)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_mirrored_model_assets_cover_all_reference_model_files() -> None:
    reference_files = _iter_reference_model_asset_files()
    mirrored_files = sorted(path for path in MIRRORED_MODELS_ROOT.rglob("*") if path.is_file())

    reference_root = REFERENCE_MODELS_ROOT if REFERENCE_MODELS_ROOT.exists() else MIRRORED_MODELS_ROOT
    expected_mirrored = sorted(path.relative_to(reference_root) for path in reference_files)
    observed_mirrored = sorted(path.relative_to(MIRRORED_MODELS_ROOT) for path in mirrored_files)
    assert observed_mirrored == expected_mirrored


def test_mirrored_model_assets_are_byte_equivalent() -> None:
    for reference_path in _iter_reference_model_asset_files():
        reference_root = REFERENCE_MODELS_ROOT if REFERENCE_MODELS_ROOT.exists() else MIRRORED_MODELS_ROOT
        relative = reference_path.relative_to(reference_root)
        mirrored_path = MIRRORED_MODELS_ROOT / relative
        assert mirrored_path.exists()
        assert _sha256(reference_path) == _sha256(mirrored_path)
