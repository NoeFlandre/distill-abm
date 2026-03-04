from __future__ import annotations

import hashlib
from pathlib import Path

LEGACY_MODELS_ROOT = Path("archive/legacy_repo/Code/Models")
MIRRORED_MODELS_ROOT = Path("tests/fixtures/notebook_parity/model_assets/Models")
MIGRATED_SUBDIRS = {"CSV", "JSON", "TXT", "NetLogo", "Images"}


def _iter_legacy_model_asset_files() -> list[Path]:
    files: list[Path] = []
    for path in LEGACY_MODELS_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.name == ".DS_Store":
            continue
        if ".ipynb_checkpoints" in path.parts:
            continue
        relative = path.relative_to(LEGACY_MODELS_ROOT)
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


def test_mirrored_model_assets_cover_all_legacy_model_files() -> None:
    legacy_files = _iter_legacy_model_asset_files()
    mirrored_files = sorted(path for path in MIRRORED_MODELS_ROOT.rglob("*") if path.is_file())

    expected_mirrored = sorted(path.relative_to(LEGACY_MODELS_ROOT) for path in legacy_files)
    observed_mirrored = sorted(path.relative_to(MIRRORED_MODELS_ROOT) for path in mirrored_files)
    assert observed_mirrored == expected_mirrored


def test_mirrored_model_assets_are_byte_equivalent() -> None:
    for legacy_path in _iter_legacy_model_asset_files():
        relative = legacy_path.relative_to(LEGACY_MODELS_ROOT)
        mirrored_path = MIRRORED_MODELS_ROOT / relative
        assert mirrored_path.exists()
        assert _sha256(legacy_path) == _sha256(mirrored_path)
