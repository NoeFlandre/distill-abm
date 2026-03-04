"""Builds a full archive manifest with required migration classifications."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Classification = Literal[
    "runtime_required",
    "prompt_reference",
    "human_ground_truth",
    "experiment_setting",
    "historical_nonruntime",
    "reference_visualization",
]
Action = Literal["migrate", "retain_record_only"]

ROOT = Path("archive")
NOTEBOOK_ROOT = Path("archive/reference_repo/Code")
JSON_OUT = Path("docs/archive_full_manifest.json")
MD_OUT = Path("docs/archive_full_manifest.md")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill_abm.reference.notebook_loader import required_notebook_function_sources  # noqa: E402

RUNTIME_REQUIRED_NOTEBOOKS = {path.as_posix() for path in required_notebook_function_sources().values()}


@dataclass(frozen=True)
class ManifestRow:
    path: str
    sha256: str
    size_bytes: int
    extension: str
    classification: Classification
    action: Action
    target_path: str | None
    rationale: str
    source_notebook_links: list[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classify(path: Path) -> tuple[Classification, Action, str | None, str]:
    rel = path.as_posix()
    ext = path.suffix.lower()
    rel_code = rel.split("archive/reference_repo/Code/", 1)[1] if "archive/reference_repo/Code/" in rel else rel
    if rel in RUNTIME_REQUIRED_NOTEBOOKS:
        return (
            "runtime_required",
            "retain_record_only",
            rel,
            "Notebook file currently required by runtime loader for compatibility dispatch; retained in place.",
        )
    if path.name == ".DS_Store" or path.name.startswith("~$"):
        return (
            "historical_nonruntime",
            "retain_record_only",
            None,
            "Temporary/editor artifact retained only for historical traceability.",
        )
    if "/Evaluation/" in rel:
        eval_rel = rel.split("/Evaluation/", 1)[1]
        target = "tests/fixtures/notebook_parity/evaluation_assets/Evaluation/" + eval_rel
        classification: Classification = "historical_nonruntime"
        if "/Evaluation/Human Assessment/" in rel:
            classification = "human_ground_truth"
        elif "/Evaluation/Qualitative Assessment using LLMs/Examples/" in rel:
            classification = "prompt_reference"
        elif path.name in {"reducedmilk.csv", "reduced3.csv", "FinalResultsYesNo.csv", "Yes-No Format.csv"}:
            classification = "experiment_setting"
        return (
            classification,
            "migrate",
            target,
            "Canonical mirrored evaluation artifact for notebook-parity provenance and safe archive cleanup.",
        )
    if path.name in {"reducedmilk.csv", "reduced3.csv", "FinalResultsYesNo.csv", "Yes-No Format.csv"}:
        target = "tests/fixtures/notebook_parity/experiment_settings/" + rel_code
        return ("experiment_setting", "migrate", target, "Canonical notebook experiment input/settings datasets.")
    if "/Models/" in rel and ext in {".txt", ".json"}:
        target = "tests/fixtures/notebook_parity/experiment_settings/" + rel_code
        return ("experiment_setting", "migrate", target, "Model parameter/documentation inputs used in notebook runs.")
    if ext == ".ipynb":
        return (
            "historical_nonruntime",
            "retain_record_only",
            None,
            "Notebook retained for full provenance, even when non-runtime.",
        )
    if ext in {".png", ".jpg", ".jpeg", ".svg"}:
        return (
            "reference_visualization",
            "retain_record_only",
            None,
            "Visualization artifact retained for historical comparison and reproducibility.",
        )
    if ext in {".csv", ".xlsx", ".numbers", ".log", ".aux", ".out", ".spl", ".pptx", ".pdf", ".tex", ".gz"}:
        return (
            "historical_nonruntime",
            "retain_record_only",
            None,
            "Historical output/result artifact not required for production runtime.",
        )
    return (
        "historical_nonruntime",
        "retain_record_only",
        None,
        "Unreferenced archival artifact retained for historical traceability.",
    )


def _source_links(path: Path) -> list[str]:
    if path.suffix.lower() == ".ipynb" and path.is_relative_to(NOTEBOOK_ROOT):
        return [path.relative_to(NOTEBOOK_ROOT).as_posix()]
    return []


def _tracked_archive_files() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--", str(ROOT)],
            check=True,
            capture_output=True,
            text=False,
        )
    except Exception:
        return sorted(path for path in ROOT.rglob("*") if path.is_file() and path.name != ".DS_Store")
    raw_items = [item for item in result.stdout.split(b"\x00") if item]
    files = [Path(item.decode("utf-8", errors="surrogateescape")) for item in raw_items]
    return sorted(path for path in files if path.is_file() and path.name != ".DS_Store")


def build_manifest() -> list[ManifestRow]:
    files = _tracked_archive_files()
    rows: list[ManifestRow] = []
    for path in files:
        classification, action, target_path, rationale = _classify(path)
        rows.append(
            ManifestRow(
                path=path.as_posix(),
                sha256=_sha256(path),
                size_bytes=path.stat().st_size,
                extension=path.suffix.lower(),
                classification=classification,
                action=action,
                target_path=target_path,
                rationale=rationale,
                source_notebook_links=_source_links(path),
            )
        )
    return rows


def write_outputs(rows: list[ManifestRow]) -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")

    class_counts = Counter(row.classification for row in rows)
    action_counts = Counter(row.action for row in rows)
    unresolved = [
        row
        for row in rows
        if row.classification in {"runtime_required", "prompt_reference", "human_ground_truth", "experiment_setting"}
        and row.target_path is None
    ]
    lines = [
        "# Archive Full Manifest",
        "",
        f"- Total files: {len(rows)}",
        "- Classification counts:",
    ]
    for key in sorted(class_counts):
        lines.append(f"  - `{key}`: {class_counts[key]}")
    lines.append("- Action counts:")
    for key in sorted(action_counts):
        lines.append(f"  - `{key}`: {action_counts[key]}")
    lines.append(f"- Unresolved mappings: {len(unresolved)}")
    lines.append("")
    lines.append("## Sample Rows")
    lines.append("")
    lines.append("| path | classification | action | target_path |")
    lines.append("| --- | --- | --- | --- |")
    for row in rows[:50]:
        lines.append(f"| `{row.path}` | `{row.classification}` | `{row.action}` | `{row.target_path or '-'}` |")
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_manifest()
    write_outputs(rows)


if __name__ == "__main__":
    main()
