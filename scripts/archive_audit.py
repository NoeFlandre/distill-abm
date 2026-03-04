"""Builds a full archive manifest with required migration classifications."""

from __future__ import annotations

import hashlib
import json
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
    "legacy_visualization",
]
Action = Literal["migrate", "retain_record_only", "archive_separately", "discard_with_rationale"]

ROOT = Path("archive")
NOTEBOOK_ROOT = Path("archive/legacy_repo/Code")
JSON_OUT = Path("docs/archive_full_manifest.json")
MD_OUT = Path("docs/archive_full_manifest.md")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill_abm.legacy.notebook_loader import required_notebook_function_sources  # noqa: E402

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
    rel_code = rel.split("archive/legacy_repo/Code/", 1)[1] if "archive/legacy_repo/Code/" in rel else rel
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
    if "/Evaluation/Human Assessment/" in rel:
        target = "tests/fixtures/notebook_parity/human_reference/" + rel.split("/Evaluation/Human Assessment/", 1)[1]
        return ("human_ground_truth", "migrate", target, "Human-scored references used for evaluation parity.")
    if "/Evaluation/Qualitative Assessment using LLMs/Examples/Text/" in rel:
        target = "configs/notebook_prompt_assets/" + rel.split("archive/legacy_repo/Code/", 1)[1]
        return (
            "prompt_reference",
            "migrate",
            target,
            "Notebook qualitative prompt exemplars and input reference text.",
        )
    if "/Evaluation/Qualitative Assessment using LLMs/Examples/Images/" in rel:
        target = (
            "tests/fixtures/notebook_parity/qual_examples/"
            + rel.split("/Evaluation/Qualitative Assessment using LLMs/Examples/", 1)[1]
        )
        return (
            "prompt_reference",
            "migrate",
            target,
            "Example image assets referenced by qualitative notebook prompts.",
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
            "legacy_visualization",
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


def build_manifest() -> list[ManifestRow]:
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and path.name != ".DS_Store")
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
