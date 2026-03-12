"""Run-oriented quantitative audit smoke built from completed summarizer smoke outputs."""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from distill_abm.cli_support import resolve_quantitative_reference_paths
from distill_abm.eval.doe_full import analyze_factorial_anova
from distill_abm.eval.metrics import SummaryScores, score_summary
from distill_abm.pipeline.quantitative_rendering import (
    METRIC_COLUMN_NAMES,
    _format_contribution_cell,
    _format_pvalue_cell,
    _lookup_metric_value,
)
from distill_abm.pipeline.quantitative_rendering import (
    render_anova_latex_table as _render_anova_latex_table,
)
from distill_abm.pipeline.quantitative_rendering import (
    render_anova_markdown_table as _render_anova_markdown_table,
)
from distill_abm.pipeline.quantitative_rendering import (
    render_factorial_latex_table as _render_factorial_latex_table,
)
from distill_abm.pipeline.quantitative_rendering import (
    render_factorial_markdown_table as _render_factorial_markdown_table,
)
from distill_abm.pipeline.quantitative_rendering import (
    render_optimal_latex_table as _render_optimal_latex_table,
)
from distill_abm.pipeline.quantitative_rendering import (
    render_optimal_markdown_table as _render_optimal_markdown_table,
)
from distill_abm.pipeline.quantitative_rendering import (
    render_evidence_summary_markdown_table as _render_evidence_summary_markdown_table,
)
from distill_abm.pipeline.report_writers import write_model_report_files
from distill_abm.pipeline.run_artifact_contracts import (
    latest_run_pointer_path,
    resolve_run_root,
    run_log_path,
)
from distill_abm.structured_logging import attach_json_log_file, get_logger, log_event

# Explicit re-exports keep the historical test/import surface stable while
# delegating the rendering logic to the extracted helper module.
_render_anova_markdown_table = _render_anova_markdown_table
_render_anova_latex_table = _render_anova_latex_table
_render_factorial_markdown_table = _render_factorial_markdown_table
_render_factorial_latex_table = _render_factorial_latex_table
_render_optimal_markdown_table = _render_optimal_markdown_table
_render_optimal_latex_table = _render_optimal_latex_table
_render_evidence_summary_markdown_table = _render_evidence_summary_markdown_table

__all__ = [
    "QuantitativeRecord",
    "QuantitativeSmokeResult",
    "_build_evidence_summary_rows",
    "_render_anova_latex_table",
    "_render_anova_markdown_table",
    "_render_evidence_summary_markdown_table",
    "_render_factorial_latex_table",
    "_render_factorial_markdown_table",
    "_render_optimal_latex_table",
    "_render_optimal_markdown_table",
    "run_quantitative_smoke",
]

QUANTITATIVE_REPORT_FILENAME = "smoke_quantitative_report.json"
ANOVA_ROW_SPECS: tuple[tuple[str, str], ...] = (
    ("Agent-Based Model", "ABM"),
    ("Summarization algorithm", "Summarizer"),
    ("Simulation evidence", "Evidence"),
    ("LLM", "LLM"),
    ("Use of roles", "Role"),
    ("Generating insights", "Insights"),
    ("Providing examples", "Example"),
)
FACTORIAL_FEATURE_ORDER: tuple[str, ...] = (
    "Example",
    "Evidence",
    "Evidence_AND_Example",
    "Evidence_AND_Insights",
    "Evidence_AND_Role",
    "Insights",
    "Insights_AND_Example",
    "Role",
    "Role_AND_Example",
    "Role_AND_Insights",
    "Summarizer",
    "Summarizer_AND_Example",
    "Summarizer_AND_Evidence",
    "Summarizer_AND_Insights",
    "Summarizer_AND_Role",
)


class QuantitativeRecord(BaseModel):
    """Scored summary output for one case and summarizer mode."""

    record_id: str
    bundle_id: str
    case_id: str
    abm: str
    llm: str
    evidence: str
    prompt: str
    role: bool
    insights: bool
    example: bool
    summarizer: str
    repetition: int
    reference_family: str
    reference_path: Path
    summary_output_path: Path
    source_run_root: Path
    success: bool
    error: str | None = None
    bleu: float = 0.0
    meteor: float = 0.0
    rouge1: float = 0.0
    rouge2: float = 0.0
    rouge_l: float = 0.0
    flesch_reading_ease: float = 0.0
    token_f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0


class QuantitativeSmokeResult(BaseModel):
    """Top-level quantitative smoke result."""

    started_at_utc: str
    finished_at_utc: str
    output_root: Path
    run_id: str
    run_root: Path
    overview_root: Path
    report_json_path: Path
    report_markdown_path: Path
    review_csv_path: Path
    quantitative_rows_path: Path
    structured_results_path: Path
    anova_csv_path: Path
    factorial_csv_path: Path
    optimal_csv_path: Path
    anova_table_markdown_path: Path
    anova_table_latex_path: Path
    evidence_summary_table_markdown_path: Path
    factorial_table_markdown_path: Path
    factorial_table_latex_path: Path
    optimal_table_markdown_path: Path
    optimal_table_latex_path: Path
    run_log_path: Path
    success: bool
    failed_record_ids: list[str] = Field(default_factory=list)
    record_count: int = 0
    reference_roots: dict[str, Path] = Field(default_factory=dict)


def run_quantitative_smoke(
    *,
    source_root: Path,
    output_root: Path,
    resume: bool = False,
    score_summary_fn: Callable[[str, str], SummaryScores] = score_summary,
    analyze_factorial_anova_fn: Callable[[Path, Path, int], pd.DataFrame | None] = analyze_factorial_anova,
) -> QuantitativeSmokeResult:
    """Score completed summarizer smoke outputs and render publication-oriented analysis tables."""
    started_at = datetime.now(UTC)
    _prepare_output_root(output_root, resume=resume)
    run_id = started_at.strftime("run_%Y%m%d_%H%M%S_%f")
    run_root = output_root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    latest_run_pointer_path(output_root).write_text(str(run_root), encoding="utf-8")
    previous_run_root = _resolve_previous_quantitative_run_root(output_root=output_root, current_run_id=run_id)
    logger = get_logger(__name__)
    attached_run_log_path = attach_json_log_file(run_log_path(run_root))

    resolved_source_root = resolve_run_root(source_root)
    source_rows = _load_summarizer_review_rows(resolved_source_root)
    review_rows: list[dict[str, str]] = []
    record_rows: list[dict[str, str]] = []
    failed_record_ids: list[str] = []
    reference_review_rows: dict[str, list[dict[str, str]]] = {}
    reference_record_rows: dict[str, list[dict[str, str]]] = {}
    reference_failed_record_ids: dict[str, list[str]] = {}
    reference_roots: dict[str, Path] = {}

    for source_row in source_rows:
        for record in _build_quantitative_records_from_source_row(source_row):
            reference_root = run_root / record.reference_family
            reference_roots.setdefault(record.reference_family, reference_root)
            record_dir = reference_root / "records" / record.record_id
            previous_record_dir = (
                None
                if previous_run_root is None
                else previous_run_root / record.reference_family / "records" / record.record_id
            )
            reused = False
            if resume and previous_record_dir is not None:
                reused = _copy_previous_record_if_valid(previous_record_dir=previous_record_dir, record_dir=record_dir)
            if reused:
                loaded_record = _load_record_json(record_dir / "record.json")
                if loaded_record is None:
                    reused = False
                else:
                    record = loaded_record
                    log_event(
                        logger,
                        "quantitative_record_reused",
                        record_id=record.record_id,
                        case_id=record.case_id,
                        reference_family=record.reference_family,
                    )
            if not reused:
                try:
                    summary_text = _read_summary_text(record.summary_output_path)
                    reference_text = _load_reference_text(record.reference_path)
                    scores = score_summary_fn(reference_text, summary_text)
                    record = record.model_copy(
                        update={
                            "success": True,
                            "error": None,
                            "bleu": scores.bleu,
                            "meteor": scores.meteor,
                            "rouge1": scores.rouge1,
                            "rouge2": scores.rouge2,
                            "rouge_l": scores.rouge_l,
                            "flesch_reading_ease": scores.flesch_reading_ease,
                            "token_f1": scores.token_f1,
                            "precision": scores.precision,
                            "recall": scores.recall,
                        }
                    )
                    log_event(
                        logger,
                        "quantitative_record_success",
                        record_id=record.record_id,
                        case_id=record.case_id,
                        reference_family=record.reference_family,
                    )
                except Exception as exc:
                    record = record.model_copy(update={"success": False, "error": str(exc)})
                    failed_record_ids.append(record.record_id)
                    reference_failed_record_ids.setdefault(record.reference_family, []).append(record.record_id)
                    log_event(
                        logger,
                        "quantitative_record_failure",
                        level=40,
                        record_id=record.record_id,
                        case_id=record.case_id,
                        reference_family=record.reference_family,
                        error=str(exc),
                    )
                _write_record_bundle(record_dir=record_dir, record=record)

            review_row = _record_to_review_row(record)
            review_rows.append(review_row)
            reference_review_rows.setdefault(record.reference_family, []).append(review_row)
            if record.success:
                record_row = _record_to_csv_row(record)
                record_rows.append(record_row)
                reference_record_rows.setdefault(record.reference_family, []).append(record_row)
            elif record.record_id not in failed_record_ids:
                failed_record_ids.append(record.record_id)
                reference_failed_record_ids.setdefault(record.reference_family, []).append(record.record_id)

    combined_root = run_root / "combined"
    combined_root.mkdir(parents=True, exist_ok=True)
    combined_paths = _write_analysis_bundle(
        bundle_root=combined_root,
        review_rows=review_rows,
        record_rows=record_rows,
        analyze_factorial_anova_fn=analyze_factorial_anova_fn,
    )

    for reference_family, reference_root in reference_roots.items():
        reference_root.mkdir(parents=True, exist_ok=True)
        _write_analysis_bundle(
            bundle_root=reference_root,
            review_rows=reference_review_rows.get(reference_family, []),
            record_rows=reference_record_rows.get(reference_family, []),
            analyze_factorial_anova_fn=analyze_factorial_anova_fn,
        )
        _write_bundle_readme(
            bundle_root=reference_root,
            title=f"{reference_family} quantitative results",
            review_rows=reference_review_rows.get(reference_family, []),
            failed_record_ids=reference_failed_record_ids.get(reference_family, []),
        )

    overview_root = run_root / "overview"
    overview_root.mkdir(parents=True, exist_ok=True)
    overview_paths = _write_overview_tables(
        overview_root=overview_root,
        reference_record_rows=reference_record_rows,
        scratch_root=combined_root,
        analyze_factorial_anova_fn=analyze_factorial_anova_fn,
    )

    result = QuantitativeSmokeResult(
        started_at_utc=started_at.isoformat(),
        finished_at_utc=datetime.now(UTC).isoformat(),
        output_root=output_root,
        run_id=run_id,
        run_root=run_root,
        overview_root=overview_root,
        report_json_path=run_root / QUANTITATIVE_REPORT_FILENAME,
        report_markdown_path=run_root / "smoke_quantitative_report.md",
        review_csv_path=combined_paths["review_csv_path"],
        quantitative_rows_path=combined_paths["quantitative_rows_path"],
        structured_results_path=combined_paths["structured_results_path"],
        anova_csv_path=combined_paths["anova_csv_path"],
        factorial_csv_path=combined_paths["factorial_csv_path"],
        optimal_csv_path=combined_paths["optimal_csv_path"],
        anova_table_markdown_path=overview_paths["anova_table_markdown_path"],
        anova_table_latex_path=combined_paths["anova_table_latex_path"],
        evidence_summary_table_markdown_path=overview_paths["evidence_summary_table_markdown_path"],
        factorial_table_markdown_path=overview_paths["factorial_table_markdown_path"],
        factorial_table_latex_path=combined_paths["factorial_table_latex_path"],
        optimal_table_markdown_path=overview_paths["optimal_table_markdown_path"],
        optimal_table_latex_path=combined_paths["optimal_table_latex_path"],
        run_log_path=attached_run_log_path,
        success=not failed_record_ids,
        failed_record_ids=sorted(dict.fromkeys(failed_record_ids)),
        record_count=len(review_rows),
        reference_roots=dict(sorted(reference_roots.items())),
    )
    write_model_report_files(
        result=result,
        report_json_path=result.report_json_path,
        report_markdown_path=result.report_markdown_path,
        markdown=_render_markdown_report(result),
    )
    return result


def _load_summarizer_review_rows(source_root: Path) -> list[dict[str, str]]:
    review_csv_path = source_root / "review.csv"
    if not review_csv_path.exists():
        raise FileNotFoundError(f"missing summarizer smoke review.csv: {review_csv_path}")
    with review_csv_path.open(encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"summarizer smoke review.csv is empty: {review_csv_path}")
    return [row for row in rows if row.get("success", "").lower() == "true"]


def _build_quantitative_records_from_source_row(source_row: dict[str, str]) -> list[QuantitativeRecord]:
    case_metadata = _load_case_metadata_from_context_output(Path(source_row["context_output_path"]))
    llm = case_metadata.get("model_id", "") or case_metadata.get("model", "") or ""
    evidence_mode = str(case_metadata["evidence_mode"])
    prompt_variant = str(case_metadata["prompt_variant"])
    repetition = int(str(case_metadata["repetition"]))
    run_root = Path(str(case_metadata["run_root"]))
    prompt_flags = _derive_prompt_flags(prompt_variant)
    records: list[QuantitativeRecord] = []
    for reference_family, reference_path in resolve_quantitative_reference_paths(source_row["abm"]).items():
        records.append(
            QuantitativeRecord(
                record_id=f"{source_row['case_id']}__{source_row['mode']}__{reference_family.replace('.', '_')}",
                bundle_id=source_row["bundle_id"],
                case_id=source_row["case_id"],
                abm=source_row["abm"],
                llm=str(llm),
                evidence=evidence_mode,
                prompt=prompt_variant,
                role=prompt_flags["role"],
                insights=prompt_flags["insights"],
                example=prompt_flags["example"],
                summarizer=source_row["mode"],
                repetition=repetition,
                reference_family=reference_family,
                reference_path=reference_path,
                summary_output_path=Path(source_row["summary_output_path"]),
                source_run_root=run_root,
                success=False,
            )
        )
    return records


def _load_case_metadata_from_context_output(context_output_path: Path) -> dict[str, object]:
    case_dir = context_output_path.parent.parent
    run_root = case_dir.parent.parent
    case_summary_path = case_dir / "00_case_summary.json"
    if case_summary_path.exists():
        payload = json.loads(case_summary_path.read_text(encoding="utf-8"))
        return {
            "evidence_mode": str(payload["evidence_mode"]),
            "prompt_variant": str(payload["prompt_variant"]),
            "repetition": int(payload.get("repetition", 1)),
            "model_id": str(payload.get("model_id", "")),
            "model": str(payload.get("model", "")),
            "run_root": run_root,
        }
    report_path = run_root / "smoke_full_case_matrix_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    for case in payload.get("cases", []):
        if str(case.get("case_dir")) == str(case_dir):
            return {
                "evidence_mode": str(case["evidence_mode"]),
                "prompt_variant": str(case["prompt_variant"]),
                "repetition": int(case["repetition"]),
                "model_id": str(case.get("model_id", "")),
                "model": str(payload.get("model_id", "") or payload.get("model", "")),
                "run_root": run_root,
            }
    raise ValueError(f"could not resolve case metadata for {context_output_path}")


def _read_summary_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"missing summary output: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"summary output is empty: {path}")
    return text


def _load_reference_text(reference_path: Path) -> str:
    return reference_path.read_text(encoding="utf-8").strip()


def _derive_prompt_flags(prompt_variant: str) -> dict[str, bool]:
    if prompt_variant == "none":
        return {"role": False, "insights": False, "example": False}
    normalized = prompt_variant.replace("all_three", "role+insights+example")
    active = set(normalized.split("+"))
    return {
        "role": "role" in active,
        "insights": "insights" in active,
        "example": "example" in active,
    }


def _copy_previous_record_if_valid(*, previous_record_dir: Path, record_dir: Path) -> bool:
    if not previous_record_dir.exists() or record_dir.exists():
        return False
    record_json_path = previous_record_dir / "record.json"
    loaded = _load_record_json(record_json_path)
    if loaded is None or not loaded.success:
        return False
    shutil.copytree(previous_record_dir, record_dir)
    return True


def _load_record_json(path: Path) -> QuantitativeRecord | None:
    if not path.exists():
        return None
    try:
        return QuantitativeRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_record_bundle(*, record_dir: Path, record: QuantitativeRecord) -> None:
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / "record.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")


def _record_to_review_row(record: QuantitativeRecord) -> dict[str, str]:
    return {
        "record_id": record.record_id,
        "bundle_id": record.bundle_id,
        "case_id": record.case_id,
        "abm": record.abm,
        "llm": record.llm,
        "evidence": record.evidence,
        "prompt": record.prompt,
        "summarizer": record.summarizer,
        "repetition": str(record.repetition),
        "summary_output_path": str(record.summary_output_path),
        "success": str(record.success),
        "reference_family": record.reference_family,
        "error": record.error or "",
    }


def _record_to_csv_row(record: QuantitativeRecord) -> dict[str, str]:
    return {
        "record_id": record.record_id,
        "bundle_id": record.bundle_id,
        "case_id": record.case_id,
        "abm": record.abm,
        "llm": record.llm,
        "evidence": record.evidence,
        "prompt": record.prompt,
        "role": str(record.role),
        "insights": str(record.insights),
        "example": str(record.example),
        "summarizer": record.summarizer,
        "repetition": str(record.repetition),
        "reference_family": record.reference_family,
        "summary_output_path": str(record.summary_output_path),
        "BLEU": f"{record.bleu:.6f}",
        "METEOR": f"{record.meteor:.6f}",
        "R-1": f"{record.rouge1:.6f}",
        "R-2": f"{record.rouge2:.6f}",
        "R-L": f"{record.rouge_l:.6f}",
        "Reading ease": f"{record.flesch_reading_ease:.6f}",
    }


def _quantitative_row_fields() -> list[str]:
    return [
        "record_id",
        "bundle_id",
        "case_id",
        "abm",
        "llm",
        "evidence",
        "prompt",
        "role",
        "insights",
        "example",
        "summarizer",
        "repetition",
        "reference_family",
        "summary_output_path",
        *METRIC_COLUMN_NAMES,
    ]


def _review_row_fields() -> list[str]:
    return [
        "record_id",
        "bundle_id",
        "case_id",
        "abm",
        "llm",
        "evidence",
        "prompt",
        "summarizer",
        "repetition",
        "summary_output_path",
        "success",
        "reference_family",
        "error",
    ]


def _structured_results_fields() -> list[str]:
    return [
        "Case study",
        "Reference family",
        "Summary",
        "LLM",
        "Role",
        "Example",
        "Insight",
        "Evidence",
        "Repetition",
        "Output",
        "BLEU",
        "METEOR",
        "ROUGE-1",
        "ROUGE-2",
        "ROUGE-L",
        "Flesch Reading Ease",
    ]


def _build_structured_results_rows(record_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Build the normalized factor sheet equivalent to the legacy final-sheet workflow."""
    structured_rows: list[dict[str, str]] = []
    for row in record_rows:
        structured_rows.append(
            {
                "Case study": row["abm"],
                "Reference family": row["reference_family"],
                "Summary": row["summarizer"],
                "LLM": row["llm"],
                "Role": "Yes" if row["role"] == "True" else "No",
                "Example": "Yes" if row["example"] == "True" else "No",
                "Insight": "Yes" if row["insights"] == "True" else "No",
                "Evidence": row["evidence"],
                "Repetition": row["repetition"],
                "Output": row["summary_output_path"],
                "BLEU": row["BLEU"],
                "METEOR": row["METEOR"],
                "ROUGE-1": row["R-1"],
                "ROUGE-2": row["R-2"],
                "ROUGE-L": row["R-L"],
                "Flesch Reading Ease": row["Reading ease"],
            }
        )
    return structured_rows


def _write_csv(path: Path, *, fieldnames: Iterable[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _write_analysis_bundle(
    *,
    bundle_root: Path,
    review_rows: list[dict[str, str]],
    record_rows: list[dict[str, str]],
    analyze_factorial_anova_fn: Callable[[Path, Path, int], pd.DataFrame | None],
) -> dict[str, Path]:
    bundle_root.mkdir(parents=True, exist_ok=True)
    quantitative_rows_path = bundle_root / "quantitative_rows.csv"
    structured_results_path = bundle_root / "structured_results.csv"
    review_csv_path = bundle_root / "review.csv"
    anova_csv_path = bundle_root / "anova_pvalues.csv"
    factorial_csv_path = bundle_root / "factorial_contributions.csv"
    optimal_csv_path = bundle_root / "best_scores.csv"
    anova_table_markdown_path = bundle_root / "anova_table.md"
    anova_table_latex_path = bundle_root / "anova_table.tex"
    factorial_table_markdown_path = bundle_root / "factorial_table.md"
    factorial_table_latex_path = bundle_root / "factorial_table.tex"
    optimal_table_markdown_path = bundle_root / "best_scores_table.md"
    optimal_table_latex_path = bundle_root / "best_scores_table.tex"

    _write_csv(
        quantitative_rows_path,
        fieldnames=list(record_rows[0].keys()) if record_rows else _quantitative_row_fields(),
        rows=record_rows,
    )
    structured_rows = _build_structured_results_rows(record_rows)
    _write_csv(
        structured_results_path,
        fieldnames=list(structured_rows[0].keys()) if structured_rows else _structured_results_fields(),
        rows=structured_rows,
    )
    _write_csv(
        review_csv_path,
        fieldnames=list(review_rows[0].keys()) if review_rows else _review_row_fields(),
        rows=review_rows,
    )

    analysis_frame = pd.DataFrame(record_rows)
    anova_rows = _compute_anova_rows(analysis_frame)
    _write_anova_csv(anova_csv_path, anova_rows)
    anova_table_markdown_path.write_text(_render_anova_markdown_table(anova_rows), encoding="utf-8")
    anova_table_latex_path.write_text(_render_anova_latex_table(anova_rows), encoding="utf-8")

    factorial_input_path = bundle_root / "factorial_input.csv"
    factorial_frame = _build_factorial_input_frame(record_rows)
    factorial_frame.to_csv(factorial_input_path, index=False)
    factorial_result = analyze_factorial_anova_fn(factorial_input_path, factorial_csv_path, 2)
    normalized_factorial = _normalize_factorial_table(factorial_result)
    normalized_factorial.to_csv(factorial_csv_path, index=False, float_format="%.8f")
    factorial_table_markdown_path.write_text(_render_factorial_markdown_table(normalized_factorial), encoding="utf-8")
    factorial_table_latex_path.write_text(_render_factorial_latex_table(normalized_factorial), encoding="utf-8")

    optimal_rows = _build_optimal_score_rows(record_rows)
    _write_optimal_csv(optimal_csv_path, optimal_rows)
    optimal_table_markdown_path.write_text(_render_optimal_markdown_table(optimal_rows), encoding="utf-8")
    optimal_table_latex_path.write_text(_render_optimal_latex_table(optimal_rows), encoding="utf-8")

    return {
        "review_csv_path": review_csv_path,
        "quantitative_rows_path": quantitative_rows_path,
        "structured_results_path": structured_results_path,
        "anova_csv_path": anova_csv_path,
        "factorial_csv_path": factorial_csv_path,
        "optimal_csv_path": optimal_csv_path,
        "anova_table_markdown_path": anova_table_markdown_path,
        "anova_table_latex_path": anova_table_latex_path,
        "factorial_table_markdown_path": factorial_table_markdown_path,
        "factorial_table_latex_path": factorial_table_latex_path,
        "optimal_table_markdown_path": optimal_table_markdown_path,
        "optimal_table_latex_path": optimal_table_latex_path,
    }


def _write_bundle_readme(
    *,
    bundle_root: Path,
    title: str,
    review_rows: list[dict[str, str]],
    failed_record_ids: list[str],
) -> None:
    (bundle_root / "README.md").write_text(
        (
            f"# {title}\n\n"
            f"- reviewed_records: `{len(review_rows)}`\n"
            f"- failed_records: `{len(set(failed_record_ids))}`\n"
            "- artifacts:\n"
            f"  - `review.csv`\n"
            f"  - `quantitative_rows.csv`\n"
            f"  - `structured_results.csv`\n"
            f"  - `best_scores.csv`\n"
            f"  - `anova_pvalues.csv`\n"
            f"  - `factorial_contributions.csv`\n"
        ),
        encoding="utf-8",
    )


def _write_overview_tables(
    *,
    overview_root: Path,
    reference_record_rows: dict[str, list[dict[str, str]]],
    scratch_root: Path,
    analyze_factorial_anova_fn: Callable[[Path, Path, int], pd.DataFrame | None],
) -> dict[str, Path]:
    overview_root.mkdir(parents=True, exist_ok=True)
    anova_table_markdown_path = overview_root / "anova_table.md"
    evidence_summary_table_markdown_path = overview_root / "evidence_summary_table.md"
    factorial_table_markdown_path = overview_root / "factorial_table.md"
    optimal_table_markdown_path = overview_root / "best_scores_table.md"

    anova_rows: list[dict[str, float | str | None]] = []
    evidence_summary_rows: list[dict[str, str]] = []
    factorial_frames: list[pd.DataFrame] = []
    optimal_rows: list[dict[str, str]] = []
    for reference_family, record_rows in sorted(reference_record_rows.items()):
        frame = pd.DataFrame(record_rows)
        for row in _compute_anova_rows(frame):
            anova_rows.append({"Reference family": reference_family, **row})
        evidence_summary_rows.extend(_build_evidence_summary_rows(record_rows))
        factorial_input_path = scratch_root / f"{reference_family}_overview_factorial_input.csv"
        factorial_output_path = scratch_root / f"{reference_family}_overview_factorial_contributions.csv"
        factorial_input_frame = _build_factorial_input_frame(record_rows)
        factorial_input_frame.to_csv(factorial_input_path, index=False)
        factorial_result = analyze_factorial_anova_fn(factorial_input_path, factorial_output_path, 2)
        factorial_frame = _normalize_factorial_table(factorial_result)
        if not factorial_frame.empty:
            factorial_frame.insert(0, "Reference family", reference_family)
            factorial_frames.append(factorial_frame)
        optimal_rows.extend(_build_optimal_score_rows(record_rows))

    combined_factorial = (
        pd.concat(factorial_frames, ignore_index=True)
        if factorial_frames
        else pd.DataFrame(columns=["Reference family", "Feature", *METRIC_COLUMN_NAMES])
    )
    anova_table_markdown_path.write_text(_render_overview_anova_markdown_table(anova_rows), encoding="utf-8")
    evidence_summary_table_markdown_path.write_text(
        _render_evidence_summary_markdown_table(evidence_summary_rows),
        encoding="utf-8",
    )
    factorial_table_markdown_path.write_text(
        _render_overview_factorial_markdown_table(combined_factorial),
        encoding="utf-8",
    )
    optimal_table_markdown_path.write_text(_render_optimal_markdown_table(optimal_rows), encoding="utf-8")
    return {
        "anova_table_markdown_path": anova_table_markdown_path,
        "evidence_summary_table_markdown_path": evidence_summary_table_markdown_path,
        "factorial_table_markdown_path": factorial_table_markdown_path,
        "optimal_table_markdown_path": optimal_table_markdown_path,
    }


def _render_overview_anova_markdown_table(rows: list[dict[str, float | str | None]]) -> str:
    header = (
        "| Reference family | Variable / metric | BLEU | METEOR | R-1 | R-2 | R-L | Reading ease |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    body = [
        "| "
        + " | ".join(
            [
                str(row["Reference family"]),
                str(row["label"]),
                *(_format_pvalue_cell(_lookup_metric_value(row, metric)) for metric in METRIC_COLUMN_NAMES),
            ]
        )
        + " |"
        for row in rows
    ]
    return "# ANOVA Table\n\n" + "\n".join([header, *body]) + "\n"


def _render_overview_factorial_markdown_table(frame: pd.DataFrame) -> str:
    header = (
        "| Reference family | Feature | BLEU | METEOR | R-1 | R-2 | R-L | Reading ease |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    body = []
    for row in frame.to_dict(orient="records"):
        values = [_format_contribution_cell(row[column]) for column in METRIC_COLUMN_NAMES]
        body.append("| " + " | ".join([str(row["Reference family"]), str(row["Feature"]), *values]) + " |")
    return "# Factorial Contributions\n\n" + "\n".join([header, *body]) + "\n"


def _compute_anova_rows(frame: pd.DataFrame) -> list[dict[str, float | str | None]]:
    rows: list[dict[str, float | str | None]] = []
    for label, factor_column in ANOVA_ROW_SPECS:
        row: dict[str, float | str | None] = {"label": label}
        for metric in METRIC_COLUMN_NAMES:
            row[metric] = _one_way_anova_pvalue(frame, factor_column=factor_column, metric=metric)
        rows.append(row)
    return rows


def _one_way_anova_pvalue(frame: pd.DataFrame, *, factor_column: str, metric: str) -> float | None:
    working = frame.copy()
    series_name = factor_column
    if factor_column in {"ABM", "LLM", "Summarizer", "Evidence"}:
        series_name = factor_column.lower()
    elif factor_column in {"Role", "Insights", "Example"}:
        series_name = factor_column.lower()
    if series_name not in working.columns or metric not in working.columns:
        return None
    working = working[[series_name, metric]].copy()
    working = working.replace("", pd.NA).dropna()
    if working.empty or working[series_name].nunique() < 2:
        return None
    groups = [group[metric].astype(float).tolist() for _, group in working.groupby(series_name)]
    if len(groups) < 2 or any(len(group) == 0 for group in groups):
        return None
    try:
        from scipy.stats import f_oneway

        _stat, pvalue = f_oneway(*groups)
    except Exception:
        return None
    if pvalue is None or not math.isfinite(float(pvalue)):
        return None
    return float(pvalue)


def _write_anova_csv(path: Path, rows: list[dict[str, float | str | None]]) -> None:
    fieldnames = ["Variable", *METRIC_COLUMN_NAMES]
    serialized_rows: list[dict[str, str]] = []
    for row in rows:
        serialized = {"Variable": str(row["label"])}
        for metric in METRIC_COLUMN_NAMES:
            serialized[metric] = _format_pvalue_cell(_lookup_metric_value(row, metric))
        serialized_rows.append(serialized)
    _write_csv(path, fieldnames=fieldnames, rows=serialized_rows)


def _build_factorial_input_frame(record_rows: list[dict[str, str]]) -> pd.DataFrame:
    if not record_rows:
        return pd.DataFrame(columns=["Summarizer", "Evidence", "Role", "Insights", "Example", *METRIC_COLUMN_NAMES])
    normalized_rows: list[dict[str, object]] = []
    for row in record_rows:
        normalized_rows.append(
            {
                "Summarizer": row["summarizer"],
                "Evidence": row["evidence"],
                "Role": "on" if row["role"] == "True" else "off",
                "Insights": "on" if row["insights"] == "True" else "off",
                "Example": "on" if row["example"] == "True" else "off",
                "BLEU": float(row["BLEU"]),
                "METEOR": float(row["METEOR"]),
                "R-1": float(row["R-1"]),
                "R-2": float(row["R-2"]),
                "R-L": float(row["R-L"]),
                "Reading ease": float(row["Reading ease"]),
            }
        )
    return pd.DataFrame(normalized_rows)


def _normalize_factorial_table(frame: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["Feature", *METRIC_COLUMN_NAMES]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    normalized = frame.copy()
    rename_map = {
        column: column.replace("ROUGE-", "R-").replace("Reading_ease", "Reading ease") for column in normalized.columns
    }
    normalized = normalized.rename(columns=rename_map)
    normalized["Feature"] = normalized["Feature"].astype(str).map(_canonicalize_factorial_feature_name)
    normalized = normalized.groupby("Feature", as_index=False, dropna=False).agg("sum")
    for feature in FACTORIAL_FEATURE_ORDER:
        if feature not in set(normalized["Feature"]):
            normalized.loc[len(normalized)] = {column: float("nan") if column != "Feature" else feature for column in columns}
    normalized = normalized[columns].copy()
    normalized["Feature"] = normalized["Feature"].astype(str)
    for metric in METRIC_COLUMN_NAMES:
        metric_total = float(normalized[metric].sum())
        if metric_total > 0:
            normalized[metric] = normalized[metric].astype(float) / metric_total * 100.0
    normalized["_order"] = normalized["Feature"].apply(_factorial_feature_sort_key)
    normalized = normalized.sort_values(["_order", "Feature"]).drop(columns=["_order"]).reset_index(drop=True)
    return normalized


def _build_optimal_score_rows(record_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not record_rows:
        return []
    frame = pd.DataFrame(record_rows).copy()
    group_columns = ["reference_family", "abm", "summarizer", "llm"]
    optimal_rows: list[dict[str, str]] = []
    for group_values, group_frame in frame.groupby(group_columns, sort=True):
        reference_family, abm, summarizer, llm = group_values
        numeric_frame = group_frame.copy()
        for metric in METRIC_COLUMN_NAMES:
            numeric_frame[metric] = numeric_frame[metric].astype(float)
        optimal_rows.append(
            {
                "Reference family": str(reference_family),
                "ABM": str(abm),
                "Summary": str(summarizer),
                "LLM": str(llm),
                "BLEU": _format_float_cell(numeric_frame["BLEU"].max()),
                "METEOR": _format_float_cell(numeric_frame["METEOR"].max()),
                "R-1": _format_float_cell(numeric_frame["R-1"].max()),
                "R-2": _format_float_cell(numeric_frame["R-2"].max()),
                "R-L": _format_float_cell(numeric_frame["R-L"].max()),
                "Reading ease": _format_float_cell(numeric_frame["Reading ease"].max()),
            }
        )
    return optimal_rows


def _build_evidence_summary_rows(record_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not record_rows:
        return []
    frame = pd.DataFrame(record_rows).copy()
    group_columns = ["reference_family", "evidence", "abm"]
    for metric in METRIC_COLUMN_NAMES:
        frame[metric] = frame[metric].astype(float)
    evidence_order = {"plot": 0, "table": 1, "plot+table": 2}
    rows: list[dict[str, str]] = []
    for group_values, group_frame in frame.groupby(group_columns, sort=True):
        reference_family, evidence, abm = group_values
        row: dict[str, str] = {
            "Reference family": str(reference_family),
            "Evidence": str(evidence),
            "ABM": str(abm),
        }
        for metric in METRIC_COLUMN_NAMES:
            row[f"Avg {metric}"] = _format_float_cell(group_frame[metric].mean())
            row[f"Best {metric}"] = _format_float_cell(group_frame[metric].max())
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["Reference family"],
            evidence_order.get(row["Evidence"], len(evidence_order)),
            row["ABM"],
        )
    )
    return rows


def _write_optimal_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _write_csv(path, fieldnames=["Reference family", "ABM", "Summary", "LLM", *METRIC_COLUMN_NAMES], rows=rows)


def _factorial_feature_sort_key(feature: str) -> int:
    try:
        return FACTORIAL_FEATURE_ORDER.index(feature)
    except ValueError:
        return len(FACTORIAL_FEATURE_ORDER)


def _canonicalize_factorial_feature_name(feature: str) -> str:
    alias_map = {
        "Insight": "Insights",
        "Example_AND_Insight": "Insights_AND_Example",
        "Evidence_AND_Insight": "Evidence_AND_Insights",
        "Role_AND_Insight": "Role_AND_Insights",
        "Summarizer_AND_Insight": "Summarizer_AND_Insights",
        "Example_AND_Insights": "Insights_AND_Example",
        "Insights_AND_Role": "Role_AND_Insights",
        "Example_AND_Role": "Role_AND_Example",
        "Example_AND_Summarizer": "Summarizer_AND_Example",
        "Evidence_AND_Summarizer": "Summarizer_AND_Evidence",
        "Insights_AND_Summarizer": "Summarizer_AND_Insights",
        "Role_AND_Summarizer": "Summarizer_AND_Role",
    }
    return alias_map.get(feature, feature)


def _format_float_cell(value: float | int | str) -> str:
    return f"{float(value):.2f}"


def _render_markdown_report(result: QuantitativeSmokeResult) -> str:
    reference_families = ", ".join(sorted(result.reference_roots)) or "none"
    return (
        "# Quantitative Smoke Report\n\n"
        f"- success: `{str(result.success).lower()}`\n"
        f"- record_count: `{result.record_count}`\n"
        f"- failed_record_count: `{len(result.failed_record_ids)}`\n"
        f"- overview_root: `{result.overview_root}`\n"
        f"- reference_families: `{reference_families}`\n"
        f"- quantitative_rows_path: `{result.quantitative_rows_path}`\n"
        f"- structured_results_path: `{result.structured_results_path}`\n"
        f"- anova_csv_path: `{result.anova_csv_path}`\n"
        f"- evidence_summary_table_markdown_path: `{result.evidence_summary_table_markdown_path}`\n"
        f"- factorial_csv_path: `{result.factorial_csv_path}`\n"
        f"- optimal_csv_path: `{result.optimal_csv_path}`\n"
    )


def _prepare_output_root(output_root: Path, *, resume: bool) -> None:
    if output_root.exists() and not resume:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _resolve_previous_quantitative_run_root(*, output_root: Path, current_run_id: str) -> Path | None:
    runs_root = output_root / "runs"
    if not runs_root.exists():
        return None
    candidates = sorted(
        (path for path in runs_root.iterdir() if path.is_dir() and path.name != current_run_id),
        reverse=True,
    )
    return candidates[0] if candidates else None
