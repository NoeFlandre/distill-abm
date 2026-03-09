"""Run one generation smoke across all ABMs using the full-case matrix core."""

from __future__ import annotations

import csv
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from distill_abm.llm.adapters.base import LLMAdapter
from distill_abm.llm.resilience import CIRCUIT_BREAKER_OPEN_SECONDS, is_transient_provider_error
from distill_abm.pipeline.full_case_matrix_smoke import (
    FullCaseMatrixCaseSpec,
    run_full_case_matrix_smoke,
)
from distill_abm.pipeline.full_case_smoke import FullCaseSmokeInput
from distill_abm.pipeline.local_qwen_sample_smoke import _write_json, _write_text
from distill_abm.pipeline.run_artifact_contracts import latest_run_pointer_path, run_log_path
from distill_abm.structured_logging import attach_json_log_file, get_logger, log_event

LOGGER = get_logger(__name__)
DEFAULT_MAX_ABM_ATTEMPTS = 3
SUITE_PROGRESS_FILENAME = "suite_progress.json"


class FullCaseSuiteAbmResult(BaseModel):
    """One ABM result inside the all-ABMs generation smoke."""

    abm: str
    success: bool
    run_root: Path
    report_json_path: Path
    report_markdown_path: Path
    review_csv_path: Path
    review_html_path: Path
    planned_case_count: int
    failed_case_ids: list[str] = Field(default_factory=list)


class FullCaseSuiteSmokeResult(BaseModel):
    """Top-level result for the all-ABMs generation smoke."""

    started_at_utc: str
    finished_at_utc: str
    output_root: Path
    run_id: str
    run_root: Path
    run_log_path: Path
    report_json_path: Path
    report_markdown_path: Path
    review_csv_path: Path
    review_html_path: Path
    success: bool
    failed_abms: list[str] = Field(default_factory=list)
    abms: list[FullCaseSuiteAbmResult] = Field(default_factory=list)


class FullCaseSuiteProgressAbm(BaseModel):
    """Live progress snapshot for one ABM inside a suite run."""

    abm: str
    status: str
    attempt: int | None = None
    planned_case_count: int
    failed_case_count: int = 0
    run_root: Path | None = None
    run_log_path: Path | None = None
    review_html_path: Path | None = None
    report_json_path: Path | None = None
    last_error: str | None = None


class FullCaseSuiteProgress(BaseModel):
    """Stable suite-level progress contract for live monitoring and HTML refresh."""

    run_id: str
    run_root: Path
    output_root: Path
    model: str
    started_at_utc: str
    finished_at_utc: str | None = None
    status: str
    current_abm: str | None = None
    current_attempt: int | None = None
    total_abms: int
    completed_abm_count: int
    failed_abm_count: int
    planned_case_count: int
    failed_case_count: int
    remaining_abms: list[str] = Field(default_factory=list)
    abms: list[FullCaseSuiteProgressAbm] = Field(default_factory=list)


def _validate_suite_inputs(
    *,
    abm_inputs: dict[str, FullCaseSmokeInput],
    cases_by_abm: dict[str, tuple[FullCaseMatrixCaseSpec, ...]],
) -> None:
    """Validate that suite inputs and case specs are aligned before starting work."""

    for abm, case_input in abm_inputs.items():
        if abm not in cases_by_abm:
            raise ValueError(f"missing case specs for ABM '{abm}'")
        if case_input.abm != abm:
            raise ValueError(f"ABM input key '{abm}' does not match case input '{case_input.abm}'")


def run_full_case_suite_smoke(
    *,
    abm_inputs: dict[str, FullCaseSmokeInput],
    cases_by_abm: dict[str, tuple[FullCaseMatrixCaseSpec, ...]],
    adapter: LLMAdapter,
    model: str,
    output_root: Path,
    max_tokens: int = 32768,
    max_retries: int | None = None,
    retry_backoff_seconds: float | None = None,
    resume_existing: bool = True,
    max_abm_attempts: int = DEFAULT_MAX_ABM_ATTEMPTS,
) -> FullCaseSuiteSmokeResult:
    """Run the full-case matrix smoke across all configured ABMs."""

    started_at = datetime.now(UTC)
    _validate_suite_inputs(abm_inputs=abm_inputs, cases_by_abm=cases_by_abm)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = started_at.strftime("run_%Y%m%d_%H%M%S_%f")
    run_root = output_root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    _write_text(latest_run_pointer_path(output_root), str(run_root))
    log_path = attach_json_log_file(run_log_path(run_root))
    progress_path = output_root / SUITE_PROGRESS_FILENAME
    log_event(
        LOGGER,
        "full_case_suite_start",
        model=model,
        abm_count=len(abm_inputs),
        run_root=str(run_root),
    )

    abm_results_by_name: dict[str, FullCaseSuiteAbmResult] = {}
    progress_by_name = {
        abm: FullCaseSuiteProgressAbm(
            abm=abm,
            status="pending",
            planned_case_count=len(cases_by_abm[abm]),
        )
        for abm in abm_inputs
    }
    remaining_abms = list(abm_inputs)
    _write_suite_progress(
        output_root=output_root,
        progress_path=progress_path,
        progress=_build_suite_progress(
            run_id=run_id,
            run_root=run_root,
            output_root=output_root,
            model=model,
            started_at=started_at,
            status="running",
            current_abm=None,
            current_attempt=None,
            remaining_abms=remaining_abms,
            progress_by_name=progress_by_name,
        ),
    )
    for attempt in range(1, max(max_abm_attempts, 1) + 1):
        next_remaining: list[str] = []
        transient_failure_seen = False
        for abm in remaining_abms:
            case_input = abm_inputs[abm]
            abm_output_root = output_root / "abms" / abm
            case_specs = cases_by_abm[abm]
            progress_by_name[abm] = progress_by_name[abm].model_copy(
                update={"status": "running", "attempt": attempt, "last_error": None}
            )
            _write_suite_progress(
                output_root=output_root,
                progress_path=progress_path,
                progress=_build_suite_progress(
                    run_id=run_id,
                    run_root=run_root,
                    output_root=output_root,
                    model=model,
                    started_at=started_at,
                    status="running",
                    current_abm=abm,
                    current_attempt=attempt,
                    remaining_abms=remaining_abms,
                    progress_by_name=progress_by_name,
                ),
            )
            log_event(
                LOGGER,
                "full_case_suite_abm_start",
                abm=abm,
                case_count=len(case_specs),
                output_root=str(abm_output_root),
                attempt=attempt,
            )
            try:
                matrix_result = run_full_case_matrix_smoke(
                    case_input=case_input,
                    adapter=adapter,
                    model=model,
                    output_root=abm_output_root,
                    cases=case_specs,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                    resume_existing=resume_existing,
                )
                latest_abm_run = Path((abm_output_root / "latest_run.txt").read_text(encoding="utf-8").strip())
                abm_result = FullCaseSuiteAbmResult(
                    abm=abm,
                    success=matrix_result.success,
                    run_root=latest_abm_run,
                    report_json_path=matrix_result.report_json_path,
                    report_markdown_path=matrix_result.report_markdown_path,
                    review_csv_path=matrix_result.review_csv_path,
                    review_html_path=matrix_result.viewer_html_path,
                    planned_case_count=len(case_specs),
                    failed_case_ids=list(matrix_result.failed_case_ids),
                )
            except Exception as exc:
                error_text = str(exc)
                abm_result = FullCaseSuiteAbmResult(
                    abm=abm,
                    success=False,
                    run_root=abm_output_root,
                    report_json_path=abm_output_root / "smoke_full_case_matrix_report.json",
                    report_markdown_path=abm_output_root / "smoke_full_case_matrix_report.md",
                    review_csv_path=abm_output_root / "request_review.csv",
                    review_html_path=abm_output_root / "review.html",
                    planned_case_count=len(case_specs),
                    failed_case_ids=["abm_runner_failed"],
                )
                log_event(
                    LOGGER,
                    "full_case_suite_abm_failed",
                    abm=abm,
                    error=error_text,
                    output_root=str(abm_output_root),
                    attempt=attempt,
                )
                if is_transient_provider_error(error_text) and attempt < max_abm_attempts:
                    next_remaining.append(abm)
                    transient_failure_seen = True
                progress_by_name[abm] = progress_by_name[abm].model_copy(
                    update={
                        "status": "retrying" if abm in next_remaining else "failed",
                        "attempt": attempt,
                        "failed_case_count": len(abm_result.failed_case_ids),
                        "run_root": abm_result.run_root,
                        "run_log_path": run_log_path(abm_result.run_root),
                        "review_html_path": abm_result.review_html_path,
                        "report_json_path": abm_result.report_json_path,
                        "last_error": error_text,
                    }
                )
            abm_results_by_name[abm] = abm_result
            if abm_result.success:
                progress_by_name[abm] = progress_by_name[abm].model_copy(
                    update={
                        "status": "completed",
                        "attempt": attempt,
                        "failed_case_count": len(abm_result.failed_case_ids),
                        "run_root": abm_result.run_root,
                        "run_log_path": run_log_path(abm_result.run_root),
                        "review_html_path": abm_result.review_html_path,
                        "report_json_path": abm_result.report_json_path,
                    }
                )
            log_event(
                LOGGER,
                "full_case_suite_abm_complete",
                abm=abm,
                success=abm_result.success,
                failed_case_count=len(abm_result.failed_case_ids),
                run_root=str(abm_result.run_root),
                attempt=attempt,
            )
            _write_suite_progress(
                output_root=output_root,
                progress_path=progress_path,
                progress=_build_suite_progress(
                    run_id=run_id,
                    run_root=run_root,
                    output_root=output_root,
                    model=model,
                    started_at=started_at,
                    status="running",
                    current_abm=abm,
                    current_attempt=attempt,
                    remaining_abms=next_remaining or [item for item in remaining_abms if item != abm],
                    progress_by_name=progress_by_name,
                ),
            )
        if not next_remaining:
            break
        if transient_failure_seen:
            log_event(
                LOGGER,
                "full_case_suite_retry_wait",
                wait_seconds=CIRCUIT_BREAKER_OPEN_SECONDS,
                remaining_abms=next_remaining,
                next_attempt=attempt + 1,
            )
            _write_suite_progress(
                output_root=output_root,
                progress_path=progress_path,
                progress=_build_suite_progress(
                    run_id=run_id,
                    run_root=run_root,
                    output_root=output_root,
                    model=model,
                    started_at=started_at,
                    status="waiting_to_retry",
                    current_abm=None,
                    current_attempt=attempt + 1,
                    remaining_abms=next_remaining,
                    progress_by_name=progress_by_name,
                ),
            )
            time.sleep(CIRCUIT_BREAKER_OPEN_SECONDS)
        remaining_abms = next_remaining

    abm_results = [abm_results_by_name[abm] for abm in abm_inputs]
    summary_rows = [
        {
            "abm": abm_result.abm,
            "success": str(abm_result.success).lower(),
            "planned_case_count": str(abm_result.planned_case_count),
            "failed_case_count": str(len(abm_result.failed_case_ids)),
            "run_root": str(abm_result.run_root),
            "run_log_path": str(run_log_path(abm_result.run_root)),
            "report_json_path": str(abm_result.report_json_path),
            "review_csv_path": str(abm_result.review_csv_path),
            "review_html_path": str(abm_result.review_html_path),
        }
        for abm_result in abm_results
    ]

    finished_at = datetime.now(UTC)
    result = FullCaseSuiteSmokeResult(
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
        output_root=output_root,
        run_id=run_id,
        run_root=run_root,
        run_log_path=log_path,
        report_json_path=run_root / "smoke_full_case_suite_report.json",
        report_markdown_path=run_root / "smoke_full_case_suite_report.md",
        review_csv_path=run_root / "review.csv",
        review_html_path=run_root / "review.html",
        success=all(item.success for item in abm_results),
        failed_abms=[item.abm for item in abm_results if not item.success],
        abms=abm_results,
    )
    _write_json(result.report_json_path, result.model_dump(mode="json"))
    _write_text(result.report_markdown_path, _render_report(result))
    _write_summary_csv(result.review_csv_path, summary_rows)
    _write_text(result.review_html_path, _render_html(result))
    _write_suite_progress(
        output_root=output_root,
        progress_path=progress_path,
        progress=_build_suite_progress(
            run_id=run_id,
            run_root=run_root,
            output_root=output_root,
            model=model,
            started_at=started_at,
            status="completed" if result.success else "failed",
            current_abm=None,
            current_attempt=None,
            remaining_abms=[],
            progress_by_name=progress_by_name,
            finished_at=finished_at,
        ),
    )
    log_event(
        LOGGER,
        "full_case_suite_complete",
        success=result.success,
        failed_abms=result.failed_abms,
        run_root=str(run_root),
    )
    return result


def _write_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "abm",
        "success",
        "planned_case_count",
        "failed_case_count",
        "run_root",
        "run_log_path",
        "report_json_path",
        "review_csv_path",
        "review_html_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_report(result: FullCaseSuiteSmokeResult) -> str:
    lines = [
        "# Full-Case Suite Smoke",
        "",
        f"- success: `{str(result.success).lower()}`",
        f"- failed ABMs: `{', '.join(result.failed_abms) if result.failed_abms else 'none'}`",
        "",
        "| ABM | Success | Planned cases | Failed cases | Review HTML |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in result.abms:
        failed_case_count = len(item.failed_case_ids)
        lines.append(
            "| "
            f"{item.abm} | "
            f"{str(item.success).lower()} | "
            f"{item.planned_case_count} | "
            f"{failed_case_count} | "
            f"{item.review_html_path} |"
        )
    return "\n".join(lines)


def _render_html(result: FullCaseSuiteSmokeResult) -> str:
    total_abms = len(result.abms)
    total_cases = sum(item.planned_case_count for item in result.abms)
    total_failed_cases = sum(len(item.failed_case_ids) for item in result.abms)
    cards = "\n".join(
        (
            '<article class="case-card">'
            f"<header><h2>{item.abm}</h2><span class=\"status {'ok' if item.success else 'bad'}\">"
            f"{'success' if item.success else 'failed'}</span></header>"
            f"<dl><div><dt>Planned cases</dt><dd>{item.planned_case_count}</dd></div>"
            f"<div><dt>Failed cases</dt><dd>{len(item.failed_case_ids)}</dd></div></dl>"
            f'<p class="links"><a href="{item.review_html_path}">reviewer</a>'
            f'<a href="{item.report_markdown_path}">report</a>'
            f'<a href="{run_log_path(item.run_root)}">log</a></p>'
            "</article>"
        )
        for item in result.abms
    )
    summary_cards = (
        f'<section class="summary-grid">'
        f'<article class="summary-card"><h2>Run</h2><p>{result.run_id}</p></article>'
        f'<article class="summary-card"><h2>ABMs</h2><p>{total_abms}</p></article>'
        f'<article class="summary-card"><h2>Planned cases</h2><p>{total_cases}</p></article>'
        f'<article class="summary-card"><h2>Failed cases</h2><p>{total_failed_cases}</p></article>'
        f"<article class=\"summary-card\"><h2>Status</h2><p>{'success' if result.success else 'failed'}</p></article>"
        "</section>"
    )
    cards = "\n".join(line for line in [cards] if line)
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>Mistral Generation Dashboard</title>'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<style>"
        "body{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "margin:0;background:#f5f3ef;color:#171717}"
        "header{padding:28px 32px 12px 32px}"
        "h1{margin:0 0 6px 0;font-size:28px}"
        ".subtle{color:#5f5a53;font-size:14px}"
        ".summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));"
        "gap:12px;padding:0 32px 20px 32px}"
        ".summary-card,.case-card{background:#fff;border:1px solid #ddd6cc;"
        "border-radius:14px;box-shadow:0 1px 3px rgba(0,0,0,.04)}"
        ".summary-card{padding:16px 18px}"
        ".summary-card h2{margin:0 0 8px 0;font-size:13px;color:#6b6358;text-transform:uppercase;letter-spacing:.04em}"
        ".summary-card p{margin:0;font-size:24px}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;padding:0 32px 32px 32px}"
        ".case-card{padding:18px}"
        ".case-card header{display:flex;justify-content:space-between;align-items:center;padding:0;margin:0 0 14px 0}"
        ".case-card h2{margin:0;font-size:20px}"
        ".status{font-size:12px;padding:4px 8px;border-radius:999px;text-transform:uppercase;letter-spacing:.04em}"
        ".status.ok{background:#e8f7ea;color:#1f6a32}"
        ".status.bad{background:#fdebec;color:#9b2335}"
        "dl{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:0 0 14px 0}"
        "dt{font-size:12px;color:#6b6358;text-transform:uppercase;letter-spacing:.04em}"
        "dd{margin:4px 0 0 0;font-size:18px}"
        ".links{display:flex;gap:14px;flex-wrap:wrap;margin:0}"
        "a{color:#0d5bd7;text-decoration:none}"
        "a:hover{text-decoration:underline}"
        "</style></head><body>"
        f'<header><h1>Mistral Generation Dashboard</h1><p class="subtle">Run {result.run_id}</p></header>'
        f"{summary_cards}<main>{cards}</main></body></html>"
    )


def _build_suite_progress(
    *,
    run_id: str,
    run_root: Path,
    output_root: Path,
    model: str,
    started_at: datetime,
    status: str,
    current_abm: str | None,
    current_attempt: int | None,
    remaining_abms: list[str],
    progress_by_name: dict[str, FullCaseSuiteProgressAbm],
    finished_at: datetime | None = None,
) -> FullCaseSuiteProgress:
    abm_progress = [progress_by_name[abm] for abm in progress_by_name]
    completed_abm_count = sum(1 for item in abm_progress if item.status == "completed")
    failed_abm_count = sum(1 for item in abm_progress if item.status == "failed")
    planned_case_count = sum(item.planned_case_count for item in abm_progress)
    failed_case_count = sum(item.failed_case_count for item in abm_progress)
    return FullCaseSuiteProgress(
        run_id=run_id,
        run_root=run_root,
        output_root=output_root,
        model=model,
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat() if finished_at is not None else None,
        status=status,
        current_abm=current_abm,
        current_attempt=current_attempt,
        total_abms=len(abm_progress),
        completed_abm_count=completed_abm_count,
        failed_abm_count=failed_abm_count,
        planned_case_count=planned_case_count,
        failed_case_count=failed_case_count,
        remaining_abms=list(remaining_abms),
        abms=abm_progress,
    )


def _write_suite_progress(*, output_root: Path, progress_path: Path, progress: FullCaseSuiteProgress) -> None:
    _write_json(progress_path, progress.model_dump(mode="json"))
    _write_text(output_root / "review.html", _render_live_html(progress))


def _render_live_html(progress: FullCaseSuiteProgress) -> str:
    style_rules = [
        (
            "body{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"
            "'Segoe UI',sans-serif;margin:0;background:#f5f3ef;color:#171717}"
        ),
        "header{padding:28px 32px 12px 32px}",
        "h1{margin:0 0 6px 0;font-size:28px}",
        ".subtle{color:#5f5a53;font-size:14px}",
        (
            ".summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));"
            "gap:12px;padding:0 32px 20px 32px}"
        ),
        (
            ".summary-card,.abm-card{background:#fff;border:1px solid #ddd6cc;"
            "border-radius:14px;box-shadow:0 1px 3px rgba(0,0,0,.04)}"
        ),
        ".summary-card{padding:16px 18px}",
        ".summary-card h2{margin:0 0 8px 0;font-size:13px;color:#6b6358;text-transform:uppercase;letter-spacing:.04em}",
        ".summary-card p{margin:0;font-size:24px}",
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;padding:0 32px 32px 32px}",
        ".abm-card{padding:18px}",
        ".abm-card h2{margin:0 0 8px 0;font-size:20px}",
        ".status{font-size:12px;padding:4px 8px;border-radius:999px;text-transform:uppercase;letter-spacing:.04em}",
        ".status.ok{background:#e8f7ea;color:#1f6a32}",
        ".status.bad{background:#fdebec;color:#9b2335}",
        ".status.running{background:#fff4d8;color:#8a5a00}",
        ".status.waiting{background:#e8f0ff;color:#1d4ed8}",
        ".status.pending{background:#efefef;color:#555}",
        "dl{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:0 0 14px 0}",
        "dt{font-size:12px;color:#6b6358;text-transform:uppercase;letter-spacing:.04em}",
        "dd{margin:4px 0 0 0;font-size:18px}",
        ".links{display:flex;gap:14px;flex-wrap:wrap;margin:0}",
        ".links a{color:#0d5bd7;text-decoration:none}",
        ".links a:hover{text-decoration:underline}",
        ".error{color:#9b2335;font-size:13px;margin-top:12px;white-space:pre-wrap}",
        ".dim{color:#6b6358}",
    ]
    def status_class(status: str) -> str:
        if status == "completed":
            return "ok"
        if status == "failed":
            return "bad"
        if status == "running":
            return "running"
        if status == "waiting_to_retry":
            return "waiting"
        return "pending"

    summary_cards = [
        ("Run", progress.run_id),
        ("Status", progress.status),
        ("Current ABM", progress.current_abm or "-"),
        ("Attempt", "-" if progress.current_attempt is None else str(progress.current_attempt)),
        ("Completed ABMs", f"{progress.completed_abm_count}/{progress.total_abms}"),
        ("Failed cases", str(progress.failed_case_count)),
    ]
    summary_html = "".join(
        f'<article class="summary-card"><h2>{title}</h2><p>{value}</p></article>'
        for title, value in summary_cards
    )
    abm_cards: list[str] = []
    for item in progress.abms:
        links: list[str] = []
        if item.review_html_path is not None:
            links.append(f'<a href="{item.review_html_path}">reviewer</a>')
        if item.report_json_path is not None:
            links.append(f'<a href="{item.report_json_path}">report</a>')
        if item.run_log_path is not None:
            links.append(f'<a href="{item.run_log_path}">log</a>')
        abm_cards.append(
            "".join(
                [
                    '<article class="abm-card">',
                    '<header style="display:flex;justify-content:space-between;align-items:center">',
                    f"<h2>{item.abm}</h2>",
                    f'<span class="status {status_class(item.status)}">{item.status}</span>',
                    "</header>",
                    "<dl>",
                    f"<div><dt>Planned cases</dt><dd>{item.planned_case_count}</dd></div>",
                    f"<div><dt>Failed cases</dt><dd>{item.failed_case_count}</dd></div>",
                    f"<div><dt>Attempt</dt><dd>{item.attempt if item.attempt is not None else '-'}</dd></div>",
                    (
                        '<div><dt>Run root</dt><dd class="dim">'
                        f'{item.run_root if item.run_root is not None else "-"}'
                        "</dd></div>"
                    ),
                    "</dl>",
                    f'<p class="links">{"".join(links)}</p>',
                    f'<p class="error">{item.last_error}</p>' if item.last_error else "",
                    "</article>",
                ]
            )
        )
    return "".join(
        [
            '<!doctype html><html><head><meta charset="utf-8"><title>Mistral Generation Dashboard</title>',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            '<meta http-equiv="refresh" content="3">',
            "<style>",
            "".join(style_rules),
            "</style></head><body>",
            (
                '<header><h1>Mistral Generation Dashboard</h1><p class="subtle">'
                "This page refreshes automatically while the suite is running."
                "</p></header>"
            ),
            f'<section class="summary-grid">{summary_html}</section>',
            f'<main>{"".join(abm_cards)}</main>',
            "</body></html>",
        ]
    )
