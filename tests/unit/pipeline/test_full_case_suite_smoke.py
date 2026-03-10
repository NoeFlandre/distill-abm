from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from distill_abm.llm.adapters.base import LLMAdapter
from distill_abm.pipeline.full_case_matrix_smoke import FullCaseMatrixCaseSpec
from distill_abm.pipeline.full_case_smoke import FullCaseSmokeInput
from distill_abm.pipeline.full_case_suite_current_view import sync_stable_abm_current_view
from distill_abm.pipeline.full_case_suite_progress import (
    FullCaseSuiteProgressAbm,
    build_suite_progress,
    refresh_progress_abm_snapshot,
)
from distill_abm.pipeline.full_case_suite_smoke import run_full_case_suite_smoke
from distill_abm.pipeline.local_qwen_monitor import LocalQwenCaseSnapshot, LocalQwenMonitorSnapshot


class _Adapter(LLMAdapter):
    provider = "mistral"

    def complete(self, request):  # type: ignore[no-untyped-def]
        _ = request
        raise AssertionError("not used in this test")


def test_run_full_case_suite_smoke_writes_outer_artifacts(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("MISTRAL_API_KEY", "debug-token")
    recorded_abms: list[str] = []

    def fake_run_full_case_matrix_smoke(**kwargs):  # type: ignore[no-untyped-def]
        output_root = kwargs["output_root"]
        run_root = output_root / "runs" / "run_1"
        run_root.mkdir(parents=True, exist_ok=True)
        (output_root / "latest_run.txt").write_text(str(run_root), encoding="utf-8")
        recorded_abms.append(kwargs["case_input"].abm)
        return SimpleNamespace(
            success=True,
            report_json_path=run_root / "report.json",
            report_markdown_path=run_root / "report.md",
            review_csv_path=run_root / "review.csv",
            viewer_html_path=run_root / "review.html",
            failed_case_ids=[],
        )

    monkeypatch.setattr(
        "distill_abm.pipeline.full_case_suite_smoke.run_full_case_matrix_smoke",
        fake_run_full_case_matrix_smoke,
    )
    monkeypatch.setattr(
        "distill_abm.pipeline.full_case_suite_progress.collect_local_qwen_monitor_snapshot",
        lambda output_root: LocalQwenMonitorSnapshot(
            output_root=output_root / "runs" / "run_1",
            exists=True,
            mode="smoke",
            total_cases=1,
            completed_cases=1,
            failed_cases=0,
            running_case_id="01_fauna_none_plot_rep1",
            cases=(
                LocalQwenCaseSnapshot(
                    case_id="01_fauna_none_plot_rep1",
                    status="running",
                    label="01_fauna_none_plot_rep1",
                    num_ctx=None,
                    max_tokens=None,
                    context_prompt_length=None,
                    trend_prompt_length=None,
                    context_total_tokens=None,
                    trend_total_tokens=None,
                    error=None,
                    progress_detail="trend plot_07",
                    completed_steps=5,
                    total_steps=11,
                ),
            ),
        ),
    )

    suite_input = {
        "fauna": FullCaseSmokeInput(
            abm="fauna",
            csv_path=tmp_path / "fauna.csv",
            parameters_path=tmp_path / "fauna_params.txt",
            documentation_path=tmp_path / "fauna_docs.txt",
            plots=(),
        ),
        "grazing": FullCaseSmokeInput(
            abm="grazing",
            csv_path=tmp_path / "grazing.csv",
            parameters_path=tmp_path / "grazing_params.txt",
            documentation_path=tmp_path / "grazing_docs.txt",
            plots=(),
        ),
    }
    for path in [
        suite_input["fauna"].csv_path,
        suite_input["fauna"].parameters_path,
        suite_input["fauna"].documentation_path,
        suite_input["grazing"].csv_path,
        suite_input["grazing"].parameters_path,
        suite_input["grazing"].documentation_path,
    ]:
        path.write_text("x", encoding="utf-8")

    cases_by_abm = cast(
        dict[str, tuple[FullCaseMatrixCaseSpec, ...]],
        {
            abm: (
                FullCaseMatrixCaseSpec(
                    case_id=f"01_{abm}_none_plot_rep1",
                    abm=abm,
                    evidence_mode="plot",
                    prompt_variant="none",
                    repetition=1,
                ),
            )
            for abm in suite_input
        },
    )

    result = run_full_case_suite_smoke(
        abm_inputs=suite_input,
        cases_by_abm=cases_by_abm,
        adapter=_Adapter(),
        model="mistral-medium-latest",
        output_root=tmp_path / "suite",
    )

    assert result.success is True
    assert recorded_abms == ["fauna", "grazing"]
    assert result.report_json_path.exists()
    assert result.report_markdown_path.exists()
    assert result.review_csv_path.exists()
    assert result.review_html_path.exists()
    assert (tmp_path / "suite" / "suite_progress.json").exists()
    assert "Mistral Generation Dashboard" in result.review_html_path.read_text(encoding="utf-8")
    assert "Planned cases" in result.report_markdown_path.read_text(encoding="utf-8")


def test_run_full_case_suite_smoke_marks_abm_failure_without_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "debug-token")
    def fake_run_full_case_matrix_smoke(**kwargs):  # type: ignore[no-untyped-def]
        case_input = kwargs["case_input"]
        if case_input.abm == "grazing":
            raise RuntimeError("grazing failed")
        output_root = kwargs["output_root"]
        run_root = output_root / "runs" / "run_1"
        run_root.mkdir(parents=True, exist_ok=True)
        (output_root / "latest_run.txt").write_text(str(run_root), encoding="utf-8")
        return SimpleNamespace(
            success=True,
            report_json_path=run_root / "report.json",
            report_markdown_path=run_root / "report.md",
            review_csv_path=run_root / "review.csv",
            viewer_html_path=run_root / "review.html",
            failed_case_ids=[],
        )

    monkeypatch.setattr(
        "distill_abm.pipeline.full_case_suite_smoke.run_full_case_matrix_smoke",
        fake_run_full_case_matrix_smoke,
    )

    suite_input = {
        "fauna": FullCaseSmokeInput(
            abm="fauna",
            csv_path=tmp_path / "fauna.csv",
            parameters_path=tmp_path / "fauna_params.txt",
            documentation_path=tmp_path / "fauna_docs.txt",
            plots=(),
        ),
        "grazing": FullCaseSmokeInput(
            abm="grazing",
            csv_path=tmp_path / "grazing.csv",
            parameters_path=tmp_path / "grazing_params.txt",
            documentation_path=tmp_path / "grazing_docs.txt",
            plots=(),
        ),
    }
    for path in [
        suite_input["fauna"].csv_path,
        suite_input["fauna"].parameters_path,
        suite_input["fauna"].documentation_path,
        suite_input["grazing"].csv_path,
        suite_input["grazing"].parameters_path,
        suite_input["grazing"].documentation_path,
    ]:
        path.write_text("x", encoding="utf-8")

    cases_by_abm = cast(
        dict[str, tuple[FullCaseMatrixCaseSpec, ...]],
        {
            abm: (
                FullCaseMatrixCaseSpec(
                    case_id=f"01_{abm}_none_plot_rep1",
                    abm=abm,
                    evidence_mode="plot",
                    prompt_variant="none",
                    repetition=1,
                ),
            )
            for abm in suite_input
        },
    )

    result = run_full_case_suite_smoke(
        abm_inputs=suite_input,
        cases_by_abm=cases_by_abm,
        adapter=_Adapter(),
        model="mistral-medium-latest",
        output_root=tmp_path / "suite",
    )

    assert result.success is False
    assert result.failed_abms == ["grazing"]
    grazing = next(item for item in result.abms if item.abm == "grazing")
    assert grazing.success is False
    assert grazing.failed_case_ids == ["abm_runner_failed"]


def test_run_full_case_suite_smoke_rejects_missing_case_specs(tmp_path: Path) -> None:
    suite_input = {
        "fauna": FullCaseSmokeInput(
            abm="fauna",
            csv_path=tmp_path / "fauna.csv",
            parameters_path=tmp_path / "fauna_params.txt",
            documentation_path=tmp_path / "fauna_docs.txt",
            plots=(),
        ),
    }
    for path in [
        suite_input["fauna"].csv_path,
        suite_input["fauna"].parameters_path,
        suite_input["fauna"].documentation_path,
    ]:
        path.write_text("x", encoding="utf-8")

    try:
        run_full_case_suite_smoke(
            abm_inputs=suite_input,
            cases_by_abm={},
            adapter=_Adapter(),
            model="mistral-medium-latest",
            output_root=tmp_path / "suite",
        )
    except ValueError as exc:
        assert "missing case specs for ABM" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_run_full_case_suite_smoke_retries_transient_abm_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "debug-token")
    call_counts: dict[str, int] = {"fauna": 0}
    sleep_calls: list[float] = []

    def fake_run_full_case_matrix_smoke(**kwargs):  # type: ignore[no-untyped-def]
        case_input = kwargs["case_input"]
        call_counts[case_input.abm] = call_counts.get(case_input.abm, 0) + 1
        if case_input.abm == "fauna" and call_counts[case_input.abm] == 1:
            raise RuntimeError("circuit open for mistral:mistral-medium-latest; retry after 12.0s")
        output_root = kwargs["output_root"]
        run_root = output_root / "runs" / f"run_{call_counts[case_input.abm]}"
        run_root.mkdir(parents=True, exist_ok=True)
        (output_root / "latest_run.txt").write_text(str(run_root), encoding="utf-8")
        return SimpleNamespace(
            success=True,
            report_json_path=run_root / "report.json",
            report_markdown_path=run_root / "report.md",
            review_csv_path=run_root / "review.csv",
            viewer_html_path=run_root / "review.html",
            failed_case_ids=[],
        )

    monkeypatch.setattr(
        "distill_abm.pipeline.full_case_suite_smoke.run_full_case_matrix_smoke",
        fake_run_full_case_matrix_smoke,
    )
    monkeypatch.setattr("distill_abm.pipeline.full_case_suite_smoke.time.sleep", sleep_calls.append)

    suite_input = {
        "fauna": FullCaseSmokeInput(
            abm="fauna",
            csv_path=tmp_path / "fauna.csv",
            parameters_path=tmp_path / "fauna_params.txt",
            documentation_path=tmp_path / "fauna_docs.txt",
            plots=(),
        ),
    }
    for path in [
        suite_input["fauna"].csv_path,
        suite_input["fauna"].parameters_path,
        suite_input["fauna"].documentation_path,
    ]:
        path.write_text("x", encoding="utf-8")

    cases_by_abm = cast(
        dict[str, tuple[FullCaseMatrixCaseSpec, ...]],
        {
            "fauna": (
                FullCaseMatrixCaseSpec(
                    case_id="01_fauna_none_plot_rep1",
                    abm="fauna",
                    evidence_mode="plot",
                    prompt_variant="none",
                    repetition=1,
                ),
            )
        },
    )

    result = run_full_case_suite_smoke(
        abm_inputs=suite_input,
        cases_by_abm=cases_by_abm,
        adapter=_Adapter(),
        model="mistral-medium-latest",
        output_root=tmp_path / "suite",
        max_abm_attempts=2,
    )

    assert result.success is True
    assert call_counts["fauna"] == 2
    assert sleep_calls == [60.0]


def test_run_full_case_suite_smoke_rejects_missing_provider_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    suite_input = {
        "fauna": FullCaseSmokeInput(
            abm="fauna",
            csv_path=tmp_path / "fauna.csv",
            parameters_path=tmp_path / "fauna_params.txt",
            documentation_path=tmp_path / "fauna_docs.txt",
            plots=(),
        ),
    }
    for path in [
        suite_input["fauna"].csv_path,
        suite_input["fauna"].parameters_path,
        suite_input["fauna"].documentation_path,
    ]:
        path.write_text("x", encoding="utf-8")
    cases_by_abm = cast(
        dict[str, tuple[FullCaseMatrixCaseSpec, ...]],
        {
            "fauna": (
                FullCaseMatrixCaseSpec(
                    case_id="01_fauna_none_plot_rep1",
                    abm="fauna",
                    evidence_mode="plot",
                    prompt_variant="none",
                    repetition=1,
                ),
            )
        },
    )

    with pytest.raises(ValueError, match="mistral api key missing"):
        run_full_case_suite_smoke(
            abm_inputs=suite_input,
            cases_by_abm=cases_by_abm,
            adapter=_Adapter(),
            model="mistral-medium-latest",
            output_root=tmp_path / "suite",
        )


def test_run_full_case_suite_smoke_rejects_duplicate_active_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "debug-token")
    suite_root = tmp_path / "suite"
    suite_root.mkdir(parents=True, exist_ok=True)
    (suite_root / ".active_run_lock.json").write_text(
        '{"pid": 1, "run_id": "run_existing", "run_root": "/tmp/existing"}',
        encoding="utf-8",
    )

    suite_input = {
        "fauna": FullCaseSmokeInput(
            abm="fauna",
            csv_path=tmp_path / "fauna.csv",
            parameters_path=tmp_path / "fauna_params.txt",
            documentation_path=tmp_path / "fauna_docs.txt",
            plots=(),
        ),
    }
    for path in [
        suite_input["fauna"].csv_path,
        suite_input["fauna"].parameters_path,
        suite_input["fauna"].documentation_path,
    ]:
        path.write_text("x", encoding="utf-8")
    cases_by_abm = cast(
        dict[str, tuple[FullCaseMatrixCaseSpec, ...]],
        {
            "fauna": (
                FullCaseMatrixCaseSpec(
                    case_id="01_fauna_none_plot_rep1",
                    abm="fauna",
                    evidence_mode="plot",
                    prompt_variant="none",
                    repetition=1,
                ),
            )
        },
    )

    with pytest.raises(RuntimeError, match="another run is already active"):
        run_full_case_suite_smoke(
            abm_inputs=suite_input,
            cases_by_abm=cases_by_abm,
            adapter=_Adapter(),
            model="mistral-medium-latest",
            output_root=suite_root,
        )


def test_refresh_progress_abm_snapshot_updates_current_view_and_running_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "suite"
    abm_output_root = output_root / "abms" / "fauna"
    run_root = abm_output_root / "runs" / "run_1"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "run.log.jsonl").write_text('{"event":"x"}\n', encoding="utf-8")
    (run_root / "smoke_full_case_matrix_report.json").write_text("{}", encoding="utf-8")
    (run_root / "smoke_full_case_matrix_report.md").write_text("report", encoding="utf-8")
    (run_root / "request_review.csv").write_text("case_id\n01\n", encoding="utf-8")

    snapshot = LocalQwenMonitorSnapshot(
        output_root=run_root,
        exists=True,
        mode="full_case_matrix",
        total_cases=72,
        completed_cases=10,
        failed_cases=1,
        running_case_id="11_fauna_example_table_rep1",
        cases=(
            LocalQwenCaseSnapshot(
                case_id="11_fauna_example_table_rep1",
                status="running",
                label="11_fauna_example_table_rep1",
                num_ctx=None,
                max_tokens=None,
                context_prompt_length=None,
                trend_prompt_length=None,
                context_total_tokens=None,
                trend_total_tokens=None,
                error=None,
                progress_detail="trend plot_07",
                completed_steps=8,
                total_steps=15,
            ),
        ),
    )
    monkeypatch.setattr(
        "distill_abm.pipeline.full_case_suite_progress.collect_local_qwen_monitor_snapshot",
        lambda _: snapshot,
    )

    progress = FullCaseSuiteProgressAbm(
        abm="fauna",
        status="running",
        planned_case_count=72,
        completed_case_count=0,
        failed_case_count=0,
        run_root=run_root,
        run_log_path=run_root / "run.log.jsonl",
        report_json_path=run_root / "smoke_full_case_matrix_report.json",
        running_case_id=None,
        running_case_status=None,
        running_case_detail=None,
        last_error=None,
    )

    refreshed = refresh_progress_abm_snapshot(output_root=output_root, progress=progress)

    assert refreshed.completed_case_count == 10
    assert refreshed.failed_case_count == 1
    assert refreshed.running_case_id == "11_fauna_example_table_rep1"
    assert refreshed.running_case_status == "running"
    assert refreshed.running_case_detail == "trend plot_07"
    assert (abm_output_root / "current" / "run.log.jsonl").read_text(encoding="utf-8") == '{"event":"x"}\n'
    assert (abm_output_root / "current" / "smoke_full_case_matrix_report.json").read_text(encoding="utf-8") == "{}"
    assert (abm_output_root / "current" / "smoke_full_case_matrix_report.md").read_text(encoding="utf-8") == "report"
    assert (abm_output_root / "current" / "request_review.csv").read_text(encoding="utf-8") == "case_id\n01\n"
    assert (abm_output_root / "latest_run.txt").read_text(encoding="utf-8").strip() == str(run_root)


def test_sync_stable_abm_current_view_returns_typed_paths(tmp_path: Path) -> None:
    abm_output_root = tmp_path / "suite" / "abms" / "fauna"
    run_root = abm_output_root / "runs" / "run_1"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "run.log.jsonl").write_text('{"event":"x"}\n', encoding="utf-8")
    (run_root / "smoke_full_case_matrix_report.json").write_text("{}", encoding="utf-8")
    (run_root / "smoke_full_case_matrix_report.md").write_text("report", encoding="utf-8")
    (run_root / "request_review.csv").write_text("case_id\n01\n", encoding="utf-8")

    stable_paths = sync_stable_abm_current_view(abm_output_root=abm_output_root, run_root=run_root)

    assert stable_paths.run_root == run_root
    assert stable_paths.run_log_path == abm_output_root / "current" / "run.log.jsonl"
    assert stable_paths.report_json_path == abm_output_root / "current" / "smoke_full_case_matrix_report.json"
    assert stable_paths.report_markdown_path == abm_output_root / "current" / "smoke_full_case_matrix_report.md"
    assert stable_paths.review_csv_path == abm_output_root / "current" / "request_review.csv"


def test_build_suite_progress_aggregates_current_case_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "suite"
    run_root = output_root / "runs" / "run_suite"
    started_at = datetime.now(UTC)

    def fake_refresh_progress_abm_snapshot(
        *,
        output_root: Path,
        progress: FullCaseSuiteProgressAbm,
    ) -> FullCaseSuiteProgressAbm:
        _ = output_root
        if progress.abm == "fauna":
            return progress.model_copy(
                update={
                    "status": "completed",
                    "completed_case_count": 72,
                    "failed_case_count": 0,
                }
            )
        return progress.model_copy(
            update={
                "status": "running",
                "completed_case_count": 12,
                "failed_case_count": 2,
                "running_case_id": "15_grazing_role_plot_rep1",
                "running_case_status": "running",
                "running_case_detail": "context",
            }
        )

    monkeypatch.setattr(
        "distill_abm.pipeline.full_case_suite_progress.refresh_progress_abm_snapshot",
        fake_refresh_progress_abm_snapshot,
    )

    base_progress = {
        "fauna": FullCaseSuiteProgressAbm(
            abm="fauna",
            status="pending",
            planned_case_count=72,
            completed_case_count=0,
            failed_case_count=0,
            run_root=run_root,
            run_log_path=run_root / "run.log.jsonl",
            report_json_path=run_root / "report.json",
            running_case_id=None,
            running_case_status=None,
            running_case_detail=None,
            last_error=None,
        ),
        "grazing": FullCaseSuiteProgressAbm(
            abm="grazing",
            status="pending",
            planned_case_count=72,
            completed_case_count=0,
            failed_case_count=0,
            run_root=run_root,
            run_log_path=run_root / "run.log.jsonl",
            report_json_path=run_root / "report.json",
            running_case_id=None,
            running_case_status=None,
            running_case_detail=None,
            last_error=None,
        ),
    }

    progress = build_suite_progress(
        run_id="run_suite",
        run_root=run_root,
        output_root=output_root,
        model="mistral-medium-latest",
        started_at=started_at,
        status="running",
        current_abm="grazing",
        current_attempt=2,
        remaining_abms=["grazing"],
        progress_by_name=base_progress,
    )

    assert progress.total_abms == 2
    assert progress.completed_abm_count == 1
    assert progress.failed_abm_count == 0
    assert progress.planned_case_count == 144
    assert progress.completed_case_count == 84
    assert progress.failed_case_count == 2
    assert progress.current_case_id == "15_grazing_role_plot_rep1"
    assert progress.current_case_status == "running"
    assert progress.current_case_detail == "context"
