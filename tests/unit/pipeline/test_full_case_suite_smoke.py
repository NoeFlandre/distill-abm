from __future__ import annotations

import csv
import json
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
        (run_root / "prompt_compression_summary.json").write_text(
            '{"run_root":"'
            + str(run_root)
            + '","total_entries":1,"triggered_entries":1,"total_compressions":1,'
            + '"entries":[{"source_run_root":"'
            + str(run_root)
            + '","scope":"full_case_matrix_plot","case_id":"01_case","abm":"'
            + kwargs["case_input"].abm
            + '","evidence_mode":"table","prompt_variant":"role","repetition":1,"plot_index":1,'
            + '"artifacts_dir":"'
            + str(run_root / "cases" / "01_case" / "03_trends" / "plot_01")
            + '","compression_artifact_path":"'
            + str(run_root / "cases" / "01_case" / "03_trends" / "plot_01" / "trend_prompt_compression.json")
            + '","pre_compression_prompt_path":"'
            + str(run_root / "cases" / "01_case" / "03_trends" / "plot_01" / "trend_prompt_pre_compression.txt")
            + '","compressed_prompt_path":"'
            + str(run_root / "cases" / "01_case" / "03_trends" / "plot_01" / "trend_prompt_compressed.txt")
            + '","triggered":true,"compression_count":1,"attempt_count":2,'
            + '"attempts":[{"attempt_index":1,"table_downsample_stride":1,"compression_tier":0,"prompt_length":100},'
            + '{"attempt_index":2,"table_downsample_stride":2,"compression_tier":1,"prompt_length":90}]}]}',
            encoding="utf-8",
        )
        recorded_abms.append(kwargs["case_input"].abm)
        return SimpleNamespace(
            success=True,
            report_json_path=run_root / "report.json",
            report_markdown_path=run_root / "report.md",
            review_csv_path=run_root / "review.csv",
            viewer_html_path=run_root / "review.html",
            prompt_compression_summary_path=run_root / "prompt_compression_summary.json",
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
            abm: tuple(
                FullCaseMatrixCaseSpec(
                    case_id=f"{idx:02d}_{abm}_none_plot_rep1",
                    abm=abm,
                    evidence_mode="plot",
                    prompt_variant="none",
                    repetition=1,
                )
                for idx in range(1, 73)
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
    assert result.prompt_compression_summary_path.exists()
    assert (tmp_path / "suite" / "suite_progress.json").exists()
    assert "Mistral Generation Dashboard" in result.review_html_path.read_text(encoding="utf-8")
    assert "Planned cases" in result.report_markdown_path.read_text(encoding="utf-8")
    suite_summary = result.prompt_compression_summary_path.read_text(encoding="utf-8")
    assert '"triggered_entries": 2' in suite_summary
    assert '"total_compressions": 2' in suite_summary


def test_run_full_case_suite_smoke_aggregates_run_observability_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "debug-token")

    def fake_run_full_case_matrix_smoke(**kwargs):  # type: ignore[no-untyped-def]
        output_root = kwargs["output_root"]
        run_root = output_root / "runs" / "run_1"
        run_root.mkdir(parents=True, exist_ok=True)
        (output_root / "latest_run.txt").write_text(str(run_root), encoding="utf-8")
        observability_csv = run_root / "run_observability.csv"
        observability_csv.write_text(
            (
                "run_root,request_kind,case_id,abm,provider,model,temperature,runtime_provider,"
                "runtime_precision,table_downsample_stride,compression_tier,prompt_compression_applied\n"
                f"{run_root},trend,01_{kwargs['case_input'].abm},"
                f"{kwargs['case_input'].abm},openrouter,qwen/qwen3.5-27b,1.0,Fireworks,fp8,2,1,true\n"
            ),
            encoding="utf-8",
        )
        summary_json = run_root / "run_observability_summary.json"
        summary_json.write_text(
            json.dumps(
                {
                    "request_count": 1,
                    "runtime_providers": ["Fireworks"],
                    "runtime_precisions": ["fp8"],
                    "compression": {"request_count_with_compression": 1},
                }
            ),
            encoding="utf-8",
        )
        (run_root / "run_observability_summary.md").write_text("# summary\n", encoding="utf-8")
        return SimpleNamespace(
            success=True,
            report_json_path=run_root / "report.json",
            report_markdown_path=run_root / "report.md",
            review_csv_path=run_root / "review.csv",
            viewer_html_path=run_root / "review.html",
            failed_case_ids=[],
            observability_csv_path=observability_csv,
            observability_summary_json_path=summary_json,
            observability_summary_markdown_path=run_root / "run_observability_summary.md",
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

    assert result.observability_csv_path is not None
    assert result.observability_summary_json_path is not None
    rows = list(csv.DictReader(result.observability_csv_path.open(encoding="utf-8")))
    assert len(rows) == 2
    assert {row["abm"] for row in rows} == {"fauna", "grazing"}
    assert {row["runtime_provider"] for row in rows} == {"Fireworks"}
    assert {row["compression_tier"] for row in rows} == {"1"}
    summary = json.loads(result.observability_summary_json_path.read_text(encoding="utf-8"))
    assert summary["request_count"] == 2
    assert summary["runtime_providers"] == ["Fireworks"]
    assert summary["compression"]["request_count_with_compression"] == 2


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
            abm: tuple(
                FullCaseMatrixCaseSpec(
                    case_id=f"{index:02d}_{abm}_none_plot_rep1",
                    abm=abm,
                    evidence_mode="plot",
                    prompt_variant="none",
                    repetition=1,
                )
                for index in range(1, 73)
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


def test_run_full_case_suite_smoke_refreshes_suite_progress_during_active_abm_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "debug-token")
    suite_root = tmp_path / "suite"
    nested_run_root = suite_root / "abms" / "fauna" / "runs" / "run_live"
    nested_run_root.mkdir(parents=True, exist_ok=True)
    (nested_run_root / "run.log.jsonl").write_text('{"event":"x"}\n', encoding="utf-8")
    (nested_run_root / "smoke_full_case_matrix_report.json").write_text("{}", encoding="utf-8")
    (nested_run_root / "smoke_full_case_matrix_report.md").write_text("report", encoding="utf-8")
    (nested_run_root / "request_review.csv").write_text("case_id\n01\n", encoding="utf-8")
    live_completed_cases = {"value": 0}

    def fake_run_full_case_matrix_smoke(**kwargs):  # type: ignore[no-untyped-def]
        output_root = kwargs["output_root"]
        (output_root / "latest_run.txt").write_text(str(nested_run_root), encoding="utf-8")
        live_completed_cases["value"] = 51
        on_case_completed = kwargs.get("on_case_completed")
        assert on_case_completed is not None
        on_case_completed(
            SimpleNamespace(
                case_id="51_fauna_role_plot_plus_table_rep3",
                success=True,
            )
        )
        progress_payload = (suite_root / "suite_progress.json").read_text(encoding="utf-8")
        assert '"completed_case_count":51' in progress_payload.replace(" ", "")
        return SimpleNamespace(
            success=True,
            report_json_path=nested_run_root / "report.json",
            report_markdown_path=nested_run_root / "report.md",
            review_csv_path=nested_run_root / "review.csv",
            viewer_html_path=nested_run_root / "review.html",
            failed_case_ids=[],
        )

    def fake_collect_local_qwen_monitor_snapshot(_output_root: Path) -> LocalQwenMonitorSnapshot:
        completed_cases = live_completed_cases["value"]
        running_case_id = None if completed_cases >= 72 else "52_fauna_role_plot_plus_table_rep3"
        status = "completed" if completed_cases >= 72 else "running"
        progress_detail = "done" if completed_cases >= 72 else "trend"
        return LocalQwenMonitorSnapshot(
            output_root=nested_run_root,
            exists=True,
            mode="smoke",
            total_cases=72,
            completed_cases=completed_cases,
            failed_cases=0,
            running_case_id=running_case_id,
            cases=(
                LocalQwenCaseSnapshot(
                    case_id=running_case_id or "72_fauna_role_plot_plus_table_rep3",
                    status=status,
                    label=running_case_id or "72_fauna_role_plot_plus_table_rep3",
                    num_ctx=None,
                    max_tokens=None,
                    context_prompt_length=None,
                    trend_prompt_length=None,
                    context_total_tokens=None,
                    trend_total_tokens=None,
                    error=None,
                    progress_detail=progress_detail,
                    completed_steps=completed_cases,
                    total_steps=72,
                ),
            ),
        )

    monkeypatch.setattr(
        "distill_abm.pipeline.full_case_suite_smoke.run_full_case_matrix_smoke",
        fake_run_full_case_matrix_smoke,
    )
    monkeypatch.setattr(
        "distill_abm.pipeline.full_case_suite_progress.collect_local_qwen_monitor_snapshot",
        fake_collect_local_qwen_monitor_snapshot,
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

    result = run_full_case_suite_smoke(
        abm_inputs=suite_input,
        cases_by_abm=cases_by_abm,
        adapter=_Adapter(),
        model="mistral-medium-latest",
        output_root=suite_root,
    )

    assert result.success is True


def test_run_full_case_suite_smoke_resume_skips_completed_abms_and_starts_at_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "debug-token")
    suite_root = tmp_path / "suite"
    call_order: list[str] = []

    def write_run_artifacts(abm: str, run_name: str) -> Path:
        run_root = suite_root / "abms" / abm / "runs" / run_name
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "run.log.jsonl").write_text('{"event":"x"}\n', encoding="utf-8")
        (run_root / "smoke_full_case_matrix_report.json").write_text("{}", encoding="utf-8")
        (run_root / "smoke_full_case_matrix_report.md").write_text("report", encoding="utf-8")
        (run_root / "request_review.csv").write_text("case_id\n01\n", encoding="utf-8")
        (run_root / "review.html").write_text("<html></html>", encoding="utf-8")
        return run_root

    fauna_run_root = write_run_artifacts("fauna", "run_done")
    grazing_run_root = write_run_artifacts("grazing", "run_done")
    milk_run_root = write_run_artifacts("milk_consumption", "run_partial")

    def fake_collect_local_qwen_monitor_snapshot(output_root: Path) -> LocalQwenMonitorSnapshot:
        abm = output_root.name
        if abm == "fauna":
            run_root = fauna_run_root
            completed_cases = 72
            running_case_id = None
            status = "completed"
            progress_detail = "done"
        elif abm == "grazing":
            run_root = grazing_run_root
            completed_cases = 72
            running_case_id = None
            status = "completed"
            progress_detail = "done"
        else:
            run_root = milk_run_root
            completed_cases = 47
            running_case_id = "48_milk_consumption_all_three_plot_plus_table_rep2"
            status = "running"
            progress_detail = "trend"
        return LocalQwenMonitorSnapshot(
            output_root=run_root,
            exists=True,
            mode="smoke",
            total_cases=72,
            completed_cases=completed_cases,
            failed_cases=0,
            running_case_id=running_case_id,
            cases=(
                LocalQwenCaseSnapshot(
                    case_id=running_case_id or f"72_{abm}_all_three_plot_plus_table_rep3",
                    status=status,
                    label=running_case_id or f"72_{abm}_all_three_plot_plus_table_rep3",
                    num_ctx=None,
                    max_tokens=None,
                    context_prompt_length=None,
                    trend_prompt_length=None,
                    context_total_tokens=None,
                    trend_total_tokens=None,
                    error=None,
                    progress_detail=progress_detail,
                    completed_steps=completed_cases,
                    total_steps=72,
                ),
            ),
        )

    def fake_run_full_case_matrix_smoke(**kwargs):  # type: ignore[no-untyped-def]
        case_input = kwargs["case_input"]
        call_order.append(case_input.abm)
        output_root = kwargs["output_root"]
        run_root = output_root / "runs" / "run_resumed"
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
    monkeypatch.setattr(
        "distill_abm.pipeline.full_case_suite_progress.collect_local_qwen_monitor_snapshot",
        fake_collect_local_qwen_monitor_snapshot,
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
        "milk_consumption": FullCaseSmokeInput(
            abm="milk_consumption",
            csv_path=tmp_path / "milk.csv",
            parameters_path=tmp_path / "milk_params.txt",
            documentation_path=tmp_path / "milk_docs.txt",
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
        suite_input["milk_consumption"].csv_path,
        suite_input["milk_consumption"].parameters_path,
        suite_input["milk_consumption"].documentation_path,
    ]:
        path.write_text("x", encoding="utf-8")

    cases_by_abm = cast(
        dict[str, tuple[FullCaseMatrixCaseSpec, ...]],
        {
            abm: tuple(
                FullCaseMatrixCaseSpec(
                    case_id=f"{index:02d}_{abm}_none_plot_rep1",
                    abm=abm,
                    evidence_mode="plot",
                    prompt_variant="none",
                    repetition=1,
                )
                for index in range(1, 73)
            )
            for abm in suite_input
        },
    )

    result = run_full_case_suite_smoke(
        abm_inputs=suite_input,
        cases_by_abm=cases_by_abm,
        adapter=_Adapter(),
        model="mistral-medium-latest",
        output_root=suite_root,
        resume_existing=True,
    )

    progress_payload = (suite_root / "suite_progress.json").read_text(encoding="utf-8")
    assert result.success is True
    assert call_order == ["milk_consumption"]
    assert '"completed_case_count":191' in progress_payload.replace(" ", "")
    assert '"current_abm":null' in progress_payload.replace(" ", "")


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


def test_refresh_progress_abm_snapshot_preserves_progress_on_nested_snapshot_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "suite"
    abm_output_root = output_root / "abms" / "fauna"
    abm_output_root.mkdir(parents=True, exist_ok=True)
    progress = FullCaseSuiteProgressAbm(
        abm="fauna",
        status="running",
        planned_case_count=72,
        completed_case_count=5,
        failed_case_count=1,
        run_root=abm_output_root / "runs" / "run_1",
        run_log_path=abm_output_root / "runs" / "run_1" / "run.log.jsonl",
        report_json_path=abm_output_root / "runs" / "run_1" / "smoke_full_case_matrix_report.json",
        running_case_id="01_case",
        running_case_status="running",
        running_case_detail="trend plot_01",
        last_error=None,
    )
    monkeypatch.setattr(
        "distill_abm.pipeline.full_case_suite_progress.collect_local_qwen_monitor_snapshot",
        lambda _: (_ for _ in ()).throw(ValueError("bad nested snapshot")),
    )

    refreshed = refresh_progress_abm_snapshot(output_root=output_root, progress=progress)

    assert refreshed == progress


def test_build_suite_progress_aggregates_current_case_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
