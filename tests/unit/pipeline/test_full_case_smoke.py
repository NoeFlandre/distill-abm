from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path

import pytest

import distill_abm.pipeline.full_case_matrix_smoke as full_case_matrix_smoke
import distill_abm.pipeline.full_case_smoke as full_case_smoke
from distill_abm.llm.adapters.base import LLMAdapter, LLMResponse
from distill_abm.pipeline.full_case_matrix_smoke import (
    DEFAULT_MATRIX_PASS_WAIT_SECONDS,
    FullCaseMatrixCaseResult,
    FullCaseMatrixCaseSpec,
    build_full_case_matrix_case_specs,
    compute_matrix_retry_wait_seconds,
    run_full_case_matrix_smoke,
)
from distill_abm.pipeline.full_case_smoke import (
    FullCasePlotInput,
    FullCaseSmokeInput,
    FullCaseValidationState,
    _backfill_validation_state_from_artifacts,
    resolve_parallel_trend_workers,
    run_full_case_smoke,
)


class _FakeAdapter(LLMAdapter):
    provider = "openrouter"

    def __init__(self) -> None:
        self._calls = 0

    def complete(self, request):  # type: ignore[no-untyped-def]
        self._calls += 1
        payload = {"response_text": f"response-{self._calls}"}
        return LLMResponse(
            provider="openrouter",
            model=request.model,
            text=f'{{"response_text": "response-{self._calls}"}}',
            raw={
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "choices": [{"message": {"content": payload}}],
            },
        )

    @property
    def context_calls(self) -> int:
        return self._calls


class _FlakyTrendAdapter(LLMAdapter):
    provider = "openrouter"

    def __init__(self) -> None:
        self._calls = 0

    def complete(self, request):  # type: ignore[no-untyped-def]
        self._calls += 1
        if self._calls == 2:
            text = (
                '{"response_text":"The analysis is currently unavailable. '
                "The requested information cannot be retrieved or interpreted at this time. "
                'Please try again later or consult additional resources for further details."}'
            )
        else:
            text = f'{{"response_text":"response-{self._calls}"}}'
        return LLMResponse(
            provider="openrouter",
            model=request.model,
            text=text,
            raw={
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "choices": [{"message": {"content": text}}],
            },
        )


class _ConcurrentAdapter(LLMAdapter):
    provider = "openrouter"

    def __init__(self) -> None:
        self._calls = 0
        self._active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def complete(self, request):  # type: ignore[no-untyped-def]
        with self._lock:
            self._calls += 1
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            call_number = self._calls
        time.sleep(0.05)
        with self._lock:
            self._active -= 1
        return LLMResponse(
            provider="openrouter",
            model=request.model,
            text=f'{{"response_text":"response-{call_number}"}}',
            raw={
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "choices": [{"message": {"content": f'{{"response_text":"response-{call_number}"}}'}}],
            },
        )


class _ContextOverflowThenSuccessAdapter(LLMAdapter):
    provider = "openrouter"

    def __init__(self) -> None:
        self._calls = 0

    def complete(self, request):  # type: ignore[no-untyped-def]
        self._calls += 1
        if self._calls == 1:
            return LLMResponse(
                provider="openrouter",
                model=request.model,
                text='{"response_text":"response-1"}',
                raw={
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    "choices": [{"message": {"content": '{"response_text":"response-1"}'}}],
                },
            )
        if request.metadata.get("table_downsample_stride") == 1:
            raise ValueError(
                "This model's maximum context length is 131072 tokens. Your request has too many input tokens."
            )
        return LLMResponse(
            provider="openrouter",
            model=request.model,
            text=f'{{"response_text":"response-{self._calls}"}}',
            raw={
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "choices": [{"message": {"content": f'{{"response_text":"response-{self._calls}"}}'}}],
            },
        )


def test_backfill_validation_state_skips_empty_artifacts(tmp_path: Path) -> None:
    context_dir = tmp_path / "02_context"
    trends_dir = tmp_path / "03_trends"
    context_dir.mkdir(parents=True)
    (trends_dir / "plot_01").mkdir(parents=True)
    (context_dir / "context_output.txt").write_text("", encoding="utf-8")
    (context_dir / "context_trace.json").write_text("{}", encoding="utf-8")
    (trends_dir / "plot_01" / "trend_output.txt").write_text("", encoding="utf-8")
    (trends_dir / "plot_01" / "trend_trace.json").write_text("{}", encoding="utf-8")

    validation_state = FullCaseValidationState(context={}, trends={})
    _backfill_validation_state_from_artifacts(
        validation_state=validation_state,
        context_dir=context_dir,
        trends_dir=trends_dir,
        plots=(
            FullCasePlotInput(
                plot_index=1,
                reporter_pattern="metric one",
                plot_description="This plot represents herd size.",
                plot_path=tmp_path / "plot.png",
            ),
        ),
    )

    assert validation_state.context == {}
    assert validation_state.trends == {}


class _RuntimeMetadataAdapter(LLMAdapter):
    provider = "openrouter"

    def __init__(self) -> None:
        self._calls = 0

    def complete(self, request):  # type: ignore[no-untyped-def]
        self._calls += 1
        text = f'{{"response_text":"response-{self._calls}"}}'
        return LLMResponse(
            provider="openrouter",
            model=request.model,
            text=text,
            raw={
                "provider": {"name": "Fireworks", "precision": "fp8"},
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                "choices": [{"message": {"content": text}}],
            },
        )


class _InvalidStructuredAdapter(LLMAdapter):
    provider = "openrouter"

    def complete(self, request):  # type: ignore[no-untyped-def]
        return LLMResponse(
            provider="openrouter",
            model=request.model,
            text="not-json",
            raw={
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "choices": [{"message": {"content": "not-json"}}],
            },
        )


class _SerialFakeAdapter(_FakeAdapter):
    provider = "mistral"


class _ContextFailureAdapter(LLMAdapter):
    provider = "openrouter"

    def complete(self, request):  # type: ignore[no-untyped-def]
        _ = request
        return LLMResponse(
            provider="openrouter",
            model="openrouter/test",
            text="",
            raw={
                "message": {"content": "", "thinking": "reasoning only"},
                "done_reason": "length",
                "eval_count": 1024,
                "prompt_eval_count": 300,
            },
        )
def test_resolve_parallel_trend_workers_reduces_for_mistral() -> None:
    assert resolve_parallel_trend_workers("mistral") == 1
    assert resolve_parallel_trend_workers("openrouter") == 6


def test_run_full_case_smoke_writes_context_and_all_trends(tmp_path: Path) -> None:
    csv_path = tmp_path / "simulation.csv"
    csv_path.write_text("tick;metric one;metric two\n0;1;2\n1;3;4\n", encoding="utf-8")
    parameters_path = tmp_path / "parameters.txt"
    parameters_path.write_text("parameter narrative", encoding="utf-8")
    documentation_path = tmp_path / "documentation.txt"
    documentation_path.write_text("documentation body", encoding="utf-8")
    plot_one = tmp_path / "1.png"
    plot_one.write_bytes(b"plot-one")
    plot_two = tmp_path / "2.png"
    plot_two.write_bytes(b"plot-two")

    result = run_full_case_smoke(
        case_input=FullCaseSmokeInput(
            abm="grazing",
            csv_path=csv_path,
            parameters_path=parameters_path,
            documentation_path=documentation_path,
            plots=(
                FullCasePlotInput(
                    plot_index=1,
                    reporter_pattern="metric one",
                    plot_description="First plot",
                    plot_path=plot_one,
                ),
                FullCasePlotInput(
                    plot_index=2,
                    reporter_pattern="metric two",
                    plot_description="Second plot",
                    plot_path=plot_two,
                ),
            ),
        ),
        adapter=_FakeAdapter(),
        model="nvidia/nemotron-nano-12b-v2-vl:free",
        output_root=tmp_path / "out",
        evidence_mode="table",
        prompt_variant="role",
        max_tokens=128,
    )

    assert result.success is True
    assert (result.case_dir / "02_context" / "context_output.txt").read_text(encoding="utf-8") == "response-1"
    trend_outputs = {
        "1": (result.case_dir / "03_trends" / "plot_01" / "trend_output.txt").read_text(encoding="utf-8"),
        "2": (result.case_dir / "03_trends" / "plot_02" / "trend_output.txt").read_text(encoding="utf-8"),
    }
    assert set(trend_outputs.values()) == {"response-2", "response-3"}
    rows = list(csv.DictReader(result.review_csv_path.open(encoding="utf-8")))
    assert len(rows) == 3
    assert {row["plot_index"] for row in rows} == {"context", "1", "2"}
    row_by_plot = {row["plot_index"]: row for row in rows}
    assert row_by_plot["context"]["validation_status"] == "accepted"
    assert row_by_plot["1"]["validation_status"] == "accepted"
    assert row_by_plot["2"]["validation_status"] == "accepted"
    assert result.review_csv_path.read_text(encoding="utf-8").splitlines()[0] == (
        "plot_index,reporter_pattern,plot_description,trend_prompt_path,"
        "trend_output_path,image_path,table_csv_path,success,error,validation_status"
    )
    compression_payload = json.loads(
        (result.case_dir / "03_trends" / "plot_01" / "trend_prompt_compression.json").read_text(encoding="utf-8")
    )
    assert compression_payload["triggered"] is False
    assert compression_payload["compression_count"] == 0
    assert compression_payload["attempt_count"] == 1
    run_summary = json.loads(result.prompt_compression_summary_path.read_text(encoding="utf-8"))
    assert run_summary["total_entries"] == 2
    assert run_summary["triggered_entries"] == 0
    assert run_summary["total_compressions"] == 0
    assert {entry["plot_index"] for entry in run_summary["entries"]} == {1, 2}


def test_run_full_case_smoke_resume_reuses_context_and_accepted_trends(tmp_path: Path) -> None:
    csv_path = tmp_path / "simulation.csv"
    csv_path.write_text("tick;metric one;metric two\n0;1;2\n1;3;4\n", encoding="utf-8")
    parameters_path = tmp_path / "parameters.txt"
    parameters_path.write_text("parameter narrative", encoding="utf-8")
    documentation_path = tmp_path / "documentation.txt"
    documentation_path.write_text("documentation body", encoding="utf-8")
    plot_one = tmp_path / "1.png"
    plot_one.write_bytes(b"plot-one")
    plot_two = tmp_path / "2.png"
    plot_two.write_bytes(b"plot-two")
    case_input = FullCaseSmokeInput(
        abm="grazing",
        csv_path=csv_path,
        parameters_path=parameters_path,
        documentation_path=documentation_path,
        plots=(
            FullCasePlotInput(
                plot_index=1,
                reporter_pattern="metric one",
                plot_description="First plot",
                plot_path=plot_one,
            ),
            FullCasePlotInput(
                plot_index=2,
                reporter_pattern="metric two",
                plot_description="Second plot",
                plot_path=plot_two,
            ),
        ),
    )
    output_root = tmp_path / "out"
    first_adapter = _FakeAdapter()
    first = run_full_case_smoke(
        case_input=case_input,
        adapter=first_adapter,
        model="nvidia/nemotron-nano-12b-v2-vl:free",
        output_root=output_root,
        evidence_mode="table",
        prompt_variant="role",
        max_tokens=128,
        resume_existing=True,
    )
    assert first.success is True
    assert first_adapter._calls == 3

    validation_path = output_root / "cases" / "01_grazing_role_table_full_case" / "validation_state.json"
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    payload["trends"]["2"]["status"] = "retry"
    validation_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    second_adapter = _FakeAdapter()
    resumed = run_full_case_smoke(
        case_input=case_input,
        adapter=second_adapter,
        model="nvidia/nemotron-nano-12b-v2-vl:free",
        output_root=output_root,
        evidence_mode="table",
        prompt_variant="role",
        max_tokens=128,
        resume_existing=True,
    )

    assert resumed.success is True
    assert second_adapter._calls == 1
    rows = list(csv.DictReader(resumed.review_csv_path.open(encoding="utf-8")))
    row_by_plot = {row["plot_index"]: row for row in rows}
    assert row_by_plot["1"]["validation_status"] == "accepted"
    assert row_by_plot["2"]["validation_status"] == "accepted"


def test_run_full_case_smoke_ignores_stale_prompt_compression_artifacts_when_context_fails(tmp_path: Path) -> None:
    csv_path = tmp_path / "simulation.csv"
    csv_path.write_text("tick;metric one\n0;1\n1;3\n", encoding="utf-8")
    parameters_path = tmp_path / "parameters.txt"
    parameters_path.write_text("parameter narrative", encoding="utf-8")
    documentation_path = tmp_path / "documentation.txt"
    documentation_path.write_text("documentation body", encoding="utf-8")
    plot_one = tmp_path / "1.png"
    plot_one.write_bytes(b"plot-one")
    output_root = tmp_path / "out"
    trend_dir = output_root / "cases" / "01_grazing_role_table_full_case" / "03_trends" / "plot_01"
    trend_dir.mkdir(parents=True, exist_ok=True)
    (trend_dir / "trend_prompt_compression.json").write_text(
        json.dumps(
            {
                "triggered": True,
                "compression_count": 1,
                "attempt_count": 2,
                "attempts": [
                    {"attempt_index": 1, "table_downsample_stride": 1, "compression_tier": 0, "prompt_length": 100},
                    {"attempt_index": 2, "table_downsample_stride": 2, "compression_tier": 1, "prompt_length": 80},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_full_case_smoke(
        case_input=FullCaseSmokeInput(
            abm="grazing",
            csv_path=csv_path,
            parameters_path=parameters_path,
            documentation_path=documentation_path,
            plots=(
                FullCasePlotInput(
                    plot_index=1,
                    reporter_pattern="metric one",
                    plot_description="First plot",
                    plot_path=plot_one,
                ),
            ),
        ),
        adapter=_ContextFailureAdapter(),
        model="nvidia/nemotron-nano-12b-v2-vl:free",
        output_root=output_root,
        evidence_mode="table",
        prompt_variant="role",
        max_tokens=128,
    )

    assert result.success is False
    run_summary = json.loads(result.prompt_compression_summary_path.read_text(encoding="utf-8"))
    assert run_summary["total_entries"] == 0
    assert run_summary["triggered_entries"] == 0
    assert run_summary["total_compressions"] == 0
    assert run_summary["entries"] == []


def test_run_full_case_smoke_resume_reruns_invalid_accepted_trend_output(tmp_path: Path) -> None:
    csv_path = tmp_path / "simulation.csv"
    csv_path.write_text("tick;metric one;metric two\n0;1;2\n1;3;4\n", encoding="utf-8")
    parameters_path = tmp_path / "parameters.txt"
    parameters_path.write_text("parameter narrative", encoding="utf-8")
    documentation_path = tmp_path / "documentation.txt"
    documentation_path.write_text("documentation body", encoding="utf-8")
    plot_one = tmp_path / "1.png"
    plot_one.write_bytes(b"plot-one")
    plot_two = tmp_path / "2.png"
    plot_two.write_bytes(b"plot-two")
    case_input = FullCaseSmokeInput(
        abm="grazing",
        csv_path=csv_path,
        parameters_path=parameters_path,
        documentation_path=documentation_path,
        plots=(
            FullCasePlotInput(
                plot_index=1,
                reporter_pattern="metric one",
                plot_description="First plot",
                plot_path=plot_one,
            ),
            FullCasePlotInput(
                plot_index=2,
                reporter_pattern="metric two",
                plot_description="Second plot",
                plot_path=plot_two,
            ),
        ),
    )
    output_root = tmp_path / "out"
    first = run_full_case_smoke(
        case_input=case_input,
        adapter=_FakeAdapter(),
        model="nvidia/nemotron-nano-12b-v2-vl:free",
        output_root=output_root,
        evidence_mode="table",
        prompt_variant="role",
        max_tokens=128,
        resume_existing=True,
    )
    assert first.success is True

    case_dir = output_root / "cases" / "01_grazing_role_table_full_case"
    trend_dir = case_dir / "03_trends" / "plot_02"
    (trend_dir / "trend_output.txt").write_text("The analysis is currently unavailable.", encoding="utf-8")

    second_adapter = _FakeAdapter()
    resumed = run_full_case_smoke(
        case_input=case_input,
        adapter=second_adapter,
        model="nvidia/nemotron-nano-12b-v2-vl:free",
        output_root=output_root,
        evidence_mode="table",
        prompt_variant="role",
        max_tokens=128,
        resume_existing=True,
    )

    assert resumed.success is True
    assert second_adapter._calls == 1
    assert (trend_dir / "trend_output.txt").read_text(encoding="utf-8") == "response-1"


def test_run_full_case_smoke_executes_multiple_trends_concurrently(tmp_path: Path) -> None:
    csv_path = tmp_path / "simulation.csv"
    csv_path.write_text("tick;metric one;metric two\n0;1;2\n1;3;4\n", encoding="utf-8")
    parameters_path = tmp_path / "parameters.txt"
    parameters_path.write_text("parameter narrative", encoding="utf-8")
    documentation_path = tmp_path / "documentation.txt"
    documentation_path.write_text("documentation body", encoding="utf-8")
    plot_one = tmp_path / "1.png"
    plot_one.write_bytes(b"plot-one")
    plot_two = tmp_path / "2.png"
    plot_two.write_bytes(b"plot-two")
    plot_three = tmp_path / "3.png"
    plot_three.write_bytes(b"plot-three")
    adapter = _ConcurrentAdapter()

    result = run_full_case_smoke(
        case_input=FullCaseSmokeInput(
            abm="grazing",
            csv_path=csv_path,
            parameters_path=parameters_path,
            documentation_path=documentation_path,
            plots=(
                FullCasePlotInput(
                    plot_index=1,
                    reporter_pattern="metric one",
                    plot_description="First plot",
                    plot_path=plot_one,
                ),
                FullCasePlotInput(
                    plot_index=2,
                    reporter_pattern="metric two",
                    plot_description="Second plot",
                    plot_path=plot_two,
                ),
                FullCasePlotInput(
                    plot_index=3,
                    reporter_pattern="metric one",
                    plot_description="Third plot",
                    plot_path=plot_three,
                ),
            ),
        ),
        adapter=adapter,
        model="nvidia/nemotron-nano-12b-v2-vl:free",
        output_root=tmp_path / "out",
        evidence_mode="plot",
        prompt_variant="role",
        max_tokens=128,
    )

    assert result.success is True
    assert adapter.max_active >= 2
    assert [trend.plot_index for trend in result.trend_results] == [1, 2, 3]


def test_build_full_case_matrix_case_specs_covers_all_combinations() -> None:
    specs = build_full_case_matrix_case_specs(
        abm="grazing",
        evidence_modes=("plot", "table", "plot+table"),
        prompt_variants=("none", "role"),
        repetitions=(1, 2),
    )

    assert len(specs) == 12
    assert specs[0] == FullCaseMatrixCaseSpec(
        case_id="01_grazing_none_plot_rep1",
        abm="grazing",
        evidence_mode="plot",
        prompt_variant="none",
        repetition=1,
    )
    assert specs[-1] == FullCaseMatrixCaseSpec(
        case_id="12_grazing_role_plot_plus_table_rep2",
        abm="grazing",
        evidence_mode="plot+table",
        prompt_variant="role",
        repetition=2,
    )


def test_run_full_case_matrix_smoke_reuses_identical_context_across_cases(tmp_path: Path) -> None:
    csv_path = tmp_path / "simulation.csv"
    csv_path.write_text("tick;metric one\n0;1\n1;3\n", encoding="utf-8")
    parameters_path = tmp_path / "parameters.txt"
    parameters_path.write_text("parameter narrative", encoding="utf-8")
    documentation_path = tmp_path / "documentation.txt"
    documentation_path.write_text("documentation body", encoding="utf-8")
    plot_one = tmp_path / "1.png"
    plot_one.write_bytes(b"plot-one")
    case_input = FullCaseSmokeInput(
        abm="grazing",
        csv_path=csv_path,
        parameters_path=parameters_path,
        documentation_path=documentation_path,
        plots=(
            FullCasePlotInput(
                plot_index=1,
                reporter_pattern="metric one",
                plot_description="This plot represents herd size.",
                plot_path=plot_one,
            ),
        ),
    )
    adapter = _FakeAdapter()
    result = run_full_case_matrix_smoke(
        case_input=case_input,
        adapter=adapter,
        model="nvidia/nemotron-nano-12b-v2-vl:free",
        output_root=tmp_path / "out",
        cases=(
            FullCaseMatrixCaseSpec(
                case_id="01_grazing_role_plot_rep1",
                abm="grazing",
                evidence_mode="plot",
                prompt_variant="role",
                repetition=1,
            ),
            FullCaseMatrixCaseSpec(
                case_id="02_grazing_role_table_rep1",
                abm="grazing",
                evidence_mode="table",
                prompt_variant="role",
                repetition=1,
            ),
        ),
        max_tokens=128,
        resume_existing=True,
    )

    assert result.success is True
    assert result.observability_csv_path is not None
    assert result.observability_summary_json_path is not None
    assert len(result.cases) == 2
    assert adapter._calls == 3
    context_outputs = [
        (case.case_dir / "02_context" / "context_output.txt").read_text(encoding="utf-8") for case in result.cases
    ]
    assert context_outputs == ["response-1", "response-1"]
    assert result.cases[0].resumed_from_existing is False
    assert result.cases[1].resumed_from_existing is False
    rows = list(csv.DictReader(result.observability_csv_path.open(encoding="utf-8")))
    assert len(rows) == 4
    context_rows = [row for row in rows if row["request_kind"] == "context"]
    assert len(context_rows) == 2
    assert [row["counts_toward_run_totals"] for row in context_rows].count("true") == 1
    assert [row["reused_from_shared_context_cache"] for row in context_rows].count("true") == 1
    assert {row["context_materialization_source"] for row in context_rows} == {
        "fresh_request",
        "shared_context_cache",
    }
    summary = json.loads(result.observability_summary_json_path.read_text(encoding="utf-8"))
    assert summary["observed_row_count"] == 4
    assert summary["request_count"] == 3
    assert summary["reused_request_count"] == 1
    assert summary["request_counts_by_kind"] == {"context": 1, "trend": 2}
    assert summary["usage_totals"] == {"completion_tokens": 3, "prompt_tokens": 3, "total_tokens": 6}


def test_run_full_case_matrix_smoke_writes_run_observability_artifacts(tmp_path: Path) -> None:
    csv_path = tmp_path / "simulation.csv"
    csv_path.write_text("tick;metric one\n0;1\n1;3\n", encoding="utf-8")
    parameters_path = tmp_path / "parameters.txt"
    parameters_path.write_text("parameter narrative", encoding="utf-8")
    documentation_path = tmp_path / "documentation.txt"
    documentation_path.write_text("documentation body", encoding="utf-8")
    plot_one = tmp_path / "1.png"
    plot_one.write_bytes(b"plot-one")

    result = run_full_case_matrix_smoke(
        case_input=FullCaseSmokeInput(
            abm="grazing",
            csv_path=csv_path,
            parameters_path=parameters_path,
            documentation_path=documentation_path,
            plots=(
                FullCasePlotInput(
                    plot_index=1,
                    reporter_pattern="metric one",
                    plot_description="This plot represents herd size.",
                    plot_path=plot_one,
                ),
            ),
        ),
        adapter=_RuntimeMetadataAdapter(),
        model="qwen/qwen3.5-27b",
        output_root=tmp_path / "out",
        cases=(
            FullCaseMatrixCaseSpec(
                case_id="01_grazing_role_table_rep1",
                abm="grazing",
                evidence_mode="table",
                prompt_variant="role",
                repetition=1,
            ),
        ),
        max_tokens=128,
        resume_existing=True,
    )

    assert result.success is True
    assert result.observability_csv_path is not None
    assert result.observability_summary_json_path is not None
    assert result.observability_summary_markdown_path is not None
    rows = list(csv.DictReader(result.observability_csv_path.open(encoding="utf-8")))
    assert len(rows) == 2
    row_by_kind = {row["request_kind"]: row for row in rows}
    assert row_by_kind["context"]["provider"] == "openrouter"
    assert row_by_kind["context"]["runtime_provider"] == "Fireworks"
    assert row_by_kind["context"]["runtime_precision"] == "fp8"
    assert row_by_kind["context"]["temperature"] == "1.0"
    assert row_by_kind["context"]["counts_toward_run_totals"] == "true"
    assert row_by_kind["trend"]["table_downsample_stride"] == "1"
    assert row_by_kind["trend"]["compression_tier"] == "0"
    assert row_by_kind["trend"]["prompt_compression_applied"] == "false"
    summary = json.loads(result.observability_summary_json_path.read_text(encoding="utf-8"))
    assert summary["request_count"] == 2
    assert summary["observed_row_count"] == 2
    assert summary["runtime_providers"] == ["Fireworks"]
    assert summary["runtime_precisions"] == ["fp8"]
    assert summary["compression"]["request_count_with_compression"] == 0
    assert row_by_kind["context"]["context_materialization_source"] == "fresh_request"


def test_run_full_case_matrix_smoke_observability_excludes_reused_previous_run_requests(tmp_path: Path) -> None:
    csv_path = tmp_path / "simulation.csv"
    csv_path.write_text("tick;metric one\n0;1\n1;3\n", encoding="utf-8")
    parameters_path = tmp_path / "parameters.txt"
    parameters_path.write_text("parameter narrative", encoding="utf-8")
    documentation_path = tmp_path / "documentation.txt"
    documentation_path.write_text("documentation body", encoding="utf-8")
    plot_one = tmp_path / "1.png"
    plot_one.write_bytes(b"plot-one")
    case_input = FullCaseSmokeInput(
        abm="grazing",
        csv_path=csv_path,
        parameters_path=parameters_path,
        documentation_path=documentation_path,
        plots=(
            FullCasePlotInput(
                plot_index=1,
                reporter_pattern="metric one",
                plot_description="This plot represents herd size.",
                plot_path=plot_one,
            ),
        ),
    )
    cases = (
        FullCaseMatrixCaseSpec(
            case_id="01_grazing_role_table_rep1",
            abm="grazing",
            evidence_mode="table",
            prompt_variant="role",
            repetition=1,
        ),
    )
    first = run_full_case_matrix_smoke(
        case_input=case_input,
        adapter=_FakeAdapter(),
        model="qwen/qwen3.5-27b",
        output_root=tmp_path / "out",
        cases=cases,
        max_tokens=128,
        resume_existing=True,
    )
    assert first.success is True

    resumed = run_full_case_matrix_smoke(
        case_input=case_input,
        adapter=_FakeAdapter(),
        model="qwen/qwen3.5-27b",
        output_root=tmp_path / "out",
        cases=cases,
        max_tokens=128,
        resume_existing=True,
    )

    assert resumed.observability_csv_path is not None
    assert resumed.observability_summary_json_path is not None
    rows = list(csv.DictReader(resumed.observability_csv_path.open(encoding="utf-8")))
    assert len(rows) == 2
    assert {row["reused_from_previous_run"] for row in rows} == {"true"}
    assert {row["counts_toward_run_totals"] for row in rows} == {"false"}
    assert {row["context_materialization_source"] for row in rows if row["request_kind"] == "context"} == {
        "resumed_previous_run"
    }
    summary = json.loads(resumed.observability_summary_json_path.read_text(encoding="utf-8"))
    assert summary["observed_row_count"] == 2
    assert summary["request_count"] == 0
    assert summary["reused_request_count"] == 2
    assert summary["usage_totals"] == {}


def test_run_full_case_matrix_smoke_observability_preserves_failed_context_request(tmp_path: Path) -> None:
    csv_path = tmp_path / "simulation.csv"
    csv_path.write_text("tick;metric one\n0;1\n1;3\n", encoding="utf-8")
    parameters_path = tmp_path / "parameters.txt"
    parameters_path.write_text("parameter narrative", encoding="utf-8")
    documentation_path = tmp_path / "documentation.txt"
    documentation_path.write_text("documentation body", encoding="utf-8")
    plot_one = tmp_path / "1.png"
    plot_one.write_bytes(b"plot-one")

    result = run_full_case_matrix_smoke(
        case_input=FullCaseSmokeInput(
            abm="grazing",
            csv_path=csv_path,
            parameters_path=parameters_path,
            documentation_path=documentation_path,
            plots=(
                FullCasePlotInput(
                    plot_index=1,
                    reporter_pattern="metric one",
                    plot_description="This plot represents herd size.",
                    plot_path=plot_one,
                ),
            ),
        ),
        adapter=_InvalidStructuredAdapter(),
        model="qwen/qwen3.5-27b",
        output_root=tmp_path / "out",
        cases=(
            FullCaseMatrixCaseSpec(
                case_id="01_grazing_none_plot_rep1",
                abm="grazing",
                evidence_mode="plot",
                prompt_variant="none",
                repetition=1,
            ),
        ),
        max_tokens=128,
        resume_existing=True,
    )

    assert result.success is False
    assert result.observability_csv_path is not None
    assert result.observability_summary_json_path is not None
    rows = list(csv.DictReader(result.observability_csv_path.open(encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    assert row["request_kind"] == "context"
    assert row["success"] == "false"
    assert row["validation_status"] == "retry"
    assert row["counts_toward_run_totals"] == "true"
    assert row["context_materialization_source"] == "fresh_request"
    assert "structured" in row["error"].lower() or "json" in row["error"].lower()
    summary = json.loads(result.observability_summary_json_path.read_text(encoding="utf-8"))
    assert summary["observed_row_count"] == 1
    assert summary["request_count"] == 1
    assert summary["request_counts_by_kind"] == {"context": 1}


def test_run_full_case_matrix_smoke_observability_excludes_shared_context_reuse_seeded_by_resumed_case(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "simulation.csv"
    csv_path.write_text("tick;metric one\n0;1\n1;3\n", encoding="utf-8")
    parameters_path = tmp_path / "parameters.txt"
    parameters_path.write_text("parameter narrative", encoding="utf-8")
    documentation_path = tmp_path / "documentation.txt"
    documentation_path.write_text("documentation body", encoding="utf-8")
    plot_one = tmp_path / "1.png"
    plot_one.write_bytes(b"plot-one")
    case_input = FullCaseSmokeInput(
        abm="grazing",
        csv_path=csv_path,
        parameters_path=parameters_path,
        documentation_path=documentation_path,
        plots=(
            FullCasePlotInput(
                plot_index=1,
                reporter_pattern="metric one",
                plot_description="This plot represents herd size.",
                plot_path=plot_one,
            ),
        ),
    )
    first_cases = (
        FullCaseMatrixCaseSpec(
            case_id="01_grazing_role_plot_rep1",
            abm="grazing",
            evidence_mode="plot",
            prompt_variant="role",
            repetition=1,
        ),
    )
    first = run_full_case_matrix_smoke(
        case_input=case_input,
        adapter=_SerialFakeAdapter(),
        model="qwen/qwen3.5-27b",
        output_root=tmp_path / "out",
        cases=first_cases,
        max_tokens=128,
        resume_existing=True,
    )
    assert first.success is True

    resumed = run_full_case_matrix_smoke(
        case_input=case_input,
        adapter=_SerialFakeAdapter(),
        model="qwen/qwen3.5-27b",
        output_root=tmp_path / "out",
        cases=(
            first_cases[0],
            FullCaseMatrixCaseSpec(
                case_id="02_grazing_role_table_rep1",
                abm="grazing",
                evidence_mode="table",
                prompt_variant="role",
                repetition=1,
            ),
        ),
        max_tokens=128,
        resume_existing=True,
    )

    assert resumed.observability_csv_path is not None
    assert resumed.observability_summary_json_path is not None
    rows = list(csv.DictReader(resumed.observability_csv_path.open(encoding="utf-8")))
    assert len(rows) == 4
    context_rows = {row["case_id"]: row for row in rows if row["request_kind"] == "context"}
    assert context_rows["01_grazing_role_plot_rep1"]["context_materialization_source"] == "resumed_previous_run"
    assert context_rows["01_grazing_role_plot_rep1"]["counts_toward_run_totals"] == "false"
    assert context_rows["02_grazing_role_table_rep1"]["context_materialization_source"] == "shared_context_cache"
    assert context_rows["02_grazing_role_table_rep1"]["reused_from_shared_context_cache"] == "true"
    assert context_rows["02_grazing_role_table_rep1"]["counts_toward_run_totals"] == "false"
    summary = json.loads(resumed.observability_summary_json_path.read_text(encoding="utf-8"))
    assert summary["observed_row_count"] == 4
    assert summary["request_count"] == 1
    assert summary["reused_request_count"] == 3
    assert summary["request_counts_by_kind"] == {"trend": 1}
    assert summary["usage_totals"] == {"completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2}


def test_run_full_case_matrix_smoke_parallel_resumed_case_blocks_new_context_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "simulation.csv"
    csv_path.write_text("tick;metric one\n0;1\n1;3\n", encoding="utf-8")
    parameters_path = tmp_path / "parameters.txt"
    parameters_path.write_text("parameter narrative", encoding="utf-8")
    documentation_path = tmp_path / "documentation.txt"
    documentation_path.write_text("documentation body", encoding="utf-8")
    plot_one = tmp_path / "1.png"
    plot_one.write_bytes(b"plot-one")
    case_input = FullCaseSmokeInput(
        abm="grazing",
        csv_path=csv_path,
        parameters_path=parameters_path,
        documentation_path=documentation_path,
        plots=(
            FullCasePlotInput(
                plot_index=1,
                reporter_pattern="metric one",
                plot_description="This plot represents herd size.",
                plot_path=plot_one,
            ),
        ),
    )
    first_run = run_full_case_matrix_smoke(
        case_input=case_input,
        adapter=_SerialFakeAdapter(),
        model="qwen/qwen3.5-27b",
        output_root=tmp_path / "out",
        cases=(
            FullCaseMatrixCaseSpec(
                case_id="01_grazing_role_plot_rep1",
                abm="grazing",
                evidence_mode="plot",
                prompt_variant="role",
                repetition=1,
            ),
        ),
        max_tokens=128,
        resume_existing=True,
    )
    assert first_run.success is True

    original_is_context_accepted = full_case_smoke._is_context_accepted

    def delayed_is_context_accepted(
        *,
        context_dir: Path,
        validation_state: full_case_smoke.FullCaseValidationState,
    ) -> bool:
        is_accepted = original_is_context_accepted(context_dir=context_dir, validation_state=validation_state)
        if is_accepted and (context_dir / "context_output.txt").exists():
            time.sleep(0.2)
        return is_accepted

    monkeypatch.setattr(full_case_smoke, "_is_context_accepted", delayed_is_context_accepted)

    adapter = _ConcurrentAdapter()
    resumed = run_full_case_matrix_smoke(
        case_input=case_input,
        adapter=adapter,
        model="qwen/qwen3.5-27b",
        output_root=tmp_path / "out",
        cases=(
            FullCaseMatrixCaseSpec(
                case_id="02_grazing_role_plot_rep1",
                abm="grazing",
                evidence_mode="plot",
                prompt_variant="role",
                repetition=1,
            ),
            FullCaseMatrixCaseSpec(
                case_id="01_grazing_role_plot_rep1",
                abm="grazing",
                evidence_mode="plot",
                prompt_variant="role",
                repetition=1,
            ),
        ),
        max_tokens=128,
        resume_existing=True,
    )
    assert resumed.success is True
    assert adapter._calls == 1

    assert resumed.observability_csv_path is not None
    assert resumed.observability_summary_json_path is not None
    rows = list(csv.DictReader(resumed.observability_csv_path.open(encoding="utf-8")))
    context_rows = {row["case_id"]: row for row in rows if row["request_kind"] == "context"}
    assert context_rows["01_grazing_role_plot_rep1"]["context_materialization_source"] == "resumed_previous_run"
    assert context_rows["02_grazing_role_plot_rep1"]["context_materialization_source"] == "shared_context_cache"
    summary = json.loads(resumed.observability_summary_json_path.read_text(encoding="utf-8"))
    assert summary["request_count"] == 1
    assert summary["request_counts_by_kind"] == {"trend": 1}


def test_run_full_case_matrix_smoke_retries_failed_case_in_same_invocation(tmp_path: Path) -> None:
    csv_path = tmp_path / "simulation.csv"
    csv_path.write_text("tick;metric one\n0;1\n1;3\n", encoding="utf-8")
    parameters_path = tmp_path / "parameters.txt"
    parameters_path.write_text("parameter narrative", encoding="utf-8")
    documentation_path = tmp_path / "documentation.txt"
    documentation_path.write_text("documentation body", encoding="utf-8")
    plot_one = tmp_path / "1.png"
    plot_one.write_bytes(b"plot-one")
    case_input = FullCaseSmokeInput(
        abm="grazing",
        csv_path=csv_path,
        parameters_path=parameters_path,
        documentation_path=documentation_path,
        plots=(
            FullCasePlotInput(
                plot_index=1,
                reporter_pattern="metric one",
                plot_description="This plot represents herd size.",
                plot_path=plot_one,
            ),
        ),
    )
    adapter = _FlakyTrendAdapter()

    result = run_full_case_matrix_smoke(
        case_input=case_input,
        adapter=adapter,
        model="nvidia/nemotron-nano-12b-v2-vl:free",
        output_root=tmp_path / "out",
        cases=(
            FullCaseMatrixCaseSpec(
                case_id="01_grazing_none_plot_rep1",
                abm="grazing",
                evidence_mode="plot",
                prompt_variant="none",
                repetition=1,
            ),
        ),
        max_tokens=128,
        resume_existing=True,
    )

    assert result.success is True
    assert adapter._calls == 3
    assert result.failed_case_ids == []
    trend_output = (result.cases[0].case_dir / "03_trends" / "plot_01" / "trend_output.txt").read_text(encoding="utf-8")
    assert trend_output == "response-3"


def test_run_full_case_matrix_smoke_records_prompt_compression_artifacts(tmp_path: Path) -> None:
    csv_path = tmp_path / "simulation.csv"
    csv_path.write_text("tick;metric one\n0;1\n1;3\n", encoding="utf-8")
    parameters_path = tmp_path / "parameters.txt"
    parameters_path.write_text("parameter narrative", encoding="utf-8")
    documentation_path = tmp_path / "documentation.txt"
    documentation_path.write_text("documentation body", encoding="utf-8")
    plot_one = tmp_path / "1.png"
    plot_one.write_bytes(b"plot-one")
    case_input = FullCaseSmokeInput(
        abm="grazing",
        csv_path=csv_path,
        parameters_path=parameters_path,
        documentation_path=documentation_path,
        plots=(
            FullCasePlotInput(
                plot_index=1,
                reporter_pattern="metric one",
                plot_description="This plot represents herd size.",
                plot_path=plot_one,
            ),
        ),
    )

    result = run_full_case_matrix_smoke(
        case_input=case_input,
        adapter=_ContextOverflowThenSuccessAdapter(),
        model="nvidia/nemotron-nano-12b-v2-vl:free",
        output_root=tmp_path / "out",
        cases=(
            FullCaseMatrixCaseSpec(
                case_id="01_grazing_role_table_rep1",
                abm="grazing",
                evidence_mode="table",
                prompt_variant="role",
                repetition=1,
            ),
        ),
        max_tokens=128,
        resume_existing=True,
    )

    assert result.success is True
    trend_dir = result.cases[0].case_dir / "03_trends" / "plot_01"
    compression_payload = json.loads((trend_dir / "trend_prompt_compression.json").read_text(encoding="utf-8"))
    assert compression_payload["triggered"] is True
    assert compression_payload["compression_count"] == 1
    assert compression_payload["attempt_count"] == 2
    assert compression_payload["attempts"][0]["table_downsample_stride"] == 1
    assert compression_payload["attempts"][1]["table_downsample_stride"] == 2
    original_prompt = (trend_dir / "trend_prompt_pre_compression.txt").read_text(encoding="utf-8")
    compressed_prompt = (trend_dir / "trend_prompt_compressed.txt").read_text(encoding="utf-8")
    final_prompt = (trend_dir / "trend_prompt.txt").read_text(encoding="utf-8")
    assert original_prompt != compressed_prompt
    assert compressed_prompt == final_prompt
    run_summary = json.loads(result.prompt_compression_summary_path.read_text(encoding="utf-8"))
    assert run_summary["total_entries"] == 1
    assert run_summary["triggered_entries"] == 1
    assert run_summary["total_compressions"] == 1
    assert run_summary["entries"][0]["scope"] == "full_case_matrix_plot"


def test_run_full_case_matrix_smoke_executes_case_trends_concurrently(tmp_path: Path) -> None:
    csv_path = tmp_path / "simulation.csv"
    csv_path.write_text("tick;metric one;metric two\n0;1;2\n1;3;4\n", encoding="utf-8")
    parameters_path = tmp_path / "parameters.txt"
    parameters_path.write_text("parameter narrative", encoding="utf-8")
    documentation_path = tmp_path / "documentation.txt"
    documentation_path.write_text("documentation body", encoding="utf-8")
    plot_one = tmp_path / "1.png"
    plot_one.write_bytes(b"plot-one")
    plot_two = tmp_path / "2.png"
    plot_two.write_bytes(b"plot-two")
    plot_three = tmp_path / "3.png"
    plot_three.write_bytes(b"plot-three")
    adapter = _ConcurrentAdapter()

    result = run_full_case_matrix_smoke(
        case_input=FullCaseSmokeInput(
            abm="grazing",
            csv_path=csv_path,
            parameters_path=parameters_path,
            documentation_path=documentation_path,
            plots=(
                FullCasePlotInput(
                    plot_index=1,
                    reporter_pattern="metric one",
                    plot_description="First plot",
                    plot_path=plot_one,
                ),
                FullCasePlotInput(
                    plot_index=2,
                    reporter_pattern="metric two",
                    plot_description="Second plot",
                    plot_path=plot_two,
                ),
                FullCasePlotInput(
                    plot_index=3,
                    reporter_pattern="metric one",
                    plot_description="Third plot",
                    plot_path=plot_three,
                ),
            ),
        ),
        adapter=adapter,
        model="nvidia/nemotron-nano-12b-v2-vl:free",
        output_root=tmp_path / "out",
        cases=(
            FullCaseMatrixCaseSpec(
                case_id="01_grazing_none_plot_rep1",
                abm="grazing",
                evidence_mode="plot",
                prompt_variant="none",
                repetition=1,
            ),
        ),
        max_tokens=128,
        resume_existing=True,
    )

    assert result.success is True
    assert adapter.max_active >= 2


def test_run_full_case_matrix_smoke_executes_cases_concurrently(tmp_path: Path) -> None:
    csv_path = tmp_path / "simulation.csv"
    csv_path.write_text("tick;metric one;metric two\n0;1;2\n1;3;4\n", encoding="utf-8")
    parameters_path = tmp_path / "parameters.txt"
    parameters_path.write_text("parameter narrative", encoding="utf-8")
    documentation_path = tmp_path / "documentation.txt"
    documentation_path.write_text("documentation body", encoding="utf-8")
    plot_one = tmp_path / "1.png"
    plot_one.write_bytes(b"plot-one")
    plot_two = tmp_path / "2.png"
    plot_two.write_bytes(b"plot-two")
    adapter = _ConcurrentAdapter()

    result = run_full_case_matrix_smoke(
        case_input=FullCaseSmokeInput(
            abm="grazing",
            csv_path=csv_path,
            parameters_path=parameters_path,
            documentation_path=documentation_path,
            plots=(
                FullCasePlotInput(
                    plot_index=1,
                    reporter_pattern="metric one",
                    plot_description="First plot",
                    plot_path=plot_one,
                ),
            ),
        ),
        adapter=adapter,
        model="nvidia/nemotron-nano-12b-v2-vl:free",
        output_root=tmp_path / "out",
        cases=(
            FullCaseMatrixCaseSpec(
                case_id="01_grazing_none_plot_rep1",
                abm="grazing",
                evidence_mode="plot",
                prompt_variant="none",
                repetition=1,
            ),
            FullCaseMatrixCaseSpec(
                case_id="02_grazing_role_plot_rep1",
                abm="grazing",
                evidence_mode="plot",
                prompt_variant="role",
                repetition=1,
            ),
        ),
        max_tokens=128,
        resume_existing=True,
    )

    assert result.success is True
    assert adapter.max_active >= 2


def test_compute_matrix_retry_wait_seconds_uses_transient_failures() -> None:
    wait_seconds = compute_matrix_retry_wait_seconds(
        [
            FullCaseMatrixCaseResult(
                case_id="01",
                abm="grazing",
                evidence_mode="plot",
                prompt_variant="none",
                repetition=1,
                case_dir=Path("/tmp/case"),
                success=False,
                error="circuit open for openrouter:model; retry after 59.0s",
            )
        ]
    )

    assert wait_seconds == DEFAULT_MATRIX_PASS_WAIT_SECONDS
