from __future__ import annotations

import csv
import json
from pathlib import Path

from distill_abm.llm.adapters.base import LLMAdapter, LLMResponse
from distill_abm.pipeline.full_case_matrix_smoke import (
    FullCaseMatrixCaseSpec,
    build_full_case_matrix_case_specs,
    run_full_case_matrix_smoke,
)
from distill_abm.pipeline.full_case_smoke import (
    FullCasePlotInput,
    FullCaseSmokeInput,
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
    assert (result.case_dir / "03_trends" / "plot_01" / "trend_output.txt").read_text(encoding="utf-8") == "response-2"
    assert (result.case_dir / "03_trends" / "plot_02" / "trend_output.txt").read_text(encoding="utf-8") == "response-3"
    rows = list(csv.DictReader(result.review_csv_path.open(encoding="utf-8")))
    assert len(rows) == 3
    assert {row["plot_index"] for row in rows} == {"context", "1", "2"}
    row_by_plot = {row["plot_index"]: row for row in rows}
    assert row_by_plot["context"]["validation_status"] == "accepted"
    assert row_by_plot["1"]["validation_status"] == "accepted"
    assert row_by_plot["2"]["validation_status"] == "accepted"


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
    assert len(result.cases) == 2
    assert adapter._calls == 3
    context_outputs = [
        (case.case_dir / "02_context" / "context_output.txt").read_text(encoding="utf-8")
        for case in result.cases
    ]
    assert context_outputs == ["response-1", "response-1"]
    assert result.cases[0].resumed_from_existing is False
    assert result.cases[1].resumed_from_existing is False
