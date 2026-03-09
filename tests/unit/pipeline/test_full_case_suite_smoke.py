from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from distill_abm.llm.adapters.base import LLMAdapter
from distill_abm.pipeline.full_case_matrix_smoke import FullCaseMatrixCaseSpec
from distill_abm.pipeline.full_case_smoke import FullCaseSmokeInput
from distill_abm.pipeline.full_case_suite_smoke import run_full_case_suite_smoke


class _Adapter(LLMAdapter):
    provider = "mistral"

    def complete(self, request):  # type: ignore[no-untyped-def]
        _ = request
        raise AssertionError("not used in this test")


def test_run_full_case_suite_smoke_writes_outer_artifacts(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
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
    assert "Mistral Generation Dashboard" in result.review_html_path.read_text(encoding="utf-8")
    assert "Planned cases" in result.report_markdown_path.read_text(encoding="utf-8")


def test_run_full_case_suite_smoke_marks_abm_failure_without_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
