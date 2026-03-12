from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from distill_abm.pipeline.summarizer_smoke import (
    ValidatedSmokeBundle,
    default_validated_smoke_bundles,
    run_summarizer_smoke,
)


def _bundle(tmp_path: Path) -> ValidatedSmokeBundle:
    context_path = tmp_path / "context.txt"
    trend_one = tmp_path / "trend_01.txt"
    trend_two = tmp_path / "trend_02.txt"
    context_path.write_text("Valid context output.", encoding="utf-8")
    trend_one.write_text("Trend one output.", encoding="utf-8")
    trend_two.write_text("Trend two output.", encoding="utf-8")
    return ValidatedSmokeBundle(
        bundle_id="bundle_one",
        case_id="case_one",
        abm="grazing",
        context_output_path=context_path,
        trend_output_paths=(trend_one, trend_two),
        validation_note="Hand checked full case.",
    )


def test_run_summarizer_smoke_writes_bundle_outputs(tmp_path: Path) -> None:
    result = run_summarizer_smoke(
        source_root=tmp_path,
        output_root=tmp_path / "smoke",
        validated_bundles=(_bundle(tmp_path),),
        summarizer_fns={
            "bart": lambda text: f"bart::{text}",
            "bert": lambda text: f"bert::{text}",
            "t5": lambda text: f"t5::{text}",
            "longformer_ext": lambda text: f"longformer::{text}",
        },
    )

    assert result.success is True
    bundle_dir = result.bundles[0].bundle_dir
    assert result.run_root == result.report_json_path.parent
    assert result.run_log_path.exists() is True
    assert (tmp_path / "smoke" / "latest_run.txt").read_text(encoding="utf-8").strip() == str(result.run_root)
    combined_input = (bundle_dir / "01_input" / "combined_input.txt").read_text(encoding="utf-8")
    assert combined_input == "Valid context output.\n\nTrend one output.\n\nTrend two output."
    assert (bundle_dir / "02_summaries" / "none.txt").read_text(encoding="utf-8") == combined_input
    assert (bundle_dir / "02_summaries" / "bart.txt").read_text(encoding="utf-8").startswith("bart::")
    assert (bundle_dir / "03_metadata" / "raw_summaries" / "bart.txt").read_text(encoding="utf-8").startswith("bart::")
    rows = list(csv.DictReader(result.review_csv_path.open(encoding="utf-8")))
    assert len(rows) == 5
    assert {row["mode"] for row in rows} == {"none", "bart", "bert", "t5", "longformer_ext"}
    assert json.loads(result.validated_sources_path.read_text(encoding="utf-8"))[0]["bundle_id"] == "bundle_one"


def test_run_summarizer_smoke_cleans_repetition_and_keeps_raw_text(tmp_path: Path) -> None:
    repeated = "Trend repeats. Trend repeats."
    result = run_summarizer_smoke(
        source_root=tmp_path,
        output_root=tmp_path / "smoke",
        validated_bundles=(_bundle(tmp_path),),
        summarizer_fns={
            "bart": lambda _text: repeated,
            "bert": lambda text: f"bert::{text}",
            "t5": lambda text: f"t5::{text}",
            "longformer_ext": lambda text: f"longformer::{text}",
        },
    )

    bundle_dir = result.bundles[0].bundle_dir
    assert (bundle_dir / "02_summaries" / "bart.txt").read_text(encoding="utf-8") == "Trend repeats."
    assert (bundle_dir / "03_metadata" / "raw_summaries" / "bart.txt").read_text(encoding="utf-8") == repeated
    mode_results = json.loads((bundle_dir / "03_metadata" / "mode_results.json").read_text(encoding="utf-8"))
    bart_mode = next(mode for mode in mode_results["modes"] if mode["mode"] == "bart")
    assert bart_mode["postprocess_changed"] is True
    assert bart_mode["raw_output_path"].endswith("03_metadata/raw_summaries/bart.txt")


def test_run_summarizer_smoke_rejects_generic_unavailable_text(tmp_path: Path) -> None:
    bad_bundle = _bundle(tmp_path)
    bad_bundle.context_output_path.write_text(
        "The analysis is currently unavailable. Please try again later.",
        encoding="utf-8",
    )

    result = run_summarizer_smoke(
        source_root=tmp_path,
        output_root=tmp_path / "smoke",
        validated_bundles=(bad_bundle,),
        summarizer_fns={
            "bart": lambda text: text,
            "bert": lambda text: text,
            "t5": lambda text: text,
            "longformer_ext": lambda text: text,
        },
    )

    assert result.success is False
    assert result.failed_bundle_ids == ["bundle_one"]
    assert result.bundles[0].source_validation_error is not None


def test_default_validated_smoke_bundles_discovers_successful_matrix_cases(tmp_path: Path) -> None:
    run_root = tmp_path / "matrix_run"
    cases_root = run_root / "cases"
    success_case = cases_root / "01_grazing_none_plot_rep1"
    failed_case = cases_root / "02_grazing_none_table_rep1"
    for case_root in (success_case, failed_case):
        (case_root / "02_context").mkdir(parents=True)
        (case_root / "03_trends" / "plot_01").mkdir(parents=True)
        (case_root / "02_context" / "context_output.txt").write_text("Valid context.", encoding="utf-8")
        (case_root / "03_trends" / "plot_01" / "trend_output.txt").write_text("Valid trend.", encoding="utf-8")
    report = {
        "cases": [
            {
                "case_id": success_case.name,
                "case_dir": str(success_case),
                "abm": "grazing",
                "success": True,
            },
            {
                "case_id": failed_case.name,
                "case_dir": str(failed_case),
                "abm": "grazing",
                "success": False,
            },
        ]
    }
    (run_root / "smoke_full_case_matrix_report.json").write_text(json.dumps(report), encoding="utf-8")

    bundles = default_validated_smoke_bundles(run_root)

    assert [bundle.case_id for bundle in bundles] == [success_case.name]
    assert bundles[0].context_output_path == success_case / "02_context" / "context_output.txt"
    assert bundles[0].trend_output_paths == (success_case / "03_trends" / "plot_01" / "trend_output.txt",)


def test_run_summarizer_smoke_clears_stale_output_root(tmp_path: Path) -> None:
    output_root = tmp_path / "smoke"
    stale_file = output_root / "stale.txt"
    output_root.mkdir()
    stale_file.write_text("stale", encoding="utf-8")

    result = run_summarizer_smoke(
        source_root=tmp_path,
        output_root=output_root,
        validated_bundles=(_bundle(tmp_path),),
        summarizer_fns={
            "bart": lambda text: f"bart::{text}",
            "bert": lambda text: f"bert::{text}",
            "t5": lambda text: f"t5::{text}",
            "longformer_ext": lambda text: f"longformer::{text}",
        },
    )

    assert result.success is True
    assert stale_file.exists() is False


def test_run_summarizer_smoke_reuses_successful_outputs_when_resuming(tmp_path: Path) -> None:
    output_root = tmp_path / "smoke"
    bundle = _bundle(tmp_path)

    first = run_summarizer_smoke(
        source_root=tmp_path,
        output_root=output_root,
        validated_bundles=(bundle,),
        summarizer_fns={
            "bart": lambda _text: "bart::clean summary",
            "bert": lambda _text: "bert::clean summary",
            "t5": lambda _text: "t5::clean summary",
            "longformer_ext": lambda _text: "longformer::clean summary",
        },
    )

    assert first.success is True

    def _should_not_run(_: str) -> str:
        raise AssertionError("summarizer should not rerun when resume=True and output is already valid")

    resumed = run_summarizer_smoke(
        source_root=tmp_path,
        output_root=output_root,
        resume=True,
        validated_bundles=(bundle,),
        summarizer_fns={
            "bart": _should_not_run,
            "bert": _should_not_run,
            "t5": _should_not_run,
            "longformer_ext": _should_not_run,
        },
    )

    assert resumed.success is True
    assert resumed.run_root != first.run_root
    assert all(mode.success for mode in resumed.bundles[0].modes)
    assert {mode.duration_seconds for mode in resumed.bundles[0].modes} == {0.0}


def test_run_summarizer_smoke_resume_preserves_false_postprocess_changed_for_unchanged_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "smoke"
    bundle = _bundle(tmp_path)

    first = run_summarizer_smoke(
        source_root=tmp_path,
        output_root=output_root,
        validated_bundles=(bundle,),
        summarizer_fns={
            "bart": lambda _text: "bart::clean summary",
            "bert": lambda _text: "bert::clean summary",
            "t5": lambda _text: "t5::clean summary",
            "longformer_ext": lambda _text: "longformer::clean summary",
        },
    )
    assert first.success is True

    resumed = run_summarizer_smoke(
        source_root=tmp_path,
        output_root=output_root,
        resume=True,
        validated_bundles=(bundle,),
        summarizer_fns={
            "bart": lambda _text: (_ for _ in ()).throw(AssertionError("should not rerun")),
            "bert": lambda _text: (_ for _ in ()).throw(AssertionError("should not rerun")),
            "t5": lambda _text: (_ for _ in ()).throw(AssertionError("should not rerun")),
            "longformer_ext": lambda _text: (_ for _ in ()).throw(AssertionError("should not rerun")),
        },
    )

    bart_mode = next(mode for mode in resumed.bundles[0].modes if mode.mode == "bart")
    assert bart_mode.raw_output_path is not None
    assert bart_mode.postprocess_changed is False


def test_default_validated_smoke_bundles_discovers_completed_suite_abms(tmp_path: Path) -> None:
    suite_root = tmp_path / "suite"
    fauna_run = suite_root / "abms" / "fauna" / "runs" / "run_1"
    grazing_run = suite_root / "abms" / "grazing" / "runs" / "run_1"
    for abm, run_root in [("fauna", fauna_run), ("grazing", grazing_run)]:
        (suite_root / "abms" / abm).mkdir(parents=True, exist_ok=True)
        (suite_root / "abms" / abm / "latest_run.txt").write_text(str(run_root), encoding="utf-8")
        case_root = run_root / "cases" / f"01_{abm}_none_plot_rep1"
        (case_root / "02_context").mkdir(parents=True)
        (case_root / "03_trends" / "plot_01").mkdir(parents=True)
        (case_root / "02_context" / "context_output.txt").write_text("Valid context.", encoding="utf-8")
        (case_root / "03_trends" / "plot_01" / "trend_output.txt").write_text("Valid trend.", encoding="utf-8")
        (run_root / "smoke_full_case_matrix_report.json").write_text(
            json.dumps(
                {
                    "cases": [
                        {
                            "case_id": f"01_{abm}_none_plot_rep1",
                            "case_dir": str(case_root),
                            "abm": abm,
                            "success": True,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    bundles = default_validated_smoke_bundles(suite_root, include_abms=("fauna",))

    assert [bundle.abm for bundle in bundles] == ["fauna"]
    assert bundles[0].case_id == "01_fauna_none_plot_rep1"


def test_default_validated_smoke_bundles_discovers_live_suite_cases_without_matrix_report(tmp_path: Path) -> None:
    suite_root = tmp_path / "suite"
    run_root = suite_root / "abms" / "fauna" / "runs" / "run_live"
    case_root = run_root / "cases" / "01_fauna_none_plot_rep1"
    (suite_root / "abms" / "fauna").mkdir(parents=True, exist_ok=True)
    (suite_root / "abms" / "fauna" / "latest_run.txt").write_text(str(run_root), encoding="utf-8")
    (case_root / "02_context").mkdir(parents=True)
    (case_root / "03_trends" / "plot_01").mkdir(parents=True)
    (case_root / "02_context" / "context_output.txt").write_text("Valid context.", encoding="utf-8")
    (case_root / "03_trends" / "plot_01" / "trend_output.txt").write_text("Valid trend.", encoding="utf-8")
    (case_root / "00_case_summary.json").write_text(
        json.dumps(
            {
                "case_id": "01_fauna_none_plot_rep1",
                "abm": "fauna",
                "evidence_mode": "plot",
                "prompt_variant": "none",
                "repetition": 1,
                "model": "mistral-medium-latest",
            }
        ),
        encoding="utf-8",
    )
    (case_root / "validation_state.json").write_text(
        json.dumps(
            {
                "context": {"status": "accepted", "error": None},
                "trends": {"1": {"status": "accepted", "error": None}},
            }
        ),
        encoding="utf-8",
    )

    bundles = default_validated_smoke_bundles(suite_root, include_abms=("fauna",))

    assert [bundle.case_id for bundle in bundles] == ["01_fauna_none_plot_rep1"]
    assert bundles[0].context_output_path == case_root / "02_context" / "context_output.txt"
    assert bundles[0].trend_output_paths == (case_root / "03_trends" / "plot_01" / "trend_output.txt",)


def test_default_validated_smoke_bundles_skips_live_case_with_retry_trend_missing_output(tmp_path: Path) -> None:
    suite_root = tmp_path / "suite"
    run_root = suite_root / "abms" / "fauna" / "runs" / "run_live"
    case_root = run_root / "cases" / "02_fauna_none_table_rep1"
    (suite_root / "abms" / "fauna").mkdir(parents=True, exist_ok=True)
    (suite_root / "abms" / "fauna" / "latest_run.txt").write_text(str(run_root), encoding="utf-8")
    (case_root / "02_context").mkdir(parents=True)
    (case_root / "03_trends" / "plot_01").mkdir(parents=True)
    (case_root / "03_trends" / "plot_02").mkdir(parents=True)
    (case_root / "02_context" / "context_output.txt").write_text("Valid context.", encoding="utf-8")
    (case_root / "03_trends" / "plot_02" / "trend_output.txt").write_text("Valid trend 02.", encoding="utf-8")
    (case_root / "00_case_summary.json").write_text(
        json.dumps(
            {
                "case_id": "02_fauna_none_table_rep1",
                "abm": "fauna",
                "evidence_mode": "table",
                "prompt_variant": "none",
                "repetition": 1,
                "model": "qwen/qwen3.5-27b",
            }
        ),
        encoding="utf-8",
    )
    (case_root / "validation_state.json").write_text(
        json.dumps(
            {
                "context": {"status": "accepted", "error": None},
                "trends": {
                    "1": {"status": "retry", "error": "model did not return valid structured JSON"},
                    "2": {"status": "accepted", "error": None},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="missing completed ABM runs for: fauna"):
        default_validated_smoke_bundles(suite_root, include_abms=("fauna",))


def test_run_summarizer_smoke_watch_mode_processes_bundle_when_it_appears(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    source_root = tmp_path / "suite"
    output_root = tmp_path / "summaries"
    run_root = source_root / "abms" / "fauna" / "runs" / "run_live"
    abm_root = source_root / "abms" / "fauna"
    abm_root.mkdir(parents=True, exist_ok=True)
    (abm_root / "latest_run.txt").write_text(str(run_root), encoding="utf-8")

    state = {"bundle_created": False, "active": True}

    def _create_bundle() -> None:
        case_root = run_root / "cases" / "01_fauna_none_plot_rep1"
        (case_root / "02_context").mkdir(parents=True, exist_ok=True)
        (case_root / "03_trends" / "plot_01").mkdir(parents=True, exist_ok=True)
        (case_root / "02_context" / "context_output.txt").write_text("Valid context.", encoding="utf-8")
        (case_root / "03_trends" / "plot_01" / "trend_output.txt").write_text("Valid trend.", encoding="utf-8")
        (case_root / "00_case_summary.json").write_text(
            json.dumps(
                {
                    "case_id": "01_fauna_none_plot_rep1",
                    "abm": "fauna",
                    "evidence_mode": "plot",
                    "prompt_variant": "none",
                    "repetition": 1,
                    "model": "mistral-medium-latest",
                }
            ),
            encoding="utf-8",
        )
        (case_root / "validation_state.json").write_text(
            json.dumps(
                {
                    "context": {"status": "accepted", "error": None},
                    "trends": {"1": {"status": "accepted", "error": None}},
                }
            ),
            encoding="utf-8",
        )

    def fake_read_active_run_lock(_: Path) -> SimpleNamespace | None:
        if state["active"]:
            return SimpleNamespace(pid=123, run_id="run_live", run_root=run_root)
        return None

    def fake_sleep(_: float) -> None:
        if not state["bundle_created"]:
            _create_bundle()
            state["bundle_created"] = True
            state["active"] = False

    monkeypatch.setattr("distill_abm.pipeline.summarizer_smoke.read_active_run_lock", fake_read_active_run_lock)
    monkeypatch.setattr("distill_abm.pipeline.summarizer_smoke.time.sleep", fake_sleep)

    result = run_summarizer_smoke(
        source_root=source_root,
        output_root=output_root,
        watch=True,
        poll_interval_seconds=0.01,
        summarizer_fns={
            "bart": lambda text: f"bart::{text}",
            "bert": lambda text: f"bert::{text}",
            "t5": lambda text: f"t5::{text}",
            "longformer_ext": lambda text: f"longformer::{text}",
        },
    )

    assert result.success is True
    assert state["bundle_created"] is True
    assert [bundle.case_id for bundle in result.bundles] == ["01_fauna_none_plot_rep1"]
    assert (result.bundles[0].bundle_dir / "02_summaries" / "bart.txt").read_text(encoding="utf-8").startswith("bart::")
