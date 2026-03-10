from __future__ import annotations

import csv
import json
from pathlib import Path

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
    rows = list(csv.DictReader(result.review_csv_path.open(encoding="utf-8")))
    assert len(rows) == 5
    assert {row["mode"] for row in rows} == {"none", "bart", "bert", "t5", "longformer_ext"}
    assert json.loads(result.validated_sources_path.read_text(encoding="utf-8"))[0]["bundle_id"] == "bundle_one"


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
            "bart": lambda text: f"bart::{text}",
            "bert": lambda text: f"bert::{text}",
            "t5": lambda text: f"t5::{text}",
            "longformer_ext": lambda text: f"longformer::{text}",
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
