from __future__ import annotations

import csv
import json
from pathlib import Path

from distill_abm.pipeline.summarizer_smoke import ValidatedSmokeBundle, run_summarizer_smoke


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
