from pathlib import Path

from distill_abm.cli_support import (
    resolve_additional_scoring_reference_paths,
    resolve_quantitative_reference_kind,
)


def test_resolve_additional_scoring_reference_paths_returns_expected_reference_families() -> None:
    grazing_references = resolve_additional_scoring_reference_paths("grazing")
    assert set(grazing_references) == {"gpt5.2_short", "gpt5.2_long"}
    assert grazing_references["gpt5.2_short"] == Path("data/summaries/gpt5.2/grazing_gpt5.2_short_ground_truth.txt")
    assert grazing_references["gpt5.2_long"] == Path("data/summaries/gpt5.2/grazing_gpt5.2_long_ground_truth.txt")

    fauna_references = resolve_additional_scoring_reference_paths("fauna")
    assert set(fauna_references) == {"modeler", "gpt5.2_short", "gpt5.2_long"}
    assert fauna_references["modeler"] == Path("data/summaries/modelers/fauna_modeler_ground_truth.txt")

    milk_references = resolve_additional_scoring_reference_paths("milk_consumption")
    assert set(milk_references) == {"modeler", "gpt5.2_short", "gpt5.2_long"}
    assert milk_references["modeler"] == Path("data/summaries/modelers/milk_modeler_ground_truth.txt")


def test_resolve_quantitative_reference_kind_classifies_summary_and_full_report_references() -> None:
    assert resolve_quantitative_reference_kind("author") == "summary"
    assert resolve_quantitative_reference_kind("modeler") == "summary"
    assert resolve_quantitative_reference_kind("gpt5.2_short") == "summary"
    assert resolve_quantitative_reference_kind("gpt5.2_long") == "full_report"
