from pathlib import Path

from distill_abm.cli_support import resolve_additional_scoring_reference_paths


def test_resolve_additional_scoring_reference_paths_returns_expected_reference_families() -> None:
    grazing_references = resolve_additional_scoring_reference_paths("grazing")
    assert set(grazing_references) == {"gpt5.2_short", "gpt5.2_long"}
    assert grazing_references["gpt5.2_short"] == Path("configs/ground_truth/grazing_gpt5.2_short_ground_truth.txt")
    assert grazing_references["gpt5.2_long"] == Path("configs/ground_truth/grazing_gpt5.2_long_ground_truth.txt")

    milk_references = resolve_additional_scoring_reference_paths("milk_consumption")
    assert set(milk_references) == {"modeler", "gpt5.2_short", "gpt5.2_long"}
    assert milk_references["modeler"] == Path("configs/ground_truth/milk_modeler_ground_truth.txt")
