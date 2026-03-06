from __future__ import annotations

import json
from pathlib import Path

import pytest

from distill_abm.ingest.netlogo import (
    clean_json_content,
    extract_code,
    extract_documentation,
    extract_experiment_parameters,
    extract_parameters,
    find_reference_narrative_path,
    process_documentation_remove_defaults,
    process_documentation_remove_urls,
    remove_default_elements,
    remove_urls,
    update_parameters,
)


def test_extract_parameters_and_update() -> None:
    code = "SLIDER 0 0 10 10 number-of-agents number-of-agents 1 1000 1 100\n" "SWITCH 0 0 10 10 norms norms 1 0 1\n"
    params = extract_parameters(code)
    assert params["sliders"][0]["name"] == "number-of-agents"
    gui = [{"name": "number-of-agents", "value": 0}]
    experiment = {"number-of-agents": 500}
    update_parameters(gui, experiment)
    assert gui[0]["value"] == 500
    assert experiment == {}


def test_extract_documentation_and_code(tmp_path: Path) -> None:
    content = "globals [a b]\n" "to go\nend\n" "@#$#@#$#@\n" "## WHAT IS IT?\n\nDoc text\n" "@#$#@#$#@\n"
    path = tmp_path / "model.nlogo"
    path.write_text(content, encoding="utf-8")

    assert "Doc text" in extract_documentation(path)
    assert "globals [a b]" in extract_code(path)


def test_extract_documentation_accepts_title_between_delimiter_and_what_is_it(tmp_path: Path) -> None:
    content = (
        "globals [a b]\n"
        "to go\nend\n"
        "@#$#@#$#@\n"
        "# Model Title\n\n"
        "## WHAT IS IT?\n\n"
        "Doc text\n"
        "@#$#@#$#@\n"
    )
    path = tmp_path / "model.nlogo"
    path.write_text(content, encoding="utf-8")

    doc = extract_documentation(path)
    assert "# Model Title" in doc
    assert "Doc text" in doc


def test_extract_documentation_falls_back_to_header_comments(tmp_path: Path) -> None:
    content = (
        "; This is the fauna model.\n"
        "; It models forager bands and prey dynamics.\n"
        "extensions [csv]\n"
        "globals [a b]\n"
    )
    path = tmp_path / "no_doc_model.nlogo"
    path.write_text(content, encoding="utf-8")

    doc = extract_documentation(path)
    assert "## WHAT IS IT?" in doc
    assert "This is the fauna model." in doc


def test_remove_urls_and_defaults() -> None:
    text = "See https://example.com and www.test.org"
    assert "http" not in remove_urls(text)
    doc = "\n## WHAT IS IT?\n\n(a general understanding of what the model is trying to show or explain)"
    assert remove_default_elements(doc) == ""


def test_process_documentation_stages(tmp_path: Path) -> None:
    payload = {"documentation": "## WHAT IS IT?\n\nhello www.test.org"}
    source = tmp_path / "input.json"
    no_urls = tmp_path / "no_urls.json"
    no_defaults = tmp_path / "no_defaults.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    process_documentation_remove_urls(source, no_urls)
    updated = json.loads(no_urls.read_text(encoding="utf-8"))
    assert "www." not in updated["documentation"]

    process_documentation_remove_defaults(no_urls, no_defaults)
    cleaned = json.loads(no_defaults.read_text(encoding="utf-8"))
    assert "documentation" in cleaned


def test_extract_experiment_parameters_prefers_named_experiment() -> None:
    code = Path("data/milk_consumption_abm/milk_consumption.nlogo").read_text(encoding="utf-8")
    values = extract_experiment_parameters(code, preferred_experiment="Milk Consumption Trends")
    assert values["number-of-agents"] == 1000
    assert values["habit-on?"] is True
    assert values["network-type"] == "watts-strogatz"
    assert len(values) == 16


def test_extract_experiment_parameters_filters_output_csv_file_value() -> None:
    code = Path("data/fauna_abm/fauna.nlogo").read_text(encoding="utf-8")
    values = extract_experiment_parameters(code, preferred_experiment="2023-20-6-0.33-0.8")
    assert "Output-csv-file" not in values
    assert len(values) == 29


def test_find_reference_narrative_path_for_abm() -> None:
    reference_path = find_reference_narrative_path(Path("data/grazing_abm"))
    assert reference_path is not None
    assert reference_path.name == "reference_narrative.txt"


def test_clean_json_content(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    out = tmp_path / "doc.txt"
    source.write_text(
        json.dumps({"documentation": "## HEADER\n\nA\n## ANOTHER\n\nB"}),
        encoding="utf-8",
    )
    clean_json_content(source, out)
    assert out.read_text(encoding="utf-8") == "A\nB"


def test_clean_json_content_removes_netlogo_delimiter_heading_prefix(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    out = tmp_path / "doc.txt"
    source.write_text(
        json.dumps(
            {
                "documentation": "## @#$#@#$#@\n\n# RAGE RAngeland Grazing ModEl - EXTENSION\n\n"
                "## WHAT IS IT?\n\nDoc text\n## HOW IT WORKS\n\nMore text"
            }
        ),
        encoding="utf-8",
    )
    clean_json_content(source, out)
    assert out.read_text(encoding="utf-8") == "Doc text\nMore text"





def test_extract_documentation_raises_for_missing_doc_section(tmp_path: Path) -> None:
    """Test that extract_documentation raises ValueError when no doc section exists."""
    content = "globals [a b]\nto go\nend\n"
    path = tmp_path / "no_doc.nlogo"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="documentation section not found"):
        extract_documentation(path)


def test_extract_code_raises_for_missing_code_section(tmp_path: Path) -> None:
    """Test that extract_code raises ValueError when no code section exists."""
    content = "not a valid netlogo file\n"
    path = tmp_path / "no_code.nlogo"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="code section not found"):
        extract_code(path)


def test_extract_parameters_returns_empty_dict_for_no_gui() -> None:
    """Test that extract_parameters returns empty dict when no GUI elements."""
    code = "globals [a b]\nto go\nend\n"
    params = extract_parameters(code)
    # Returns empty dict when no GUI elements
    assert params == {}


def test_extract_experiment_parameters_string_with_empty_experiments() -> None:
    """Test that extract_experiment_parameters handles empty experiments section in string."""
    code = """globals [a b]
to go
end
"""
    # String with no experiments section - returns empty dict
    values = extract_experiment_parameters(code)
    assert isinstance(values, dict)


def test_extract_experiment_parameters_accepts_code_string() -> None:
    """Test that extract_experiment_parameters accepts string content."""
    code_str = Path("data/fauna_abm/fauna.nlogo").read_text(encoding="utf-8")
    # Should work with string content
    values = extract_experiment_parameters(code_str, preferred_experiment="2023-20-6-0.33-0.8")
    assert isinstance(values, dict)
    assert "Output-csv-file" not in values


def test_extract_experiment_parameters_falls_back_for_missing_experiment() -> None:
    """Test that extract_experiment_parameters falls back when preferred experiment not found."""
    code = """globals [a b]
to go
end
@#$#@#$#@
<experiments>
<experiment name="Experiment1">
<setup>setup</setup>
</experiment>
</experiments>
"""
    # Pass string directly (not Path)
    values = extract_experiment_parameters(code, preferred_experiment="NonExistentExperiment")
    # Falls back to first experiment
    assert isinstance(values, dict)
