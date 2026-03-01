from __future__ import annotations

import json
from pathlib import Path

from distill_abm.ingest.netlogo import (
    clean_json_content,
    extract_code,
    extract_documentation,
    extract_parameters,
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


def test_clean_json_content(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    out = tmp_path / "doc.txt"
    source.write_text(
        json.dumps({"documentation": "## HEADER\n\nA\n## ANOTHER\n\nB"}),
        encoding="utf-8",
    )
    clean_json_content(source, out)
    assert out.read_text(encoding="utf-8") == "A\nB"
