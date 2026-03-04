from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import distill_abm.compat.compat_io as compat_io


def test_append_to_csv_and_append_to_csv2_round_trip_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "bundle"

    compat_io.append_to_csv(csv_path, "combo", "context prompt", "context response", ["trend-a", "trend-b"])
    compat_io.append_to_csv2(
        csv_path,
        "combo2",
        "context prompt 2",
        "context response 2",
        ["trend prompt"],
        ["trend response", "extra response"],
    )
    compat_io.append_analysis_to_csv(csv_path, ["analysis-a", "analysis-b"])

    rows = compat_io.load_existing_rows(csv_path)
    assert rows == [
        ["combo", "context prompt", "context response", "trend-a", "trend-b"],
        ["combo2", "context prompt 2", "context response 2", "trend prompt", "trend response"],
        ["analysis-a", "analysis-b"],
    ]


def test_load_existing_rows_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert compat_io.load_existing_rows(tmp_path / "missing.csv") == []


def test_normalize_output_csv_keeps_csv_and_appends_missing_suffix() -> None:
    assert str(compat_io._normalize_output_csv("report")) == "report.csv"
    assert str(compat_io._normalize_output_csv("report.csv")) == "report.csv"


def test_process_documentation_removes_urls_and_default_sections(tmp_path: Path) -> None:
    payload = {
        "documentation": (
            "WHAT IS IT?\n\n"
            "(a general understanding of what the model is trying to show or explain)\n"
            "## HOW IT WORKS\n\n"
            "This model tracks agent energy."
        ),
        "nested": {"url": "https://example.com/keep-me-off"},
    }
    input_json = tmp_path / "input.json"
    output_json = tmp_path / "output.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    compat_io.process_documentation(input_json, output_json)
    cleaned = json.loads(output_json.read_text(encoding="utf-8"))

    assert "WHAT IS IT" not in cleaned["documentation"]
    assert "This model tracks agent energy." in cleaned["documentation"]
    assert "https://example.com/keep-me-off" not in cleaned["nested"]["url"]


def test_create_collage_falls_back_when_pil_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.setitem(sys.modules, "PIL.Image", None)

    output_path = tmp_path / "fallback.txt"
    result = compat_io.create_collage([tmp_path / "missing.png"], output_path)

    assert result == output_path
    assert output_path.read_text(encoding="utf-8") == "PIL not available"
