from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner

import distill_abm.cli as cli_module
from distill_abm.cli import app

runner = CliRunner()


def _write_min_nlogo_model_dir(root: Path, abm: str, docs_text: str) -> None:
    model_dir = root / "abms" / abm
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / f"{abm}.nlogo").write_text(
        "\n".join(
            [
                "globals []",
                "",
                "@#$#@#$#@",
                "NetLogo 6.4.0",
                "@#$#@#$#@",
                "## WHAT IS IT?",
                docs_text,
            ]
        ),
        encoding="utf-8",
    )


def test_cli_quality_gate_uses_scope_defaults(tmp_path: Path, monkeypatch: Any) -> None:
    model_root = tmp_path / "data"
    _write_min_nlogo_model_dir(model_root, "fauna", "Fauna doc")
    _write_min_nlogo_model_dir(model_root, "grazing", "Grazing doc")
    _write_min_nlogo_model_dir(model_root, "milk_consumption", "Milk doc")

    captured: dict[str, Any] = {}

    def fake_run_validation_suite(*, output_root, abm_models, checks, ingest_stage_ids, profile):  # type: ignore[no-untyped-def]
        captured["checks"] = checks
        captured["profile"] = profile
        return SimpleNamespace(
            success=True,
            failed_checks=[],
            ingest_smoke_report_json_path=None,
            ingest_smoke_report_markdown_path=None,
            report_json_path=Path("validation_report.json"),
            report_markdown_path=Path("validation_report.md"),
            model_dump_json=lambda indent=2: "{}",
        )

    monkeypatch.setattr(cli_module, "run_validation_suite", fake_run_validation_suite)

    result = runner.invoke(
        app,
        [
            "quality-gate",
            "--models-root",
            str(model_root),
            "--scope",
            "static",
        ],
    )

    assert result.exit_code == 0
    assert captured["profile"] == "quick"
    assert captured["checks"] == ["ruff", "mypy"]


def test_cli_quality_gate_allows_explicit_checks(tmp_path: Path, monkeypatch: Any) -> None:
    model_root = tmp_path / "data"
    _write_min_nlogo_model_dir(model_root, "fauna", "Fauna doc")
    _write_min_nlogo_model_dir(model_root, "grazing", "Grazing doc")
    _write_min_nlogo_model_dir(model_root, "milk_consumption", "Milk doc")

    captured: dict[str, Any] = {}

    def fake_run_validation_suite(*, output_root, abm_models, checks, ingest_stage_ids, profile):  # type: ignore[no-untyped-def]
        captured["checks"] = checks
        captured["profile"] = profile
        return SimpleNamespace(
            success=True,
            failed_checks=[],
            ingest_smoke_report_json_path=None,
            ingest_smoke_report_markdown_path=None,
            report_json_path=Path("validation_report.json"),
            report_markdown_path=Path("validation_report.md"),
            model_dump_json=lambda indent=2: "{}",
        )

    monkeypatch.setattr(cli_module, "run_validation_suite", fake_run_validation_suite)

    result = runner.invoke(
        app,
        [
            "quality-gate",
            "--models-root",
            str(model_root),
            "--scope",
            "pre-llm",
            "--check",
            "pytest",
        ],
    )

    assert result.exit_code == 0
    assert captured["profile"] == "quick"
    assert captured["checks"] == ["pytest"]
