from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner

import distill_abm.cli as cli_module
from distill_abm.cli import app

runner = CliRunner()


def _write_model(root: Path, abm: str) -> None:
    model_dir = root / "abms" / abm
    model_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"{abm}.nlogo" if abm != "milk_consumption" else "model.nlogo"
    (model_dir / model_name).write_text(
        "globals [a b]\n" "to go\n" "end\n" "@#$#@#$#@\n" "## WHAT IS IT?\n\nDoc text\n" "@#$#@#$#@\n",
        encoding="utf-8",
    )


def test_cli_smoke_ingest_netlogo_forwards_stage_selection(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    model_root = tmp_path / "data"
    _write_model(model_root, "fauna")
    _write_model(model_root, "grazing")
    _write_model(model_root, "milk_consumption")
    captured: dict[str, Any] = {}

    def fake_run_ingest_smoke_suite(*, abm_models, output_root, stage_ids):  # type: ignore[no-untyped-def]
        captured["abm_models"] = abm_models
        captured["output_root"] = output_root
        captured["stage_ids"] = stage_ids
        return SimpleNamespace(
            success=True,
            failed_abms=[],
            report_markdown_path=Path("ingest_smoke.md"),
            report_json_path=Path("ingest_smoke.json"),
        )

    monkeypatch.setattr(cli_module, "run_ingest_smoke_suite", fake_run_ingest_smoke_suite)

    result = runner.invoke(
        app,
        [
            "smoke-ingest-netlogo",
            "--models-root",
            str(model_root),
            "--output-root",
            str(tmp_path / "ingest-smoke"),
            "--stage",
            "documentation",
            "--stage",
            "final-documentation",
        ],
    )

    assert result.exit_code == 0
    assert captured["stage_ids"] == ["documentation", "final-documentation"]
    assert sorted(captured["abm_models"]) == ["fauna", "grazing", "milk_consumption"]


def test_cli_smoke_ingest_netlogo_supports_json_and_require_stage(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    model_root = tmp_path / "data"
    _write_model(model_root, "fauna")
    _write_model(model_root, "grazing")
    _write_model(model_root, "milk_consumption")

    def fake_run_ingest_smoke_suite(*, abm_models, output_root, stage_ids):  # type: ignore[no-untyped-def]
        _ = abm_models, output_root, stage_ids
        return SimpleNamespace(
            success=True,
            failed_abms=[],
            selected_stage_ids=["documentation"],
            report_markdown_path=Path("ingest_smoke.md"),
            report_json_path=Path("ingest_smoke.json"),
        )

    monkeypatch.setattr(cli_module, "run_ingest_smoke_suite", fake_run_ingest_smoke_suite)

    result = runner.invoke(
        app,
        [
            "smoke-ingest-netlogo",
            "--models-root",
            str(model_root),
            "--stage",
            "documentation",
            "--require-stage",
            "documentation",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"command": "smoke-ingest-netlogo"' in result.output


def test_cli_smoke_ingest_netlogo_rejects_unknown_stage(tmp_path: Path) -> None:
    model_root = tmp_path / "data"
    _write_model(model_root, "fauna")
    _write_model(model_root, "grazing")
    _write_model(model_root, "milk_consumption")

    result = runner.invoke(
        app,
        [
            "smoke-ingest-netlogo",
            "--models-root",
            str(model_root),
            "--stage",
            "unknown-stage",
        ],
    )

    assert result.exit_code != 0
    assert "unknown ingest smoke stage" in result.output
