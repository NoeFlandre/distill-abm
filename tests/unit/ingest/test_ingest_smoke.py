from __future__ import annotations

import json
from pathlib import Path

from distill_abm.ingest.ingest_smoke import (
    default_ingest_smoke_stages,
    run_ingest_smoke_suite,
)


def _write_model(root: Path, abm: str, documentation: str) -> Path:
    model_dir = root / f"{abm}_abm"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / (f"{abm}.nlogo" if abm != "milk_consumption" else "model.nlogo")
    model_path.write_text(
        "globals [a b]\n"
        "to go\n"
        "end\n"
        "@#$#@#$#@\n"
        f"## WHAT IS IT?\n\n{documentation}\n"
        "@#$#@#$#@\n"
        "SLIDER 0 0 10 10 slider-a slider-a 0 10 1 5\n"
        "SWITCH 0 0 10 10 switch-b switch-b 1 0 0\n",
        encoding="utf-8",
    )
    return model_path


def test_default_ingest_smoke_stages_are_granular() -> None:
    stage_ids = [stage.stage_id for stage in default_ingest_smoke_stages()]
    assert stage_ids == [
        "experiment-parameters",
        "gui-parameters",
        "updated-parameters",
        "narrative",
        "documentation",
        "cleaned-documentation",
        "final-documentation",
        "code",
    ]


def test_run_ingest_smoke_suite_writes_stage_level_report(tmp_path: Path) -> None:
    models_root = tmp_path / "data"
    _write_model(models_root, "fauna", "Fauna documentation")
    _write_model(models_root, "grazing", "Grazing documentation")
    _write_model(models_root, "milk_consumption", "Milk documentation")

    result = run_ingest_smoke_suite(
        abm_models={
            "fauna": models_root / "fauna_abm" / "fauna.nlogo",
            "grazing": models_root / "grazing_abm" / "grazing.nlogo",
            "milk_consumption": models_root / "milk_consumption_abm" / "model.nlogo",
        },
        output_root=tmp_path / "ingest-smoke",
        stage_ids=["documentation", "final-documentation"],
    )

    assert result.success is True
    assert result.run_root.parent == tmp_path / "ingest-smoke" / "runs"
    assert result.run_log_path.exists()
    assert (tmp_path / "ingest-smoke" / "latest_run.txt").exists()
    assert result.report_json_path.exists()
    assert result.report_markdown_path.exists()
    assert len(result.abms) == 3
    assert all(len(abm.stage_results) == 2 for abm in result.abms)

    payload = json.loads(result.report_json_path.read_text(encoding="utf-8"))
    fauna = next(item for item in payload["abms"] if item["abm"] == "fauna")
    documentation_stage = next(
        stage
        for stage in fauna["stage_results"]
        if stage["stage"]["stage_id"] == "documentation"
    )
    assert documentation_stage["status"] == "ok"
    assert documentation_stage["artifact"]["exists"] is True
    assert documentation_stage["artifact"]["preview"]


def test_code_stage_allows_todo_comments_in_real_code(tmp_path: Path) -> None:
    model_dir = tmp_path / "data" / "grazing_abm"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "grazing.nlogo"
    model_path.write_text(
        "globals [\n"
        "  value ; todo keep this comment\n"
        "]\n"
        "to go\nend\n"
        "@#$#@#$#@\n"
        "## WHAT IS IT?\n\n"
        "Documentation\n"
        "@#$#@#$#@\n",
        encoding="utf-8",
    )

    result = run_ingest_smoke_suite(
        abm_models={"grazing": model_path},
        output_root=tmp_path / "ingest-smoke",
        stage_ids=["code"],
    )

    assert result.success is True
    assert result.run_log_path.exists()
    assert result.abms[0].stage_results[0].status == "ok"
