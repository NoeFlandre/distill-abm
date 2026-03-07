from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

import distill_abm.cli as cli_module
from distill_abm.cli import app
from distill_abm.configs.models import ABMConfig
from distill_abm.pipeline.doe_smoke import DoESmokeModelSpec

runner = CliRunner()


def _write_min_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text("tick;mean-incum-1\n0;1\n1;2\n", encoding="utf-8")
    params = tmp_path / "params.txt"
    docs = tmp_path / "docs.txt"
    prompts = tmp_path / "prompts.yaml"
    params.write_text("p=1", encoding="utf-8")
    docs.write_text("d=1", encoding="utf-8")
    prompts.write_text(
        'context_prompt: "Context {parameters} {documentation}"\ntrend_prompt: "Trend {description}"\n',
        encoding="utf-8",
    )
    return csv_path, params, docs, prompts


def _write_min_nlogo_model(tmp_path: Path) -> tuple[Path, Path]:
    model_path = tmp_path / "model.nlogo"
    model_path.write_text(
        "globals [a b]\n"
        "to go\nend\n"
        "@#$#@#$#@\n"
        "## WHAT IS IT?\n\nDemo doc text for testing ingestion.\n"
        "@#$#@#$#@\n"
        "SLIDER 0 0 10 10 slider-a slider-a 0 10 1 5\n"
        "SWITCH 0 0 10 10 switch-b switch-b 1 0 0\n",
        encoding="utf-8",
    )
    experiment_parameters_path = tmp_path / "experiment-parameters.json"
    experiment_parameters_path.write_text('{"slider-a": 3, "switch-b": true}', encoding="utf-8")
    return model_path, experiment_parameters_path


def _write_min_nlogo_model_dir(root: Path, abm_name: str, doc_text: str) -> tuple[Path, Path, Path]:
    model_dir = root / f"{abm_name}_abm"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{abm_name}.nlogo"
    model_path.write_text(
        "globals [a b]\n"
        "to go\nend\n"
        "@#$#@#$#@\n"
        f"## WHAT IS IT?\n\n{doc_text}\n"
        "@#$#@#$#@\n"
        "SLIDER 0 0 10 10 slider-a slider-a 0 10 1 5\n"
        "SWITCH 0 0 10 10 switch-b switch-b 1 0 0\n",
        encoding="utf-8",
    )
    experiment_parameters_path = model_dir / "experiment_parameters.json"
    experiment_parameters_path.write_text('{"slider-a": 3, "switch-b": true}', encoding="utf-8")
    return model_path, experiment_parameters_path, model_dir


def _write_min_nlogo_model_file(
    root: Path,
    filename: str,
    doc_text: str,
) -> Path:
    model_path = root / filename
    model_path.write_text(
        "globals [a b]\n"
        "to go\nend\n"
        "@#$#@#$#@\n"
        f"## WHAT IS IT?\n\n{doc_text}\n"
        "@#$#@#$#@\n"
        "SLIDER 0 0 10 10 slider-a slider-a 0 10 1 5\n"
        "SWITCH 0 0 10 10 switch-b switch-b 1 0 0\n",
        encoding="utf-8",
    )
    return model_path


def test_cli_run_forwards_paper_modes_and_summarizers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path, params, docs, prompts = _write_min_inputs(tmp_path)
    captured: dict[str, Any] = {}

    class _Result:
        def __init__(self, output_dir: Path) -> None:
            self.plot_path = output_dir / "plot.png"
            self.report_csv = output_dir / "report.csv"

    def fake_run_pipeline(*, inputs, prompts, adapter):  # type: ignore[no-untyped-def]
        captured["inputs"] = inputs
        output_dir = inputs.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "plot.png").write_text("plot", encoding="utf-8")
        (output_dir / "report.csv").write_text("report", encoding="utf-8")
        return _Result(output_dir)

    monkeypatch.setattr(cli_module, "_validate_model_policy", lambda **_: None)
    monkeypatch.setattr(cli_module, "create_adapter", lambda provider, model: object())
    monkeypatch.setattr(cli_module, "run_pipeline", fake_run_pipeline)

    result = runner.invoke(
        app,
        [
            "run",
            "--csv-path",
            str(csv_path),
            "--parameters-path",
            str(params),
            "--documentation-path",
            str(docs),
            "--prompts-path",
            str(prompts),
            "--output-dir",
            str(tmp_path / "out"),
            "--provider",
            "echo",
            "--model",
            "echo-model",
            "--metric-pattern",
            "mean-incum",
            "--metric-description",
            "weekly milk",
            "--evidence-mode",
            "table",
            "--text-source-mode",
            "summary_only",
            "--summarizer",
            "bart",
            "--summarizer",
            "t5",
            "--allow-summary-fallback",
        ],
    )

    assert result.exit_code == 0
    inputs = captured["inputs"]
    assert inputs.evidence_mode == "table"
    assert inputs.text_source_mode == "summary_only"
    assert inputs.summarizers == ("bart", "t5")
    assert inputs.allow_summary_fallback is True


def test_cli_run_with_model_id_uses_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path, params, docs, prompts = _write_min_inputs(tmp_path)
    models = tmp_path / "models.yaml"
    models.write_text(
        """
models:
  kimi:
    provider: openrouter
    model: moonshotai/kimi-k2.5
""",
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    class _Result:
        def __init__(self, output_dir: Path) -> None:
            self.plot_path = output_dir / "plot.png"
            self.report_csv = output_dir / "report.csv"

    def fake_create_adapter(provider: str, model: str):  # type: ignore[no-untyped-def]
        captured["provider"] = provider
        captured["model"] = model
        return object()

    def fake_run_pipeline(*, inputs, prompts, adapter):  # type: ignore[no-untyped-def]
        _ = prompts, adapter
        output_dir = inputs.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "plot.png").write_text("plot", encoding="utf-8")
        (output_dir / "report.csv").write_text("report", encoding="utf-8")
        return _Result(output_dir)

    monkeypatch.setattr(cli_module, "_validate_model_policy", lambda **_: None)
    monkeypatch.setattr(cli_module, "create_adapter", fake_create_adapter)
    monkeypatch.setattr(cli_module, "run_pipeline", fake_run_pipeline)

    result = runner.invoke(
        app,
        [
            "run",
            "--csv-path",
            str(csv_path),
            "--parameters-path",
            str(params),
            "--documentation-path",
            str(docs),
            "--prompts-path",
            str(prompts),
            "--models-path",
            str(models),
            "--model-id",
            "kimi",
        ],
    )

    assert result.exit_code == 0
    assert captured["provider"] == "openrouter"
    assert captured["model"] == "moonshotai/kimi-k2.5"


def test_cli_validate_workspace_prints_json_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_root = tmp_path / "data"
    _write_min_nlogo_model_dir(model_root, "fauna", "Fauna doc")
    _write_min_nlogo_model_dir(model_root, "grazing", "Grazing doc")
    _write_min_nlogo_model_dir(model_root, "milk_consumption", "Milk doc")

    captured: dict[str, Any] = {}

    def fake_run_validation_suite(*, output_root, abm_models, checks, ingest_stage_ids, profile):  # type: ignore[no-untyped-def]
        captured["output_root"] = output_root
        captured["abm_models"] = abm_models
        captured["checks"] = checks
        captured["ingest_stage_ids"] = ingest_stage_ids
        captured["profile"] = profile
        return SimpleNamespace(
            success=True,
            failed_checks=[],
            ingest_smoke_report_json_path=Path("ingest_smoke.json"),
            ingest_smoke_report_markdown_path=Path("ingest_smoke.md"),
            report_json_path=Path("validation_report.json"),
            report_markdown_path=Path("validation_report.md"),
            model_dump_json=lambda indent=2: '{"success": true, "selected_checks": ["pytest"]}',
        )

    monkeypatch.setattr(cli_module, "run_validation_suite", fake_run_validation_suite)

    result = runner.invoke(
        app,
        [
            "validate-workspace",
            "--models-root",
            str(model_root),
            "--check",
            "pytest",
            "--ingest-stage",
            "documentation",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"success": true' in result.output
    assert captured["checks"] == ["pytest"]
    assert captured["ingest_stage_ids"] == ["documentation"]
    assert captured["profile"] == "default"
    assert sorted(captured["abm_models"]) == ["fauna", "grazing", "milk_consumption"]


def test_cli_validate_workspace_fails_on_unknown_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_root = tmp_path / "data"
    _write_min_nlogo_model_dir(model_root, "fauna", "Fauna doc")
    _write_min_nlogo_model_dir(model_root, "grazing", "Grazing doc")
    _write_min_nlogo_model_dir(model_root, "milk_consumption", "Milk doc")

    def fake_run_validation_suite(*, output_root, abm_models, checks, ingest_stage_ids, profile):  # type: ignore[no-untyped-def]
        _ = output_root, abm_models, ingest_stage_ids, profile
        raise ValueError(f"unknown validation check(s): {', '.join(checks or [])}. Known checks: pytest")

    monkeypatch.setattr(cli_module, "run_validation_suite", fake_run_validation_suite)

    result = runner.invoke(
        app,
        [
            "validate-workspace",
            "--models-root",
            str(model_root),
            "--check",
            "unknown-check",
        ],
    )

    assert result.exit_code != 0
    assert "unknown validation check" in result.output


def test_cli_validate_workspace_accepts_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_root = tmp_path / "data"
    _write_min_nlogo_model_dir(model_root, "fauna", "Fauna doc")
    _write_min_nlogo_model_dir(model_root, "grazing", "Grazing doc")
    _write_min_nlogo_model_dir(model_root, "milk_consumption", "Milk doc")

    captured: dict[str, Any] = {}

    def fake_run_validation_suite(*, output_root, abm_models, checks, ingest_stage_ids, profile):  # type: ignore[no-untyped-def]
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
            "validate-workspace",
            "--models-root",
            str(model_root),
            "--profile",
            "quick",
        ],
    )

    assert result.exit_code == 0
    assert captured["profile"] == "quick"


def test_health_check_prints_json_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_root = tmp_path / "data"
    _write_min_nlogo_model_dir(model_root, "fauna", "Fauna doc")
    _write_min_nlogo_model_dir(model_root, "grazing", "Grazing doc")
    _write_min_nlogo_model_dir(model_root, "milk_consumption", "Milk doc")
    ingest_root = tmp_path / "ingest"
    viz_root = tmp_path / "viz"
    ingest_root.mkdir()
    viz_root.mkdir()

    models = tmp_path / "models.yaml"
    models.write_text(
        """
models:
  qwen3_5_local:
    provider: ollama
    model: qwen3.5:0.8b
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(cli_module, "_assert_ollama_model_available", lambda _model: None)
    result = runner.invoke(
        app,
        [
            "health-check",
            "--models-root",
            str(model_root),
            "--ingest-root",
            str(ingest_root),
            "--viz-root",
            str(viz_root),
            "--models-path",
            str(models),
            "--include-ollama",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["success"] is True
    assert payload["checks"]["ollama_local_qwen"]["ok"] is True


def test_cli_run_with_abm_uses_scoring_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path, params, docs, prompts = _write_min_inputs(tmp_path)
    captured: dict[str, Any] = {}

    def fake_load_abm_config(_path: Path) -> ABMConfig:
        return ABMConfig(
            name="grazing",
            metric_pattern="household-risk-att-init",
            metric_description="grazing dynamics",
            plot_descriptions=["first plot description"],
        )

    class _Result:
        def __init__(self, output_dir: Path) -> None:
            self.plot_path = output_dir / "plot.png"
            self.report_csv = output_dir / "report.csv"

    def fake_run_pipeline(*, inputs, prompts, adapter):  # type: ignore[no-untyped-def]
        captured["inputs"] = inputs
        output_dir = inputs.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "plot.png").write_text("plot", encoding="utf-8")
        (output_dir / "report.csv").write_text("report", encoding="utf-8")
        return _Result(output_dir)

    monkeypatch.setattr(cli_module, "_validate_model_policy", lambda **_: None)
    monkeypatch.setattr(cli_module, "create_adapter", lambda provider, model: object())
    monkeypatch.setattr(cli_module, "load_abm_config", fake_load_abm_config)
    monkeypatch.setattr(cli_module, "_resolve_scoring_reference_path", lambda _abm: Path("ground_truth.txt"))
    monkeypatch.setattr(cli_module, "run_pipeline", fake_run_pipeline)

    result = runner.invoke(
        app,
        [
            "run",
            "--csv-path",
            str(csv_path),
            "--parameters-path",
            str(params),
            "--documentation-path",
            str(docs),
            "--prompts-path",
            str(prompts),
            "--provider",
            "echo",
            "--model",
            "echo-model",
            "--abm",
            "grazing",
        ],
    )

    assert result.exit_code == 0
    inputs = captured["inputs"]
    assert inputs.metric_pattern == "household-risk-att-init"
    assert inputs.metric_description == "grazing dynamics"
    assert inputs.plot_description == "first plot description"
    assert inputs.scoring_reference_path == Path("ground_truth.txt")


def test_cli_run_rejects_invalid_summarizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path, params, docs, prompts = _write_min_inputs(tmp_path)
    monkeypatch.setattr(cli_module, "_validate_model_policy", lambda **_: None)
    result = runner.invoke(
        app,
        [
            "run",
            "--csv-path",
            str(csv_path),
            "--parameters-path",
            str(params),
            "--documentation-path",
            str(docs),
            "--prompts-path",
            str(prompts),
            "--provider",
            "echo",
            "--model",
            "echo-model",
            "--summarizer",
            "bad",
        ],
    )
    assert result.exit_code != 0
    assert result.exception is not None


def test_cli_ingest_netlogo_generates_visible_artifacts(tmp_path: Path) -> None:
    model_path, experiment_parameters_path = _write_min_nlogo_model(tmp_path)
    output_dir = tmp_path / "ingest_output"
    result = runner.invoke(
        app,
        [
            "ingest-netlogo",
            "--model-path",
            str(model_path),
            "--experiment-parameters-path",
            str(experiment_parameters_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "JSON" / "documentation.json").exists()
    assert (output_dir / "JSON" / "cleaned_documentation.json").exists()
    assert (output_dir / "JSON" / "documentation_without_default.json").exists()
    assert (output_dir / "TXT" / "final_documentation.txt").exists()
    assert (output_dir / "TXT" / "narrative_combined.txt").exists()


def test_cli_ingest_netlogo_supports_json_output(tmp_path: Path) -> None:
    model_path, experiment_parameters_path = _write_min_nlogo_model(tmp_path)
    output_dir = tmp_path / "ingest_output"
    result = runner.invoke(
        app,
        [
            "ingest-netlogo",
            "--model-path",
            str(model_path),
            "--experiment-parameters-path",
            str(experiment_parameters_path),
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"command": "ingest-netlogo"' in result.output
    assert '"artifact_manifest_path"' in result.output


def test_cli_ingest_netlogo_suite_generates_all_configured_abms(tmp_path: Path) -> None:
    model_root = tmp_path / "data"
    for abm, doc in [
        ("fauna", "Fauna documentation"),
        ("grazing", "Grazing documentation"),
        ("milk_consumption", "Milk documentation"),
    ]:
        _write_min_nlogo_model_dir(model_root, abm, doc)
    result = runner.invoke(
        app,
        [
            "ingest-netlogo-suite",
            "--models-root",
            str(model_root),
            "--output-root",
            str(tmp_path / "ingest"),
        ],
    )

    assert result.exit_code == 0
    for abm in ["fauna", "grazing", "milk_consumption"]:
        assert (tmp_path / "ingest" / abm / "JSON" / "documentation.json").exists()
        assert (tmp_path / "ingest" / abm / "JSON" / "cleaned_documentation.json").exists()
        assert (tmp_path / "ingest" / abm / "TXT" / "final_documentation.txt").exists()
        assert (tmp_path / "ingest" / abm / "TXT" / "narrative_combined.txt").exists()


def test_cli_describe_abm_returns_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_root = tmp_path / "data"
    _write_min_nlogo_model_dir(model_root, "grazing", "Grazing doc")

    monkeypatch.setattr(cli_module, "_resolve_scoring_reference_path", lambda _abm: Path("ground_truth.txt"))

    result = runner.invoke(
        app,
        [
            "describe-abm",
            "--abm",
            "grazing",
            "--models-root",
            str(model_root),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"abm": "grazing"' in result.output
    assert '"model_path"' in result.output


def test_cli_describe_ingest_artifacts_reads_manifest(tmp_path: Path) -> None:
    ingest_root = tmp_path / "ingest"
    ingest_root.mkdir(parents=True, exist_ok=True)
    artifact = ingest_root / "artifact.txt"
    artifact.write_text("hello", encoding="utf-8")
    manifest = ingest_root / "ingest_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "command": "ingest-netlogo",
                "output_dir": str(ingest_root),
                "artifact_manifest_path": str(manifest),
                "artifacts": {
                    "artifact": {
                        "path": str(artifact),
                        "exists": True,
                        "size_bytes": 5,
                        "sha256": "abc",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["describe-ingest-artifacts", "--root", str(ingest_root), "--json"])

    assert result.exit_code == 0
    assert '"manifest_path"' in result.output
    assert '"artifact"' in result.output


def test_cli_describe_run_returns_metadata_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "pipeline_run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "artifacts": {"plot_path": str(output_dir / "plot.png")},
                "reproducibility": {"run_signature": "sig-123"},
                "execution": {
                    "selected_text_source": "summary_only",
                    "evidence_mode": "plot+table",
                    "requested_evidence_mode": "plot+table",
                },
                "debug_trace": {"frame_summary": {"matched_metric_columns": ["metric-a"]}},
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["describe-run", "--output-dir", str(output_dir), "--json"])

    assert result.exit_code == 0
    assert '"run_signature": "sig-123"' in result.output
    assert '"matched_metric_columns"' in result.output


def test_cli_smoke_viz_supports_json_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_specs = {"milk_consumption": object()}

    def fake_run_viz_smoke_suite(*, specs, netlogo_home, output_root, stage_ids):  # type: ignore[no-untyped-def]
        _ = specs, netlogo_home, output_root, stage_ids
        return SimpleNamespace(
            success=True,
            failed_abms=[],
            selected_stage_ids=["plot-1"],
            report_markdown_path=Path("viz_smoke.md"),
            report_json_path=Path("viz_smoke.json"),
        )

    monkeypatch.setattr(cli_module, "_resolve_viz_smoke_specs", lambda **_: fake_specs)
    monkeypatch.setattr(cli_module, "run_viz_smoke_suite", fake_run_viz_smoke_suite)

    result = runner.invoke(
        app,
        [
            "smoke-viz",
            "--abm",
            "milk_consumption",
            "--models-root",
            str(tmp_path),
            "--netlogo-home",
            "/fake/netlogo",
            "--stage",
            "plot-1",
            "--require-stage",
            "plot-1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"command": "smoke-viz"' in result.output


def test_cli_smoke_doe_supports_json_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_root = tmp_path / "data"
    _write_min_nlogo_model_dir(model_root, "fauna", "Fauna doc")
    _write_min_nlogo_model_dir(model_root, "grazing", "Grazing doc")
    _write_min_nlogo_model_dir(model_root, "milk_consumption", "Milk doc")

    ingest_root = tmp_path / "ingest"
    viz_root = tmp_path / "viz"
    for abm in ["fauna", "grazing", "milk_consumption"]:
        txt_dir = ingest_root / abm / "TXT"
        txt_dir.mkdir(parents=True, exist_ok=True)
        (txt_dir / "narrative_combined.txt").write_text("parameters", encoding="utf-8")
        (txt_dir / "final_documentation.txt").write_text("documentation", encoding="utf-8")
        abm_viz = viz_root / abm
        (abm_viz / "plots").mkdir(parents=True, exist_ok=True)
        (abm_viz / "simulation.csv").write_text("[step],mean\n0,1\n", encoding="utf-8")
        (abm_viz / "plots" / "1.png").write_bytes(b"png")
        (abm_viz / "artifact_source.txt").write_text("fallback\n", encoding="utf-8")

    def fake_run_doe_smoke_suite(  # type: ignore[no-untyped-def]
        *,
        abm_inputs,
        prompts,
        model_specs,
        output_root,
        evidence_modes,
        summarization_specs,
        prompt_variants,
        repetitions,
    ):
        _ = (
            abm_inputs,
            prompts,
            model_specs,
            output_root,
            evidence_modes,
            summarization_specs,
            prompt_variants,
            repetitions,
        )
        return SimpleNamespace(
            success=True,
            failed_case_ids=[],
            report_markdown_path=Path("doe_smoke.md"),
            report_json_path=Path("doe_smoke.json"),
            design_matrix_csv_path=Path("design_matrix.csv"),
            request_matrix_csv_path=Path("request_matrix.csv"),
        )

    monkeypatch.setattr(cli_module, "run_doe_smoke_suite", fake_run_doe_smoke_suite)

    result = runner.invoke(
        app,
        [
            "smoke-doe",
            "--models-root",
            str(model_root),
            "--ingest-root",
            str(ingest_root),
            "--viz-root",
            str(viz_root),
            "--model-id",
            "kimi_k2_5",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"command": "smoke-doe"' in result.output


def test_cli_smoke_doe_treats_candidate_models_as_design_factors_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "data"
    _write_min_nlogo_model_dir(model_root, "fauna", "Fauna doc")
    _write_min_nlogo_model_dir(model_root, "grazing", "Grazing doc")
    _write_min_nlogo_model_dir(model_root, "milk_consumption", "Milk doc")

    ingest_root = tmp_path / "ingest"
    viz_root = tmp_path / "viz"
    for abm in ["fauna", "grazing", "milk_consumption"]:
        txt_dir = ingest_root / abm / "TXT"
        txt_dir.mkdir(parents=True, exist_ok=True)
        (txt_dir / "narrative_combined.txt").write_text("parameters", encoding="utf-8")
        (txt_dir / "final_documentation.txt").write_text("documentation", encoding="utf-8")
        abm_viz = viz_root / abm
        (abm_viz / "plots").mkdir(parents=True, exist_ok=True)
        (abm_viz / "simulation.csv").write_text("[step],mean\n0,1\n", encoding="utf-8")
        (abm_viz / "plots" / "1.png").write_bytes(b"png")
        (abm_viz / "artifact_source.txt").write_text("fallback\n", encoding="utf-8")

    captured_model_specs: list[DoESmokeModelSpec] = []

    def fake_run_doe_smoke_suite(  # type: ignore[no-untyped-def]
        *,
        abm_inputs,
        prompts,
        model_specs,
        output_root,
        evidence_modes,
        summarization_specs,
        prompt_variants,
        repetitions,
    ):
        _ = (
            abm_inputs,
            prompts,
            output_root,
            evidence_modes,
            summarization_specs,
            prompt_variants,
            repetitions,
        )
        captured_model_specs.extend(model_specs)
        return SimpleNamespace(
            success=True,
            failed_case_ids=[],
            report_markdown_path=Path("doe_smoke.md"),
            report_json_path=Path("doe_smoke.json"),
            design_matrix_csv_path=Path("design_matrix.csv"),
            request_matrix_csv_path=Path("request_matrix.csv"),
        )

    monkeypatch.setattr(cli_module, "run_doe_smoke_suite", fake_run_doe_smoke_suite)

    result = runner.invoke(
        app,
        [
            "smoke-doe",
            "--models-root",
            str(model_root),
            "--ingest-root",
            str(ingest_root),
            "--viz-root",
            str(viz_root),
            "--model-id",
            "qwen3_5_local",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured_model_specs[0].model_id == "qwen3_5_local"
    assert captured_model_specs[0].preflight_error is None


def test_cli_ingest_netlogo_suite_supports_root_level_model_files(tmp_path: Path) -> None:
    model_root = tmp_path / "data"
    model_root.mkdir()
    _write_min_nlogo_model_file(model_root, "fauna.nlogo", "Root fauna documentation")
    _write_min_nlogo_model_file(model_root, "grazing.nlogo", "Root grazing documentation")
    _write_min_nlogo_model_file(model_root, "model.nlogo", "Root milk consumption documentation")
    result = runner.invoke(
        app,
        [
            "ingest-netlogo-suite",
            "--models-root",
            str(model_root),
            "--output-root",
            str(tmp_path / "ingest"),
        ],
    )

    assert result.exit_code == 0
    for abm in ["fauna", "grazing", "milk_consumption"]:
        assert (tmp_path / "ingest" / abm / "JSON" / "documentation.json").exists()
        assert (tmp_path / "ingest" / abm / "JSON" / "cleaned_documentation.json").exists()
        assert (tmp_path / "ingest" / abm / "TXT" / "final_documentation.txt").exists()
        assert (tmp_path / "ingest" / abm / "TXT" / "narrative_combined.txt").exists()


def test_cli_evaluate_qualitative_outputs_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "_validate_model_policy", lambda **_: None)
    monkeypatch.setattr(cli_module, "create_adapter", lambda provider, model: object())
    monkeypatch.setattr(
        cli_module,
        "evaluate_qualitative_score",
        lambda **_: SimpleNamespace(model_dump_json=lambda: '{"score": 4, "reasoning": "ok"}'),
    )

    result = runner.invoke(
        app,
        [
            "evaluate-qualitative",
            "--summary-text",
            "summary",
            "--source-text",
            "source",
            "--metric",
            "coverage",
            "--provider",
            "echo",
            "--model",
            "echo-model",
            "--allow-debug-model",
        ],
    )

    assert result.exit_code == 0
    assert '"score": 4' in result.stdout


def test_cli_smoke_qwen_forwards_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path, params, docs, prompts = _write_min_inputs(tmp_path)
    captured: dict[str, Any] = {}

    def fake_run_smoke(*, inputs, prompts, adapter, run_qualitative, doe_input_csv, run_sweep, cases, resume_existing):  # type: ignore[no-untyped-def]
        _ = prompts, adapter, run_qualitative, doe_input_csv, run_sweep, cases, resume_existing
        captured["inputs"] = inputs
        return SimpleNamespace(
            report_markdown_path=Path("smoke.md"),
            report_json_path=Path("smoke.json"),
            doe_output_csv=None,
            sweep_output_csv=None,
            success=True,
            failed_cases=[],
        )

    monkeypatch.setattr(cli_module, "_validate_model_policy", lambda **_: None)
    monkeypatch.setattr(cli_module, "create_adapter", lambda provider, model: object())
    monkeypatch.setattr(cli_module, "run_qwen_smoke_suite", fake_run_smoke)

    result = runner.invoke(
        app,
        [
            "smoke-qwen",
            "--csv-path",
            str(csv_path),
            "--parameters-path",
            str(params),
            "--documentation-path",
            str(docs),
            "--prompts-path",
            str(prompts),
            "--provider",
            "echo",
            "--model",
            "echo-model",
            "--evidence-mode",
            "table",
            "--text-source-mode",
            "full_text_only",
            "--summarizer",
            "bert",
            "--allow-summary-fallback",
        ],
    )

    assert result.exit_code == 0
    inputs = captured["inputs"]
    assert inputs.evidence_mode == "table"
    assert inputs.text_source_mode == "full_text_only"
    assert inputs.allow_summary_fallback is True
    assert inputs.summarizers == ("bert",)


def test_validate_model_policy_blocks_unsupported_model() -> None:
    with pytest.raises(typer.BadParameter):
        cli_module._validate_model_policy(provider="openai", model="gpt-4o", allow_debug_model=False)


def test_validate_model_policy_allows_supported_benchmark_models(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], check: bool, capture_output: bool, text: bool):  # type: ignore[no-untyped-def]
        _ = (cmd, check, capture_output, text)
        return SimpleNamespace(stdout="NAME           ID\nqwen3.5:0.8b   0\n", returncode=0)

    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)
    cli_module._validate_model_policy(
        provider="openrouter", model="moonshotai/kimi-k2.5", allow_debug_model=False
    )
    cli_module._validate_model_policy(
        provider="openrouter", model="google/gemini-3.1-pro-preview", allow_debug_model=False
    )
    cli_module._validate_model_policy(provider="ollama", model="qwen3.5:0.8b", allow_debug_model=False)


def test_validate_model_policy_blocks_unsupported_benchmark_model() -> None:
    with pytest.raises(typer.BadParameter):
        cli_module._validate_model_policy(provider="openrouter", model="unsupported-model", allow_debug_model=False)


def test_validate_model_policy_supported_model_is_allowed_with_flag() -> None:
    cli_module._validate_model_policy(
        provider="openrouter",
        model="google/gemini-3.1-pro-preview",
        allow_debug_model=True,
    )


def test_validate_model_policy_requires_local_ollama_model_available(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_run(cmd: list[str], check: bool, capture_output: bool, text: bool):  # type: ignore[no-untyped-def]
        captured["cmd"] = " ".join(cmd)
        return SimpleNamespace(stdout="NAME           ID\nqwen3.5:0.8b   0\n", returncode=0)

    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)
    cli_module._validate_model_policy(provider="ollama", model="qwen3.5:0.8b", allow_debug_model=False)
    assert captured["cmd"] == "ollama list"


def test_cli_analyze_doe_exits_on_analysis_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that analyze-doe exits with code 1 when analysis returns None."""
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("a,b,score\n1,1,10\n2,2,20\n", encoding="utf-8")

    def fake_analyze(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(cli_module, "analyze_factorial_anova", fake_analyze)

    result = runner.invoke(
        app,
        [
            "analyze-doe",
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(tmp_path / "output.csv"),
        ],
    )

    assert result.exit_code == 1


def test_cli_run_fails_on_missing_csv_file(tmp_path: Path) -> None:
    """Test that run command fails when CSV file doesn't exist."""
    csv_path = tmp_path / "nonexistent.csv"
    params = tmp_path / "params.txt"
    docs = tmp_path / "docs.txt"
    prompts = tmp_path / "prompts.yaml"

    params.write_text("p=1", encoding="utf-8")
    docs.write_text("d=1", encoding="utf-8")
    prompts.write_text(
        'context_prompt: "Context {parameters} {documentation}"\ntrend_prompt: "Trend {description}"\n',
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "run",
            "--csv-path",
            str(csv_path),
            "--parameters-path",
            str(params),
            "--documentation-path",
            str(docs),
            "--prompts-path",
            str(prompts),
            "--provider",
            "echo",
            "--model",
            "echo-model",
        ],
    )

    assert result.exit_code != 0
    assert result.exit_code != 0  # Typer validates file existence


def test_cli_ingest_netlogo_suite_continues_on_missing_with_flag(tmp_path: Path) -> None:
    """Test that ingest-netlogo-suite continues processing when ABM is missing and --continue-on-missing is set."""
    model_root = tmp_path / "data"
    # Only create one ABM folder
    _write_min_nlogo_model_dir(model_root, "fauna", "Fauna documentation")

    result = runner.invoke(
        app,
        [
            "ingest-netlogo-suite",
            "--models-root",
            str(model_root),
            "--output-root",
            str(tmp_path / "ingest"),
            "--abm",
            "fauna",
            "--abm",
            "grazing",  # This one doesn't exist
            "--continue-on-missing",
        ],
    )

    assert result.exit_code == 0
    assert "skipped ABMs" in result.stdout
    assert "grazing" in result.stdout


def test_cli_ingest_netlogo_suite_fails_on_missing_without_flag(tmp_path: Path) -> None:
    """Test that ingest-netlogo-suite fails when ABM is missing and --continue-on-missing is not set."""
    model_root = tmp_path / "data"
    # Only create one ABM folder
    _write_min_nlogo_model_dir(model_root, "fauna", "Fauna documentation")

    result = runner.invoke(
        app,
        [
            "ingest-netlogo-suite",
            "--models-root",
            str(model_root),
            "--output-root",
            str(tmp_path / "ingest"),
            "--abm",
            "fauna",
            "--abm",
            "grazing",  # This one doesn't exist
        ],
    )

    assert result.exit_code != 0
    assert "grazing" in result.output


def test_cli_smoke_qwen_exits_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that smoke-qwen exits with code 1 when the smoke suite reports failure."""
    csv_path, params, docs, prompts = _write_min_inputs(tmp_path)

    def fake_run_smoke(*, inputs, prompts, adapter, run_qualitative, doe_input_csv, run_sweep, cases, resume_existing):  # type: ignore[no-untyped-def]
        _ = prompts, adapter, run_qualitative, doe_input_csv, run_sweep, cases, resume_existing
        return SimpleNamespace(
            report_markdown_path=Path("smoke.md"),
            report_json_path=Path("smoke.json"),
            doe_output_csv=None,
            sweep_output_csv=None,
            success=False,
            failed_cases=["case-1", "case-2"],
        )

    monkeypatch.setattr(cli_module, "_validate_model_policy", lambda **_: None)
    monkeypatch.setattr(cli_module, "create_adapter", lambda provider, model: object())
    monkeypatch.setattr(cli_module, "run_qwen_smoke_suite", fake_run_smoke)

    result = runner.invoke(
        app,
        [
            "smoke-qwen",
            "--csv-path",
            str(csv_path),
            "--parameters-path",
            str(params),
            "--documentation-path",
            str(docs),
            "--prompts-path",
            str(prompts),
            "--provider",
            "echo",
            "--model",
            "echo-model",
        ],
    )

    assert result.exit_code == 1
    assert "smoke suite failed" in result.stdout


def test_cli_smoke_qwen_rejects_unknown_case_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that smoke-qwen rejects unknown --case-id values."""
    csv_path, params, docs, prompts = _write_min_inputs(tmp_path)

    monkeypatch.setattr(cli_module, "_validate_model_policy", lambda **_: None)

    result = runner.invoke(
        app,
        [
            "smoke-qwen",
            "--csv-path",
            str(csv_path),
            "--parameters-path",
            str(params),
            "--documentation-path",
            str(docs),
            "--prompts-path",
            str(prompts),
            "--provider",
            "echo",
            "--model",
            "echo-model",
            "--case-id",
            "nonexistent-case-id",
        ],
    )

    assert result.exit_code != 0
    assert "unknown --case-id" in result.output


def test_cli_run_fails_on_unknown_model_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that run command fails when model_id is not found in registry."""
    csv_path, params, docs, prompts = _write_min_inputs(tmp_path)
    models = tmp_path / "models.yaml"
    models.write_text(
        """
models:
  kimi:
    provider: openrouter
    model: moonshotai/kimi-k2.5
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(cli_module, "_validate_model_policy", lambda **_: None)

    result = runner.invoke(
        app,
        [
            "run",
            "--csv-path",
            str(csv_path),
            "--parameters-path",
            str(params),
            "--documentation-path",
            str(docs),
            "--prompts-path",
            str(prompts),
            "--models-path",
            str(models),
            "--model-id",
            "nonexistent-model",  # Not in registry
        ],
    )

    assert result.exit_code != 0
    assert "unknown model_id" in result.output


def test_cli_ingest_fails_on_missing_experiment_parameters_file(tmp_path: Path) -> None:
    """Test that ingest-netlogo fails when experiment parameters file is specified but doesn't exist."""
    model_path, _ = _write_min_nlogo_model(tmp_path)
    output_dir = tmp_path / "ingest_output"
    missing_params = tmp_path / "nonexistent_params.json"

    result = runner.invoke(
        app,
        [
            "ingest-netlogo",
            "--model-path",
            str(model_path),
            "--experiment-parameters-path",
            str(missing_params),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code != 0
    assert "not found" in result.output


def test_cli_ingest_fails_on_invalid_experiment_parameters_json(tmp_path: Path) -> None:
    """Test that ingest-netlogo fails when experiment parameters file contains invalid JSON."""
    model_path, _ = _write_min_nlogo_model(tmp_path)
    output_dir = tmp_path / "ingest_output"
    invalid_params = tmp_path / "invalid_params.json"
    invalid_params.write_text("not valid json", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "ingest-netlogo",
            "--model-path",
            str(model_path),
            "--experiment-parameters-path",
            str(invalid_params),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code != 0
    assert "JSON" in result.output
