from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

import distill_abm.cli as cli_module
from distill_abm.cli import app
from distill_abm.configs.models import ABMConfig

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


def test_validate_model_policy_blocks_debug_model_without_flag() -> None:
    with pytest.raises(typer.BadParameter):
        cli_module._validate_model_policy(
            provider="openrouter",
            model="qwen/qwen3-vl-235b-a22b-thinking",
            allow_debug_model=False,
        )


def test_validate_model_policy_allows_supported_benchmark_models() -> None:
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


def test_validate_model_policy_debug_model_allowed_with_flag() -> None:
    cli_module._validate_model_policy(
        provider="openrouter",
        model="qwen/qwen3-vl-235b-a22b-thinking",
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
