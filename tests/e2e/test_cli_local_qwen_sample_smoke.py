from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import distill_abm.cli as cli_module
from distill_abm.cli import app

runner = CliRunner()


def _write_supporting_results(root: Path) -> None:
    for abm in ("fauna", "grazing", "milk_consumption"):
        ingest_dir = root / "ingest" / abm / "TXT"
        viz_dir = root / "viz" / abm / "plots"
        ingest_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(parents=True, exist_ok=True)
        (ingest_dir / "narrative_combined.txt").write_text("params", encoding="utf-8")
        (ingest_dir / "final_documentation.txt").write_text("docs", encoding="utf-8")
        (root / "viz" / abm / "simulation.csv").write_text("[step];metric-a\n0;1\n", encoding="utf-8")
        (viz_dir / "1.png").write_bytes(b"png")


def test_cli_smoke_local_qwen_invokes_sample_smoke(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _write_supporting_results(tmp_path)
    captured: dict[str, object] = {}
    adapter_calls: dict[str, object] = {}

    def fake_run_local_qwen_sample_smoke(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        output_root = kwargs["output_root"]
        return SimpleNamespace(
            success=True,
            report_json_path=Path(output_root) / "report.json",
            report_markdown_path=Path(output_root) / "report.md",
            review_csv_path=Path(output_root) / "review.csv",
            viewer_html_path=Path(output_root) / "review.html",
            failed_case_ids=[],
        )

    monkeypatch.setattr(cli_module, "_assert_ollama_model_available", lambda _model: None)
    def fake_create_adapter(provider, model, **kwargs):  # type: ignore[no-untyped-def]
        adapter_calls.update({"provider": provider, "model": model, **kwargs})
        return object()

    monkeypatch.setattr(cli_module, "create_adapter", fake_create_adapter)
    monkeypatch.setattr(cli_module, "run_local_qwen_sample_smoke", fake_run_local_qwen_sample_smoke)

    result = runner.invoke(
        app,
        [
            "smoke-local-qwen",
            "--ingest-root",
            str(tmp_path / "ingest"),
            "--viz-root",
            str(tmp_path / "viz"),
            "--output-root",
            str(tmp_path / "out"),
            "--max-tokens",
            "2000",
            "--num-ctx",
            "4096",
            "--plot-num-ctx",
            "4096",
            "--table-num-ctx",
            "4096",
            "--plot-table-num-ctx",
            "8192",
            "--resume",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert "report.json" in result.output
    assert captured["model"] == "qwen3.5:0.8b"
    assert captured["resume_existing"] is True
    assert captured["max_tokens"] == 2000
    assert captured["ollama_num_ctx"] == 4096
    assert captured["ollama_num_ctx_by_mode"] == {"plot": 4096, "table": 4096, "plot+table": 8192}
    assert adapter_calls["timeout_seconds"] == 900.0
    assert "review.html" in result.output


def test_cli_smoke_local_qwen_accepts_openrouter_model_alias(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _write_supporting_results(tmp_path)
    captured: dict[str, object] = {}
    adapter_calls: dict[str, object] = {}

    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        """
models:
  nemotron:
    provider: openrouter
    model: nvidia/nemotron-nano-12b-v2-vl:free
""".strip(),
        encoding="utf-8",
    )

    def fake_run_local_qwen_sample_smoke(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        output_root = kwargs["output_root"]
        return SimpleNamespace(
            success=True,
            report_json_path=Path(output_root) / "report.json",
            report_markdown_path=Path(output_root) / "report.md",
            review_csv_path=Path(output_root) / "review.csv",
            viewer_html_path=Path(output_root) / "review.html",
            failed_case_ids=[],
        )

    def fail_if_ollama_checked(_model):  # type: ignore[no-untyped-def]
        raise AssertionError("ollama availability check should not run for openrouter models")

    def fake_create_adapter(provider, model, **kwargs):  # type: ignore[no-untyped-def]
        adapter_calls.update({"provider": provider, "model": model, **kwargs})
        return object()

    monkeypatch.setattr(cli_module, "_assert_ollama_model_available", fail_if_ollama_checked)
    monkeypatch.setattr(cli_module, "create_adapter", fake_create_adapter)
    monkeypatch.setattr(cli_module, "run_local_qwen_sample_smoke", fake_run_local_qwen_sample_smoke)

    result = runner.invoke(
        app,
        [
            "smoke-local-qwen",
            "--models-path",
            str(models_path),
            "--model-id",
            "nemotron",
            "--ingest-root",
            str(tmp_path / "ingest"),
            "--viz-root",
            str(tmp_path / "viz"),
            "--output-root",
            str(tmp_path / "out"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured["model"] == "nvidia/nemotron-nano-12b-v2-vl:free"
    assert adapter_calls["provider"] == "openrouter"
    assert adapter_calls["model"] == "nvidia/nemotron-nano-12b-v2-vl:free"
    assert adapter_calls["timeout_seconds"] == 900.0


def test_cli_render_run_viewer_writes_html(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    case_root = run_root / "cases" / "01_case"
    (case_root / "01_inputs").mkdir(parents=True)
    (case_root / "02_requests").mkdir(parents=True)
    (case_root / "03_outputs").mkdir(parents=True)
    (run_root / "smoke_local_qwen_report.json").write_text(
        (
            '{"success": true, "failed_case_ids": [], "cases": '
            '[{"case_id": "01_case", "abm": "fauna", "evidence_mode": "plot", '
            '"prompt_variant": "none", "model": "m", "resumed_from_existing": false, '
            '"success": true, "error": null}]}'
        ),
        encoding="utf-8",
    )
    (run_root / "run.log.jsonl").write_text('{"message":"x"}\n', encoding="utf-8")
    (case_root / "00_case_summary.json").write_text(
        """{"case_id":"01_case","abm":"fauna","evidence_mode":"plot","prompt_variant":"none","model":"m"}""",
        encoding="utf-8",
    )
    (case_root / "01_inputs" / "context_prompt.txt").write_text("ctx", encoding="utf-8")
    (case_root / "01_inputs" / "documentation.txt").write_text("docs", encoding="utf-8")
    (case_root / "01_inputs" / "parameters.txt").write_text("params", encoding="utf-8")
    (case_root / "01_inputs" / "trend_prompt.txt").write_text("trend", encoding="utf-8")
    (case_root / "03_outputs" / "context_output.txt").write_text("ctx out", encoding="utf-8")
    (case_root / "03_outputs" / "trend_output.txt").write_text("trend out", encoding="utf-8")
    (case_root / "03_outputs" / "context_trace.json").write_text("{}", encoding="utf-8")
    (case_root / "03_outputs" / "trend_trace.json").write_text("{}", encoding="utf-8")

    result = runner.invoke(app, ["render-run-viewer", "--run-root", str(run_root)])

    assert result.exit_code == 0
    assert (run_root / "review.html").exists()
