from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import distill_abm.cli as cli_module
from distill_abm.cli import app

runner = CliRunner()


def test_cli_study_llm_same_settings_invokes_runner(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}
    gemini_root = tmp_path / "gemini"
    kimi_root = tmp_path / "kimi"
    qwen_root = tmp_path / "qwen"
    opus_root = tmp_path / "opus"
    for root in (gemini_root, kimi_root, qwen_root, opus_root):
        root.mkdir()

    def fake_run_llm_same_settings_study(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        run_root = Path(kwargs["output_root"]) / "runs" / "run_1"
        result = SimpleNamespace(
            run_root=run_root,
            report_json_path=run_root / "report.json",
            report_markdown_path=run_root / "report.md",
        )
        result.model_dump_json = lambda indent=2: (
            '{\n'
            f'  "run_root": "{run_root}",\n'
            f'  "report_json_path": "{run_root / "report.json"}",\n'
            f'  "report_markdown_path": "{run_root / "report.md"}"\n'
            '}'
        )
        return result

    monkeypatch.setattr(cli_module, "run_llm_same_settings_study", fake_run_llm_same_settings_study)

    result = runner.invoke(
        app,
        [
            "study-llm-same-settings",
            "--anchor-source-root",
            str(gemini_root),
            "--comparison-source-root",
            str(kimi_root),
            "--comparison-source-root",
            str(qwen_root),
            "--comparison-source-root",
            str(opus_root),
            "--output-root",
            str(tmp_path / "study"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert "report.json" in result.output
    assert captured["anchor_source_root"] == gemini_root
    assert captured["comparison_source_roots"] == [kimi_root, qwen_root, opus_root]
    assert captured["output_root"] == tmp_path / "study"


def test_cli_study_llm_same_settings_uses_default_output_root(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}
    gemini_root = tmp_path / "gemini"
    kimi_root = tmp_path / "kimi"
    qwen_root = tmp_path / "qwen"
    for root in (gemini_root, kimi_root, qwen_root):
        root.mkdir()

    def fake_run_llm_same_settings_study(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        run_root = Path(kwargs["output_root"]) / "runs" / "run_1"
        return SimpleNamespace(
            run_root=run_root,
            report_json_path=run_root / "report.json",
            report_markdown_path=run_root / "report.md",
        )

    monkeypatch.setattr(cli_module, "run_llm_same_settings_study", fake_run_llm_same_settings_study)

    result = runner.invoke(
        app,
        [
            "study-llm-same-settings",
            "--anchor-source-root",
            str(gemini_root),
            "--comparison-source-root",
            str(kimi_root),
            "--comparison-source-root",
            str(qwen_root),
        ],
    )

    assert result.exit_code == 0
    assert captured["output_root"] == Path("results/side_studies/optimization_same_settings_llm_comparison")
