from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import distill_abm.cli as cli_module
from distill_abm.cli import app

runner = CliRunner()


def test_cli_smoke_quantitative_invokes_runner(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    def fake_run_quantitative_smoke(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        output_root = Path(kwargs["output_root"])
        run_root = output_root / "runs" / "run_1"
        return SimpleNamespace(
            success=True,
            run_root=run_root,
            report_json_path=run_root / "report.json",
            report_markdown_path=run_root / "report.md",
            review_csv_path=run_root / "review.csv",
            quantitative_rows_path=run_root / "quantitative_rows.csv",
            structured_results_path=run_root / "structured_results.csv",
            anova_csv_path=run_root / "anova_pvalues.csv",
            factorial_csv_path=run_root / "factorial_contributions.csv",
            optimal_csv_path=run_root / "best_scores.csv",
            run_log_path=run_root / "run.log.jsonl",
            failed_record_ids=[],
        )

    monkeypatch.setattr(cli_module, "run_quantitative_smoke", fake_run_quantitative_smoke)

    result = runner.invoke(
        app,
        [
            "smoke-quantitative",
            "--source-root",
            str(tmp_path / "source"),
            "--output-root",
            str(tmp_path / "out"),
            "--resume",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert "quantitative_rows.csv" in result.output
    assert "best_scores.csv" in result.output
    assert captured["source_root"] == tmp_path / "source"
    assert captured["output_root"] == tmp_path / "out"
    assert captured["resume"] is True
