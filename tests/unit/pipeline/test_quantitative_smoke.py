from __future__ import annotations

import csv
import json
from pathlib import Path

from distill_abm.eval.metrics import SummaryScores
from distill_abm.pipeline.quantitative_smoke import (
    _build_factorial_input_frame,
    _derive_prompt_flags,
    _render_anova_markdown_table,
    run_quantitative_smoke,
)


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _build_source_roots(tmp_path: Path) -> tuple[Path, Path]:
    matrix_root = tmp_path / "matrix" / "runs" / "run_1"
    summarizer_root = tmp_path / "summ" / "runs" / "run_1"
    matrix_cases = []
    review_rows: list[dict[str, str]] = []
    prompt_variants = ["none", "role", "insights", "example"]
    evidence_modes = ["plot", "table", "plot+table", "plot"]
    modes = ["none", "bart", "bert", "t5", "longformer_ext"]
    for idx, (prompt_variant, evidence_mode) in enumerate(
        zip(prompt_variants, evidence_modes, strict=True),
        start=1,
    ):
        case_id = f"{idx:02d}_grazing_{prompt_variant}_{evidence_mode.replace('+', '_plus_')}_rep1"
        case_dir = matrix_root / "cases" / case_id
        context_output = _write_text(case_dir / "02_context" / "context_output.txt", f"context {case_id}")
        for plot_idx in range(1, 3):
            _write_text(
                case_dir / "03_trends" / f"plot_{plot_idx:02d}" / "trend_output.txt",
                f"trend {plot_idx} {case_id}",
            )
        matrix_cases.append(
            {
                "case_id": case_id,
                "case_dir": str(case_dir),
                "abm": "grazing",
                "evidence_mode": evidence_mode,
                "prompt_variant": prompt_variant,
                "repetition": 1,
                "success": True,
            }
        )
        bundle_dir = summarizer_root / "bundles" / case_id
        combined_input = _write_text(bundle_dir / "01_input" / "combined_input.txt", f"combined {case_id}")
        for mode in modes:
            summary_path = _write_text(
                bundle_dir / "02_summaries" / f"{mode}.txt",
                f"summary {mode} {prompt_variant} {evidence_mode} {case_id}",
            )
            review_rows.append(
                {
                    "bundle_id": case_id,
                    "case_id": case_id,
                    "abm": "grazing",
                    "mode": mode,
                    "success": "True",
                    "context_output_path": str(context_output),
                    "trend_output_paths": "|".join(
                        str(case_dir / "03_trends" / f"plot_{plot_idx:02d}" / "trend_output.txt")
                        for plot_idx in range(1, 3)
                    ),
                    "combined_input_path": str(combined_input),
                    "summary_output_path": str(summary_path),
                    "input_length": "100",
                    "output_length": "80",
                    "duration_seconds": "0.1",
                    "validation_note": "validated",
                    "error": "",
                }
            )
    (matrix_root / "smoke_full_case_matrix_report.json").write_text(
        json.dumps({"cases": matrix_cases}),
        encoding="utf-8",
    )
    (summarizer_root / "review.csv").parent.mkdir(parents=True, exist_ok=True)
    with (summarizer_root / "review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0].keys()))
        writer.writeheader()
        writer.writerows(review_rows)
    return summarizer_root, matrix_root


def _fake_score_summary(reference: str, candidate: str) -> SummaryScores:
    text = candidate.lower()
    mode_bonus = 0.0
    if "bart" in text:
        mode_bonus = 0.1
    elif "bert" in text:
        mode_bonus = 0.2
    elif "t5" in text:
        mode_bonus = 0.3
    elif "longformer_ext" in text:
        mode_bonus = 0.4
    prompt_bonus = 0.05 if "role" in text else 0.0
    evidence_bonus = 0.07 if "plot+table" in text else (0.03 if "table" in text else 0.0)
    score = 0.2 + mode_bonus + prompt_bonus + evidence_bonus
    return SummaryScores(
        token_f1=score,
        precision=score,
        recall=score,
        bleu=score,
        meteor=score + 0.01,
        rouge1=score + 0.02,
        rouge2=score + 0.03,
        rouge_l=score + 0.04,
        flesch_reading_ease=50.0 + score * 10,
        reference_length=len(reference.split()),
        candidate_length=len(candidate.split()),
    )


def test_derive_prompt_flags() -> None:
    assert _derive_prompt_flags("none") == {"role": False, "insights": False, "example": False}
    assert _derive_prompt_flags("role+insights") == {"role": True, "insights": True, "example": False}
    assert _derive_prompt_flags("all_three") == {"role": True, "insights": True, "example": True}


def test_render_anova_markdown_table_marks_fixed_factors_absent() -> None:
    table = _render_anova_markdown_table(
        rows=[
            {
                "label": "Agent-Based Model",
                "BLEU": None,
                "METEOR": None,
                "ROUGE-1": None,
                "ROUGE-2": None,
                "ROUGE-L": None,
                "Reading ease": None,
            },
            {
                "label": "Summarization algorithm",
                "BLEU": 0.004,
                "METEOR": 0.5,
                "ROUGE-1": 0.4,
                "ROUGE-2": 0.3,
                "ROUGE-L": 0.2,
                "Reading ease": 0.1,
            },
        ]
    )
    assert "Agent-Based Model" in table
    assert "| — | — | — | — | — | — |" in table
    assert "<0.01" in table


def test_run_quantitative_smoke_writes_analysis_artifacts(tmp_path: Path) -> None:
    summarizer_root, _matrix_root = _build_source_roots(tmp_path)
    result = run_quantitative_smoke(
        source_root=summarizer_root,
        output_root=tmp_path / "quant",
        score_summary_fn=_fake_score_summary,
    )

    assert result.success is True
    assert result.quantitative_rows_path.exists() is True
    assert result.anova_csv_path.exists() is True
    assert result.factorial_csv_path.exists() is True
    assert result.anova_table_markdown_path.exists() is True
    assert result.factorial_table_markdown_path.exists() is True
    rows = list(csv.DictReader(result.quantitative_rows_path.open(encoding="utf-8")))
    assert len(rows) == 20
    assert rows[0]["reference_family"] == "author"
    assert rows[0]["evidence"]
    assert rows[0]["prompt"]
    assert rows[0]["summarizer"]
    report = json.loads(result.report_json_path.read_text(encoding="utf-8"))
    assert report["success"] is True


def test_build_factorial_input_frame_marks_prompt_flags_as_categorical_strings() -> None:
    frame = _build_factorial_input_frame(
        [
            {
                "summarizer": "bart",
                "evidence": "table",
                "role": "True",
                "insights": "False",
                "example": "True",
                "BLEU": "0.1",
                "METEOR": "0.2",
                "R-1": "0.3",
                "R-2": "0.4",
                "R-L": "0.5",
                "Reading ease": "10.0",
            }
        ]
    )
    assert frame.loc[0, "Role"] == "on"
    assert frame.loc[0, "Insights"] == "off"
    assert frame.loc[0, "Example"] == "on"


def test_run_quantitative_smoke_reuses_valid_rows_when_resuming(tmp_path: Path) -> None:
    summarizer_root, _matrix_root = _build_source_roots(tmp_path)
    first = run_quantitative_smoke(
        source_root=summarizer_root,
        output_root=tmp_path / "quant",
        score_summary_fn=_fake_score_summary,
    )
    assert first.success is True

    def _should_not_run(reference: str, candidate: str) -> SummaryScores:
        raise AssertionError("score_summary should not rerun for valid resumed records")

    resumed = run_quantitative_smoke(
        source_root=summarizer_root,
        output_root=tmp_path / "quant",
        resume=True,
        score_summary_fn=_should_not_run,
    )

    assert resumed.success is True
    assert resumed.run_root != first.run_root
