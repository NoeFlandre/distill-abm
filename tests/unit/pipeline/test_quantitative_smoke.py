from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from distill_abm.eval.metrics import SummaryScores
from distill_abm.pipeline.quantitative_smoke import (
    _build_factorial_input_frame,
    _build_structured_results_rows,
    _derive_prompt_flags,
    _normalize_factorial_table,
    _render_anova_markdown_table,
    _render_factorial_latex_table,
    _render_overview_factorial_markdown_table,
    _render_optimal_latex_table,
    _render_optimal_markdown_table,
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
        _write_text(
            case_dir / "00_case_summary.json",
            json.dumps(
                {
                    "case_id": case_id,
                    "abm": "grazing",
                    "evidence_mode": evidence_mode,
                    "prompt_variant": prompt_variant,
                    "repetition": 1,
                    "model": "nvidia/nemotron-nano-12b-v2-vl:free",
                }
            ),
        )
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


def test_render_optimal_tables_preserve_expected_headers_and_rows() -> None:
    rows = [
        {
            "Reference family": "author",
            "ABM": "grazing",
            "Summary": "bart",
            "LLM": "mistral-medium-latest",
            "BLEU": "0.12",
            "METEOR": "0.34",
            "R-1": "0.56",
            "R-2": "0.22",
            "R-L": "0.44",
            "Reading ease": "61.00",
        }
    ]

    markdown = _render_optimal_markdown_table(rows)
    latex = _render_optimal_latex_table(rows)

    assert "| Reference family | ABM | Summary | LLM | BLEU | METEOR | R-1 | R-2 | R-L | Reading ease |" in markdown
    assert "| author | grazing | bart | mistral-medium-latest | 0.12 | 0.34 | 0.56 | 0.22 | 0.44 | 61.00 |" in markdown
    assert "\\textit{\\textbf{Reference family}}" in latex
    assert "author & grazing & bart & mistral-medium-latest" in latex


def test_render_factorial_latex_table_bolds_large_contributions() -> None:
    frame = pd.DataFrame(
        [
            {
                "Feature": "Summarizer_AND_Evidence",
                "BLEU": 6.25,
                "METEOR": 4.0,
                "R-1": 5.5,
                "R-2": 1.0,
                "R-L": 0.5,
                "Reading ease": 10.0,
            }
        ]
    )

    latex = _render_factorial_latex_table(frame)

    assert "Summarizer and Evidence" in latex
    assert "\\textbf{6.25}" in latex
    assert "\\textbf{5.50}" in latex
    assert "\\textbf{10.00}" in latex
    assert "4.00" in latex


def test_render_overview_factorial_table_marks_absent_terms_and_tiny_nonzero_values() -> None:
    frame = pd.DataFrame(
        [
            {
                "Reference family": "author",
                "Feature": "Evidence",
                "BLEU": 0.004,
                "METEOR": 0.0,
                "R-1": 0.5,
                "R-2": 1.0,
                "R-L": 2.0,
                "Reading ease": 3.0,
            },
            {
                "Reference family": "author",
                "Feature": "Role_AND_Insight",
                "BLEU": float("nan"),
                "METEOR": float("nan"),
                "R-1": float("nan"),
                "R-2": float("nan"),
                "R-L": float("nan"),
                "Reading ease": float("nan"),
            },
        ]
    )

    table = _render_overview_factorial_markdown_table(frame)

    assert "<0.01" in table
    assert "| author | Role_AND_Insight | — | — | — | — | — | — |" in table


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
    assert result.optimal_csv_path.exists() is True
    assert result.anova_table_markdown_path.exists() is True
    assert result.factorial_table_markdown_path.exists() is True
    assert result.optimal_table_markdown_path.exists() is True
    assert result.anova_csv_path.parent.name == "combined"
    assert result.factorial_csv_path.parent.name == "combined"
    assert result.optimal_csv_path.parent.name == "combined"
    assert (result.run_root / "author").exists() is True
    assert (result.run_root / "gpt5.2_short").exists() is True
    assert (result.run_root / "gpt5.2_long").exists() is True
    assert (result.run_root / "overview").exists() is True
    assert sorted(path.name for path in result.overview_root.iterdir()) == [
        "anova_table.md",
        "best_scores_table.md",
        "factorial_table.md",
    ]
    rows = list(csv.DictReader(result.quantitative_rows_path.open(encoding="utf-8")))
    assert len(rows) == 60
    assert {row["reference_family"] for row in rows} == {"author", "gpt5.2_short", "gpt5.2_long"}
    assert rows[0]["evidence"]
    assert rows[0]["prompt"]
    assert rows[0]["summarizer"]
    report = json.loads(result.report_json_path.read_text(encoding="utf-8"))
    assert report["success"] is True


def test_run_quantitative_smoke_writes_best_score_table(tmp_path: Path) -> None:
    summarizer_root, _matrix_root = _build_source_roots(tmp_path)
    result = run_quantitative_smoke(
        source_root=summarizer_root,
        output_root=tmp_path / "quant",
        score_summary_fn=_fake_score_summary,
    )

    rows = list(csv.DictReader(result.optimal_csv_path.open(encoding="utf-8")))
    assert rows
    assert {"Reference family", "ABM", "Summary", "LLM", "BLEU", "METEOR", "R-1", "R-2", "R-L", "Reading ease"} == set(
        rows[0].keys()
    )
    assert {row["Reference family"] for row in rows} == {"author", "gpt5.2_short", "gpt5.2_long"}
    assert {row["Summary"] for row in rows} == {"none", "bart", "bert", "t5", "longformer_ext"}
    assert {row["LLM"] for row in rows} == {"nvidia/nemotron-nano-12b-v2-vl:free"}


def test_build_structured_results_rows_exposes_modern_factor_sheet() -> None:
    rows = _build_structured_results_rows(
        [
            {
                "case_id": "01_grazing_role_plot_plus_table_rep2",
                "abm": "grazing",
                "reference_family": "author",
                "llm": "mistral-medium-latest",
                "evidence": "plot+table",
                "prompt": "role+insights",
                "role": "True",
                "insights": "True",
                "example": "False",
                "summarizer": "t5",
                "repetition": "2",
                "summary_output_path": "/tmp/out.txt",
                "BLEU": "0.10",
                "METEOR": "0.20",
                "R-1": "0.30",
                "R-2": "0.40",
                "R-L": "0.50",
                "Reading ease": "60.00",
            }
        ]
    )

    assert rows == [
        {
            "Case study": "grazing",
            "Reference family": "author",
            "Summary": "t5",
            "LLM": "mistral-medium-latest",
            "Role": "Yes",
            "Example": "No",
            "Insight": "Yes",
            "Evidence": "plot+table",
            "Repetition": "2",
            "Output": "/tmp/out.txt",
            "BLEU": "0.10",
            "METEOR": "0.20",
            "ROUGE-1": "0.30",
            "ROUGE-2": "0.40",
            "ROUGE-L": "0.50",
            "Flesch Reading Ease": "60.00",
        }
    ]


def test_best_score_table_optimizes_each_metric_independently(tmp_path: Path) -> None:
    summarizer_root, _matrix_root = _build_source_roots(tmp_path)

    def fake_score(reference: str, candidate: str) -> SummaryScores:
        _ = reference
        score = 0.1
        reading = 10.0
        if "role table" in candidate:
            score = 0.9
        if "insights plot+table" in candidate:
            reading = 91.0
        return SummaryScores(
            token_f1=score,
            precision=score,
            recall=score,
            bleu=score,
            meteor=score,
            rouge1=score,
            rouge2=score,
            rouge_l=score,
            flesch_reading_ease=reading,
            reference_length=10,
            candidate_length=10,
        )

    result = run_quantitative_smoke(
        source_root=summarizer_root,
        output_root=tmp_path / "quant",
        score_summary_fn=fake_score,
    )
    rows = list(csv.DictReader(result.optimal_csv_path.open(encoding="utf-8")))
    none_row = next(row for row in rows if row["Summary"] == "none")
    assert none_row["BLEU"] == "0.90"
    assert none_row["Reading ease"] == "91.00"


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


def test_normalize_factorial_table_scales_each_metric_to_full_share() -> None:
    frame = _normalize_factorial_table(
        pd.DataFrame(
            [
                {
                    "Feature": "Summarizer",
                    "BLEU": 30.0,
                    "METEOR": 10.0,
                    "R-1": 20.0,
                    "R-2": 4.0,
                    "R-L": 6.0,
                    "Reading ease": 1.0,
                },
                {
                    "Feature": "Evidence",
                    "BLEU": 10.0,
                    "METEOR": 20.0,
                    "R-1": 5.0,
                    "R-2": 6.0,
                    "R-L": 4.0,
                    "Reading ease": 3.0,
                },
            ]
        )
    )
    for metric in ("BLEU", "METEOR", "R-1", "R-2", "R-L", "Reading ease"):
        assert round(float(frame[metric].sum()), 6) == 100.0


def test_normalize_factorial_table_marks_missing_features_absent_instead_of_zero() -> None:
    frame = _normalize_factorial_table(
        pd.DataFrame(
            [
                {
                    "Feature": "Summarizer",
                    "BLEU": 30.0,
                    "METEOR": 10.0,
                    "R-1": 20.0,
                    "R-2": 4.0,
                    "R-L": 6.0,
                    "Reading ease": 1.0,
                }
            ]
        )
    )

    missing_row = frame.loc[frame["Feature"] == "Role_AND_Insights"].iloc[0]
    assert pd.isna(missing_row["BLEU"])
    assert pd.isna(missing_row["Reading ease"])


def test_normalize_factorial_table_canonicalizes_insight_and_reversed_interaction_names() -> None:
    frame = _normalize_factorial_table(
        pd.DataFrame(
            [
                {
                    "Feature": "Example_AND_Insight",
                    "BLEU": 1.0,
                    "METEOR": 2.0,
                    "R-1": 3.0,
                    "R-2": 4.0,
                    "R-L": 5.0,
                    "Reading ease": 6.0,
                },
                {
                    "Feature": "Insights_AND_Example",
                    "BLEU": 1.0,
                    "METEOR": 2.0,
                    "R-1": 3.0,
                    "R-2": 4.0,
                    "R-L": 5.0,
                    "Reading ease": 6.0,
                },
                {
                    "Feature": "Insight",
                    "BLEU": 1.0,
                    "METEOR": 1.0,
                    "R-1": 1.0,
                    "R-2": 1.0,
                    "R-L": 1.0,
                    "Reading ease": 1.0,
                },
            ]
        )
    )

    assert "Example_AND_Insight" not in set(frame["Feature"])
    assert "Insight" not in set(frame["Feature"])
    assert "Insights_AND_Example" in set(frame["Feature"])
    assert "Insights" in set(frame["Feature"])


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


def test_run_quantitative_smoke_includes_modeler_only_for_supported_abms(tmp_path: Path) -> None:
    matrix_root = tmp_path / "matrix" / "runs" / "run_1"
    summarizer_root = tmp_path / "summ" / "runs" / "run_1"
    review_rows: list[dict[str, str]] = []

    for abm in ("grazing", "milk_consumption"):
        case_id = f"01_{abm}_none_plot_rep1"
        case_dir = matrix_root / "cases" / case_id
        context_output = _write_text(case_dir / "02_context" / "context_output.txt", f"context {case_id}")
        _write_text(
            case_dir / "00_case_summary.json",
            json.dumps(
                {
                    "case_id": case_id,
                    "abm": abm,
                    "evidence_mode": "plot",
                    "prompt_variant": "none",
                    "repetition": 1,
                    "model": "mistral-medium-latest",
                }
            ),
        )
        _write_text(case_dir / "03_trends" / "plot_01" / "trend_output.txt", f"trend {case_id}")
        bundle_dir = summarizer_root / "bundles" / case_id
        combined_input = _write_text(bundle_dir / "01_input" / "combined_input.txt", f"combined {case_id}")
        summary_path = _write_text(bundle_dir / "02_summaries" / "none.txt", f"summary {case_id}")
        review_rows.append(
            {
                "bundle_id": case_id,
                "case_id": case_id,
                "abm": abm,
                "mode": "none",
                "success": "True",
                "context_output_path": str(context_output),
                "trend_output_paths": str(case_dir / "03_trends" / "plot_01" / "trend_output.txt"),
                "combined_input_path": str(combined_input),
                "summary_output_path": str(summary_path),
                "input_length": "100",
                "output_length": "80",
                "duration_seconds": "0.1",
                "validation_note": "validated",
                "error": "",
            }
        )

    (summarizer_root / "review.csv").parent.mkdir(parents=True, exist_ok=True)
    with (summarizer_root / "review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0].keys()))
        writer.writeheader()
        writer.writerows(review_rows)

    result = run_quantitative_smoke(
        source_root=summarizer_root,
        output_root=tmp_path / "quant",
        score_summary_fn=_fake_score_summary,
    )

    rows = list(csv.DictReader(result.quantitative_rows_path.open(encoding="utf-8")))
    by_abm: dict[str, set[str]] = {}
    for row in rows:
        by_abm.setdefault(row["abm"], set()).add(row["reference_family"])

    assert by_abm["grazing"] == {"author", "gpt5.2_short", "gpt5.2_long"}
    assert by_abm["milk_consumption"] == {"author", "modeler", "gpt5.2_short", "gpt5.2_long"}
