from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

import distill_abm.pipeline.quantitative_smoke as quantitative_smoke_module
from distill_abm.eval.metrics import SummaryScores
from distill_abm.pipeline.quantitative_smoke import (
    _build_evidence_summary_rows,
    _build_factorial_input_frame,
    _build_factorial_input_frame_with_llm,
    _build_structured_results_rows,
    _derive_prompt_flags,
    _normalize_factorial_table,
    _render_anova_markdown_table,
    _render_evidence_summary_markdown_table,
    _render_factorial_latex_table,
    _render_optimal_latex_table,
    _render_optimal_markdown_table,
    _render_overview_factorial_markdown_table,
    run_quantitative_smoke,
    run_quantitative_smoke_multi_llm,
)


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _build_source_roots(
    tmp_path: Path,
    *,
    root_name: str = "summ",
    matrix_name: str = "matrix",
    model: str = "nvidia/nemotron-nano-12b-v2-vl:free",
) -> tuple[Path, Path]:
    matrix_root = tmp_path / matrix_name / "runs" / "run_1"
    summarizer_root = tmp_path / root_name / "runs" / "run_1"
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
                    "model": model,
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


def _build_quantitative_root(
    tmp_path: Path,
    *,
    root_name: str,
    matrix_name: str,
    model: str,
    output_name: str,
) -> Path:
    summarizer_root, _ = _build_source_roots(
        tmp_path,
        root_name=root_name,
        matrix_name=matrix_name,
        model=model,
    )
    output_root = tmp_path / output_name
    result = run_quantitative_smoke(
        source_root=summarizer_root,
        output_root=output_root,
        score_summary_fn=_fake_score_summary,
    )
    assert result.success is True
    return output_root


def _write_prompt_compression_summary(matrix_root: Path, *, final_tiers: list[int]) -> None:
    entries = []
    total_compressions = 0
    triggered_entries = 0
    for index, final_tier in enumerate(final_tiers, start=1):
        case_id = f"{index:02d}_grazing_none_plot_rep1"
        artifacts_dir = matrix_root / "cases" / case_id / "03_trends" / f"plot_{index:02d}"
        attempts = [
            {
                "attempt_index": tier + 1,
                "table_downsample_stride": tier + 1,
                "compression_tier": tier,
                "prompt_length": max(1000 - (tier * 100), 100),
            }
            for tier in range(final_tier + 1)
        ]
        if final_tier > 0:
            triggered_entries += 1
            total_compressions += final_tier
        entries.append(
            {
                "source_run_root": str(matrix_root),
                "scope": "full_case_matrix_plot",
                "case_id": case_id,
                "abm": "grazing",
                "evidence_mode": "plot",
                "prompt_variant": "none",
                "repetition": 1,
                "plot_index": index,
                "artifacts_dir": str(artifacts_dir),
                "compression_artifact_path": str(artifacts_dir / "trend_prompt_compression.json"),
                "pre_compression_prompt_path": None,
                "compressed_prompt_path": None,
                "triggered": final_tier > 0,
                "compression_count": final_tier,
                "attempt_count": len(attempts),
                "attempts": attempts,
            }
        )
    (matrix_root / "prompt_compression_summary.json").write_text(
        json.dumps(
            {
                "run_root": str(matrix_root),
                "total_entries": len(entries),
                "triggered_entries": triggered_entries,
                "total_compressions": total_compressions,
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )


def test_load_case_metadata_from_context_output_resolves_wrapper_backed_results_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    missing_context_output = Path(
        "results/qwen3.5-27b_openrouter_all_abms_chain/04_generation_smoke_latest/abms/"
        "grazing/runs/run_20260312_123500_479488/cases/01_grazing_none_plot_rep1/02_context/context_output.txt"
    )
    wrapper_context_output = (
        tmp_path
        / "results"
        / "screening"
        / "qwen3.5-27b_openrouter_all_abms_chain"
        / "04_generation_smoke_latest"
        / "abms"
        / "grazing"
        / "runs"
        / "run_20260312_123500_479488"
        / "cases"
        / "01_grazing_none_plot_rep1"
        / "02_context"
        / "context_output.txt"
    )
    _write_text(wrapper_context_output, "context")
    _write_text(
        wrapper_context_output.parent.parent.parent.parent.parent / "current" / "smoke_full_case_matrix_report.json",
        json.dumps(
            {
                "cases": [
                    {
                        "case_dir": str(
                            Path(
                                "results/qwen3.5-27b_openrouter_all_abms_chain/04_generation_smoke_latest/abms/"
                                "grazing/runs/run_20260312_123500_479488/cases/01_grazing_none_plot_rep1"
                            )
                        ),
                        "case_id": "01_grazing_none_plot_rep1",
                        "abm": "grazing",
                        "evidence_mode": "plot",
                        "prompt_variant": "none",
                        "repetition": 1,
                        "model_id": "qwen/qwen3.5-27b",
                    }
                ],
                "model_id": "qwen/qwen3.5-27b",
            }
        ),
    )

    metadata = quantitative_smoke_module._load_case_metadata_from_context_output(missing_context_output)

    assert metadata["evidence_mode"] == "plot"
    assert metadata["prompt_variant"] == "none"
    assert metadata["repetition"] == 1
    assert metadata["model_id"] == "qwen/qwen3.5-27b"
    assert metadata["run_root"] == Path(
        "results/screening/qwen3.5-27b_openrouter_all_abms_chain/04_generation_smoke_latest/abms/"
        "grazing/runs/run_20260312_123500_479488"
    )


def test_load_quantitative_source_records_resolves_wrapper_backed_summary_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source_root = (
        tmp_path
        / "results"
        / "screening"
        / "qwen3.5-27b_openrouter_all_abms_chain"
        / "05_summarizer_smoke_latest"
        / "runs"
        / "run_20260312_153523_845549"
    )
    case_id = "01_grazing_none_plot_rep1"
    _write_text(
        source_root / "review.csv",
        "\n".join(
            [
                "bundle_id,case_id,abm,mode,success,context_output_path,summary_output_path",
                ",".join(
                    [
                        case_id,
                        case_id,
                        "grazing",
                        "bart",
                        "True",
                        "results/qwen3.5-27b_openrouter_all_abms_chain/04_generation_smoke_latest/abms/"
                        "grazing/runs/run_20260312_123500_479488/cases/01_grazing_none_plot_rep1/02_context/context_output.txt",
                        "results/qwen3.5-27b_openrouter_all_abms_chain/05_summarizer_smoke_latest/runs/"
                        "run_20260312_153523_845549/bundles/01_grazing_none_plot_rep1/02_summaries/bart.txt",
                    ]
                ),
            ]
        )
        + "\n",
    )
    _write_text(
        source_root
        / "bundles"
        / case_id
        / "02_summaries"
        / "bart.txt",
        "summary",
    )
    _write_text(
        tmp_path
        / "results"
        / "screening"
        / "qwen3.5-27b_openrouter_all_abms_chain"
        / "04_generation_smoke_latest"
        / "abms"
        / "grazing"
        / "runs"
        / "run_20260312_123500_479488"
        / "cases"
        / case_id
        / "00_case_summary.json",
        json.dumps(
            {
                "case_id": case_id,
                "abm": "grazing",
                "evidence_mode": "plot",
                "prompt_variant": "none",
                "repetition": 1,
                "model_id": "qwen/qwen3.5-27b",
                "model": "qwen/qwen3.5-27b",
            }
        ),
    )
    _write_text(
        tmp_path
        / "results"
        / "screening"
        / "qwen3.5-27b_openrouter_all_abms_chain"
        / "04_generation_smoke_latest"
        / "abms"
        / "grazing"
        / "runs"
        / "run_20260312_123500_479488"
        / "cases"
        / case_id
        / "02_context"
        / "context_output.txt",
        "context",
    )

    monkeypatch.setattr(
        quantitative_smoke_module,
        "resolve_quantitative_reference_paths",
        lambda _abm: {
            "author": Path("author.txt"),
            "modeler": Path("modeler.txt"),
            "gpt5.2_short": Path("gpt5.2_short.txt"),
            "gpt5.2_long": Path("gpt5.2_long.txt"),
        },
    )
    records = quantitative_smoke_module._load_quantitative_source_records(
        (source_root,),
        require_source_llm_label=False,
        include_llm_in_record_id=False,
    )

    assert len(records) == 4
    assert all(
        record.record.summary_output_path
        == Path(
            "results/screening/qwen3.5-27b_openrouter_all_abms_chain/05_summarizer_smoke_latest/runs/"
            "run_20260312_153523_845549/bundles/01_grazing_none_plot_rep1/02_summaries/bart.txt"
        )
        for record in records
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


def test_build_evidence_summary_rows_aggregates_average_and_best_scores() -> None:
    rows = _build_evidence_summary_rows(
        [
            {
                "reference_family": "author",
                "abm": "grazing",
                "evidence": "plot",
                "BLEU": "0.10",
                "METEOR": "0.20",
                "R-1": "0.30",
                "R-2": "0.40",
                "R-L": "0.50",
                "Reading ease": "60.00",
            },
            {
                "reference_family": "author",
                "abm": "grazing",
                "evidence": "plot",
                "BLEU": "0.50",
                "METEOR": "0.60",
                "R-1": "0.70",
                "R-2": "0.80",
                "R-L": "0.90",
                "Reading ease": "80.00",
            },
            {
                "reference_family": "author",
                "abm": "grazing",
                "evidence": "table",
                "BLEU": "0.20",
                "METEOR": "0.30",
                "R-1": "0.40",
                "R-2": "0.50",
                "R-L": "0.60",
                "Reading ease": "70.00",
            },
            {
                "reference_family": "gpt5.2_short",
                "abm": "fauna",
                "evidence": "plot+table",
                "BLEU": "0.25",
                "METEOR": "0.35",
                "R-1": "0.45",
                "R-2": "0.55",
                "R-L": "0.65",
                "Reading ease": "75.00",
            },
        ]
    )

    assert rows == [
        {
            "Reference family": "author",
            "Evidence": "plot",
            "ABM": "grazing",
            "Avg BLEU": "0.30",
            "Avg METEOR": "0.40",
            "Avg R-1": "0.50",
            "Avg R-2": "0.60",
            "Avg R-L": "0.70",
            "Avg Reading ease": "70.00",
            "Best BLEU": "0.50",
            "Best METEOR": "0.60",
            "Best R-1": "0.70",
            "Best R-2": "0.80",
            "Best R-L": "0.90",
            "Best Reading ease": "80.00",
        },
        {
            "Reference family": "author",
            "Evidence": "table",
            "ABM": "grazing",
            "Avg BLEU": "0.20",
            "Avg METEOR": "0.30",
            "Avg R-1": "0.40",
            "Avg R-2": "0.50",
            "Avg R-L": "0.60",
            "Avg Reading ease": "70.00",
            "Best BLEU": "0.20",
            "Best METEOR": "0.30",
            "Best R-1": "0.40",
            "Best R-2": "0.50",
            "Best R-L": "0.60",
            "Best Reading ease": "70.00",
        },
        {
            "Reference family": "gpt5.2_short",
            "Evidence": "plot+table",
            "ABM": "fauna",
            "Avg BLEU": "0.25",
            "Avg METEOR": "0.35",
            "Avg R-1": "0.45",
            "Avg R-2": "0.55",
            "Avg R-L": "0.65",
            "Avg Reading ease": "75.00",
            "Best BLEU": "0.25",
            "Best METEOR": "0.35",
            "Best R-1": "0.45",
            "Best R-2": "0.55",
            "Best R-L": "0.65",
            "Best Reading ease": "75.00",
        },
    ]


def test_render_evidence_summary_markdown_table_groups_rows_by_reference_family() -> None:
    markdown = _render_evidence_summary_markdown_table(
        [
            {
                "Reference family": "author",
                "Evidence": "plot",
                "ABM": "grazing",
                "Avg BLEU": "0.30",
                "Avg METEOR": "0.40",
                "Avg R-1": "0.50",
                "Avg R-2": "0.60",
                "Avg R-L": "0.70",
                "Avg Reading ease": "70.00",
                "Best BLEU": "0.50",
                "Best METEOR": "0.60",
                "Best R-1": "0.70",
                "Best R-2": "0.80",
                "Best R-L": "0.90",
                "Best Reading ease": "80.00",
            },
            {
                "Reference family": "gpt5.2_short",
                "Evidence": "plot+table",
                "ABM": "fauna",
                "Avg BLEU": "0.25",
                "Avg METEOR": "0.35",
                "Avg R-1": "0.45",
                "Avg R-2": "0.55",
                "Avg R-L": "0.65",
                "Avg Reading ease": "75.00",
                "Best BLEU": "0.25",
                "Best METEOR": "0.35",
                "Best R-1": "0.45",
                "Best R-2": "0.55",
                "Best R-L": "0.65",
                "Best Reading ease": "75.00",
            },
        ]
    )

    assert markdown.startswith("# Evidence Mode Summary\n\n## author\n\n| Evidence | ABM | Avg BLEU | Avg METEOR |")
    assert (
        "| plot | grazing | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 70.00 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 80.00 |"
        in markdown
    )
    assert "\n## gpt5.2_short\n\n| Evidence | ABM | Avg BLEU | Avg METEOR |" in markdown


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
    assert result.evidence_summary_table_markdown_path.exists() is True
    assert result.factorial_table_markdown_path.exists() is True
    assert result.optimal_table_markdown_path.exists() is True
    assert result.anova_csv_path.parent.name == "combined"
    assert result.factorial_csv_path.parent.name == "combined"
    assert result.optimal_csv_path.parent.name == "combined"
    assert (result.run_root / "author").exists() is True
    assert (result.run_root / "modeler").exists() is True
    assert (result.run_root / "gpt5.2_short").exists() is True
    assert (result.run_root / "gpt5.2_long").exists() is True
    assert (result.run_root / "overview").exists() is True
    assert sorted(path.name for path in result.overview_root.iterdir()) == [
        "anova_table.md",
        "best_scores_table.md",
        "evidence_summary_table.md",
        "factorial_table.md",
    ]
    rows = list(csv.DictReader(result.quantitative_rows_path.open(encoding="utf-8")))
    assert len(rows) == 80
    assert {row["reference_family"] for row in rows} == {"author", "modeler", "gpt5.2_short", "gpt5.2_long"}
    assert rows[0]["evidence"]
    assert rows[0]["prompt"]
    assert rows[0]["summarizer"]
    report = json.loads(result.report_json_path.read_text(encoding="utf-8"))
    assert report["success"] is True
    assert result.prompt_compression_summary_markdown_path is None


def test_run_quantitative_smoke_single_llm_accepts_unlabeled_runs(tmp_path: Path) -> None:
    summarizer_root, _matrix_root = _build_source_roots(tmp_path, model="")

    result = run_quantitative_smoke(
        source_root=summarizer_root,
        output_root=tmp_path / "quant",
        score_summary_fn=_fake_score_summary,
    )

    rows = list(csv.DictReader(result.quantitative_rows_path.open(encoding="utf-8")))
    assert result.success is True
    assert rows
    assert {row["llm"] for row in rows} == {""}


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
    assert {row["Reference family"] for row in rows} == {"author", "modeler", "gpt5.2_short", "gpt5.2_long"}
    assert {row["Summary"] for row in rows if row["Reference family"] == "author"} == {
        "bart",
        "bert",
        "t5",
        "longformer_ext",
    }
    assert {row["Summary"] for row in rows if row["Reference family"] == "modeler"} == {
        "bart",
        "bert",
        "t5",
        "longformer_ext",
    }
    assert {row["Summary"] for row in rows if row["Reference family"] == "gpt5.2_short"} == {
        "bart",
        "bert",
        "t5",
        "longformer_ext",
    }
    assert {row["Summary"] for row in rows if row["Reference family"] == "gpt5.2_long"} == {"none"}
    assert {row["LLM"] for row in rows} == {"nvidia/nemotron-nano-12b-v2-vl:free"}


def test_run_quantitative_smoke_filters_evaluation_outputs_by_reference_compatibility(tmp_path: Path) -> None:
    summarizer_root, _matrix_root = _build_source_roots(tmp_path)
    result = run_quantitative_smoke(
        source_root=summarizer_root,
        output_root=tmp_path / "quant",
        score_summary_fn=_fake_score_summary,
    )

    raw_rows = list(csv.DictReader(result.quantitative_rows_path.open(encoding="utf-8")))
    best_score_rows = list(csv.DictReader(result.optimal_csv_path.open(encoding="utf-8")))
    factorial_rows = list(
        csv.DictReader((result.factorial_csv_path.parent / "factorial_input.csv").open(encoding="utf-8"))
    )
    long_factorial_rows = list(
        csv.DictReader((result.run_root / "gpt5.2_long" / "factorial_input.csv").open(encoding="utf-8"))
    )
    anova_markdown = result.anova_table_markdown_path.read_text(encoding="utf-8")
    overview_factorial_markdown = result.factorial_table_markdown_path.read_text(encoding="utf-8")

    assert len(raw_rows) == 80
    assert {row["Summary"] for row in best_score_rows if row["Reference family"] == "author"} == {
        "bart",
        "bert",
        "t5",
        "longformer_ext",
    }
    assert {row["Summary"] for row in best_score_rows if row["Reference family"] == "modeler"} == {
        "bart",
        "bert",
        "t5",
        "longformer_ext",
    }
    assert {row["Summary"] for row in best_score_rows if row["Reference family"] == "gpt5.2_short"} == {
        "bart",
        "bert",
        "t5",
        "longformer_ext",
    }
    assert {row["Summary"] for row in best_score_rows if row["Reference family"] == "gpt5.2_long"} == {"none"}
    assert {row["Summarizer"] for row in factorial_rows if row["Summarizer"]} == {
        "bart",
        "bert",
        "t5",
        "longformer_ext",
    }
    assert any(not row["Summarizer"] for row in factorial_rows)
    assert long_factorial_rows
    assert "Summarizer" not in long_factorial_rows[0]
    assert "modeler" in anova_markdown
    assert "modeler" in overview_factorial_markdown
    assert "gpt5.2_long" in anova_markdown
    assert "gpt5.2_long" in overview_factorial_markdown


def test_run_quantitative_smoke_writes_global_master_overview_for_results_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    reference_dir = tmp_path / "refs"
    reference_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        quantitative_smoke_module,
        "resolve_quantitative_reference_paths",
        lambda abm: {
            "author": _write_text(reference_dir / f"{abm}_author.txt", "author"),
            "gpt5.2_short": _write_text(reference_dir / f"{abm}_short.txt", "short"),
            "gpt5.2_long": _write_text(reference_dir / f"{abm}_long.txt", "long"),
        },
    )
    qwen_root, _ = _build_source_roots(
        tmp_path,
        root_name="summ_qwen",
        matrix_name="matrix_qwen",
        model="qwen/qwen3.5-27b",
    )
    mistral_root, _ = _build_source_roots(
        tmp_path,
        root_name="summ_mistral",
        matrix_name="matrix_mistral",
        model="mistral-medium-latest",
    )

    qwen_result = run_quantitative_smoke(
        source_root=qwen_root,
        output_root=tmp_path / "results" / "qwen3.5-27b_openrouter_all_abms_chain" / "06_quantitative_smoke_latest",
        score_summary_fn=_fake_score_summary,
    )
    mistral_result = run_quantitative_smoke(
        source_root=mistral_root,
        output_root=tmp_path / "results" / "mistral-medium-latest_all_abms_chain" / "06_quantitative_smoke_latest",
        score_summary_fn=_fake_score_summary,
    )
    multi_result = run_quantitative_smoke_multi_llm(
        source_roots=(qwen_result.output_root, mistral_result.output_root),
        output_root=tmp_path / "results" / "eval_qwen_mistral",
        score_summary_fn=_fake_score_summary,
    )

    master_root = tmp_path / "results" / "quantitative_master_overview"
    assert master_root.exists() is True
    assert sorted(path.name for path in master_root.iterdir()) == [
        "anova_table.md",
        "best_scores_table.md",
        "evidence_summary_table.md",
        "factorial_table.md",
        "prompt_compression_summary.md",
    ]
    anova_text = (master_root / "anova_table.md").read_text(encoding="utf-8")
    factorial_text = (master_root / "factorial_table.md").read_text(encoding="utf-8")
    assert "qwen3.5-27b_openrouter_all_abms_chain/06_quantitative_smoke_latest" in anova_text
    assert "eval_qwen_mistral" in anova_text
    assert "gpt5.2_long" in factorial_text
    assert "eval_qwen_mistral" in factorial_text
    assert multi_result.success is True


def test_run_quantitative_smoke_writes_prompt_compression_summary_when_metadata_exists(tmp_path: Path) -> None:
    summarizer_root, matrix_root = _build_source_roots(tmp_path)
    _write_prompt_compression_summary(matrix_root, final_tiers=[0, 1, 2])

    result = run_quantitative_smoke(
        source_root=summarizer_root,
        output_root=tmp_path / "quant",
        score_summary_fn=_fake_score_summary,
    )

    assert result.prompt_compression_summary_markdown_path is not None
    summary_text = result.prompt_compression_summary_markdown_path.read_text(encoding="utf-8")
    assert "- prompt_compression_triggered: `true`" in summary_text
    assert "- prompts_with_compression: `2`" in summary_text
    assert "- total_compression_steps: `3`" in summary_text
    assert '- compression_tiers_used: `[1, 2]`' in summary_text
    assert '- compression_steps_by_tier: `{"1": 2, "2": 1}`' in summary_text
    assert '- prompts_by_final_tier: `{"1": 1, "2": 1}`' in summary_text


def test_run_quantitative_smoke_writes_prompt_compression_summary_for_zero_trigger_case(tmp_path: Path) -> None:
    summarizer_root, matrix_root = _build_source_roots(tmp_path)
    _write_prompt_compression_summary(matrix_root, final_tiers=[0, 0])

    result = run_quantitative_smoke(
        source_root=summarizer_root,
        output_root=tmp_path / "quant",
        score_summary_fn=_fake_score_summary,
    )

    assert result.prompt_compression_summary_markdown_path is not None
    summary_text = result.prompt_compression_summary_markdown_path.read_text(encoding="utf-8")
    assert "- prompt_compression_triggered: `false`" in summary_text
    assert "- prompts_with_compression: `0`" in summary_text
    assert "- total_compression_steps: `0`" in summary_text
    assert "- compression_tiers_used: `[]`" in summary_text
    assert "- compression_steps_by_tier: `{}`" in summary_text


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
    none_row = next(row for row in rows if row["Summary"] == "none" and row["Reference family"] == "gpt5.2_long")
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


def test_build_factorial_input_frame_with_llm_includes_llm_factor() -> None:
    frame = _build_factorial_input_frame_with_llm(
        [
            {
                "llm": "mistral-medium-latest",
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
    assert frame.loc[0, "LLM"] == "mistral-medium-latest"
    assert frame.loc[0, "Role"] == "on"
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


def test_run_quantitative_smoke_resume_scores_only_new_reference_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summarizer_root, _matrix_root = _build_source_roots(tmp_path)
    modeler_reference = _write_text(tmp_path / "modeler_reference.txt", "modeler reference")

    score_calls: list[tuple[str, str]] = []

    def _counting_score_summary(reference: str, candidate: str) -> SummaryScores:
        score_calls.append((reference, candidate))
        return _fake_score_summary(reference, candidate)

    monkeypatch.setattr(
        quantitative_smoke_module,
        "resolve_quantitative_reference_paths",
        lambda abm: {"author": Path("data/summaries/authors/grazing_scoring_ground_truth.txt")},
    )
    first = run_quantitative_smoke(
        source_root=summarizer_root,
        output_root=tmp_path / "quant",
        score_summary_fn=_counting_score_summary,
    )

    assert first.success is True
    assert len(score_calls) == 20

    score_calls.clear()
    monkeypatch.setattr(
        quantitative_smoke_module,
        "resolve_quantitative_reference_paths",
        lambda abm: {
            "author": Path("data/summaries/authors/grazing_scoring_ground_truth.txt"),
            "modeler": modeler_reference,
        },
    )
    resumed = run_quantitative_smoke(
        source_root=summarizer_root,
        output_root=tmp_path / "quant",
        resume=True,
        score_summary_fn=_counting_score_summary,
    )

    assert resumed.success is True
    assert resumed.run_root != first.run_root
    assert len(score_calls) == 20
    rows = list(csv.DictReader(resumed.quantitative_rows_path.open(encoding="utf-8")))
    assert {row["reference_family"] for row in rows} == {"author", "modeler"}
    assert len(rows) == 40


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

    assert by_abm["grazing"] == {"author", "modeler", "gpt5.2_short", "gpt5.2_long"}
    assert by_abm["milk_consumption"] == {"author", "modeler", "gpt5.2_short", "gpt5.2_long"}


def test_run_quantitative_smoke_multi_llm_merges_sources_without_record_id_collisions(tmp_path: Path) -> None:
    mistral_root, _ = _build_source_roots(
        tmp_path,
        root_name="summ_mistral",
        matrix_name="matrix_mistral",
        model="mistral-medium-latest",
    )
    qwen_root, _ = _build_source_roots(
        tmp_path,
        root_name="summ_qwen",
        matrix_name="matrix_qwen",
        model="qwen/qwen3.5-27b",
    )

    result = run_quantitative_smoke_multi_llm(
        source_roots=(mistral_root, qwen_root),
        output_root=tmp_path / "quant_multi",
        score_summary_fn=_fake_score_summary,
    )

    rows = list(csv.DictReader(result.quantitative_rows_path.open(encoding="utf-8")))
    assert result.success is True
    assert len(rows) == 160
    assert {row["llm"] for row in rows} == {"mistral-medium-latest", "qwen/qwen3.5-27b"}
    assert len({row["record_id"] for row in rows}) == len(rows)
    assert all("__llm_" in row["record_id"] for row in rows)

    factorial_input_path = result.factorial_csv_path.parent / "factorial_input.csv"
    factorial_rows = list(csv.DictReader(factorial_input_path.open(encoding="utf-8")))
    assert "LLM" in factorial_rows[0]
    assert {row["LLM"] for row in factorial_rows} == {"mistral-medium-latest", "qwen/qwen3.5-27b"}

    assert "LLM" in result.anova_table_markdown_path.read_text(encoding="utf-8")


def test_run_quantitative_smoke_multi_llm_reuses_existing_single_llm_quantitative_results(tmp_path: Path) -> None:
    mistral_quant_root = _build_quantitative_root(
        tmp_path,
        root_name="summ_mistral",
        matrix_name="matrix_mistral",
        model="mistral-medium-latest",
        output_name="quant_mistral",
    )
    qwen_quant_root = _build_quantitative_root(
        tmp_path,
        root_name="summ_qwen",
        matrix_name="matrix_qwen",
        model="qwen/qwen3.5-27b",
        output_name="quant_qwen",
    )

    def _should_not_score(reference: str, candidate: str) -> SummaryScores:
        raise AssertionError("score_summary should not rerun for precomputed quantitative roots")

    result = run_quantitative_smoke_multi_llm(
        source_roots=(mistral_quant_root, qwen_quant_root),
        output_root=tmp_path / "quant_multi",
        score_summary_fn=_should_not_score,
    )

    rows = list(csv.DictReader(result.quantitative_rows_path.open(encoding="utf-8")))
    assert result.success is True
    assert len(rows) == 160
    assert {row["llm"] for row in rows} == {"mistral-medium-latest", "qwen/qwen3.5-27b"}


def test_run_quantitative_smoke_multi_llm_supports_mixed_quantitative_and_summarizer_sources(tmp_path: Path) -> None:
    mistral_quant_root = _build_quantitative_root(
        tmp_path,
        root_name="summ_mistral",
        matrix_name="matrix_mistral",
        model="mistral-medium-latest",
        output_name="quant_mistral",
    )
    qwen_summarizer_root, _ = _build_source_roots(
        tmp_path,
        root_name="summ_qwen",
        matrix_name="matrix_qwen",
        model="qwen/qwen3.5-27b",
    )
    score_calls = 0

    def _counting_score(reference: str, candidate: str) -> SummaryScores:
        nonlocal score_calls
        score_calls += 1
        return _fake_score_summary(reference, candidate)

    result = run_quantitative_smoke_multi_llm(
        source_roots=(mistral_quant_root, qwen_summarizer_root),
        output_root=tmp_path / "quant_multi",
        score_summary_fn=_counting_score,
    )

    rows = list(csv.DictReader(result.quantitative_rows_path.open(encoding="utf-8")))
    assert result.success is True
    assert len(rows) == 160
    assert score_calls == 80


def test_run_quantitative_smoke_multi_llm_keeps_distinct_ids_for_punctuation_variant_labels(tmp_path: Path) -> None:
    first_root, _ = _build_source_roots(
        tmp_path,
        root_name="summ_first",
        matrix_name="matrix_first",
        model="alpha/model-a",
    )
    second_root, _ = _build_source_roots(
        tmp_path,
        root_name="summ_second",
        matrix_name="matrix_second",
        model="alpha_model_a",
    )

    result = run_quantitative_smoke_multi_llm(
        source_roots=(first_root, second_root),
        output_root=tmp_path / "quant_multi",
        score_summary_fn=_fake_score_summary,
    )

    rows = list(csv.DictReader(result.quantitative_rows_path.open(encoding="utf-8")))
    assert result.success is True
    assert {row["llm"] for row in rows} == {"alpha/model-a", "alpha_model_a"}
    assert len({row["record_id"] for row in rows}) == len(rows)


def test_run_quantitative_smoke_multi_llm_rejects_duplicate_llm_labels(tmp_path: Path) -> None:
    first_root, _ = _build_source_roots(
        tmp_path,
        root_name="summ_first",
        matrix_name="matrix_first",
        model="mistral-medium-latest",
    )
    second_root, _ = _build_source_roots(
        tmp_path,
        root_name="summ_second",
        matrix_name="matrix_second",
        model="mistral-medium-latest",
    )

    with pytest.raises(ValueError, match="duplicate llm labels"):
        run_quantitative_smoke_multi_llm(
            source_roots=(first_root, second_root),
            output_root=tmp_path / "quant_multi",
            score_summary_fn=_fake_score_summary,
        )


def test_run_quantitative_smoke_multi_llm_rejects_missing_llm_labels(tmp_path: Path) -> None:
    first_root, _ = _build_source_roots(
        tmp_path,
        root_name="summ_first",
        matrix_name="matrix_first",
        model="",
    )
    second_root, _ = _build_source_roots(
        tmp_path,
        root_name="summ_second",
        matrix_name="matrix_second",
        model="qwen/qwen3.5-27b",
    )

    with pytest.raises(ValueError, match="missing llm label"):
        run_quantitative_smoke_multi_llm(
            source_roots=(first_root, second_root),
            output_root=tmp_path / "quant_multi",
            score_summary_fn=_fake_score_summary,
        )


def test_run_quantitative_smoke_multi_llm_rejects_quantitative_sources_with_multiple_llm_labels(tmp_path: Path) -> None:
    quant_root = _build_quantitative_root(
        tmp_path,
        root_name="summ_mistral",
        matrix_name="matrix_mistral",
        model="mistral-medium-latest",
        output_name="quant_mistral",
    )
    quant_run_root = Path((quant_root / "latest_run.txt").read_text(encoding="utf-8").strip())
    record_json_path = next((quant_run_root / "author" / "records").glob("*/record.json"))
    record_payload = json.loads(record_json_path.read_text(encoding="utf-8"))
    record_payload["llm"] = "qwen/qwen3.5-27b"
    record_json_path.write_text(json.dumps(record_payload), encoding="utf-8")

    qwen_quant_root = _build_quantitative_root(
        tmp_path,
        root_name="summ_qwen",
        matrix_name="matrix_qwen",
        model="qwen/qwen3.5-27b",
        output_name="quant_qwen",
    )

    with pytest.raises(ValueError, match="expected one llm label per multi-llm quantitative source"):
        run_quantitative_smoke_multi_llm(
            source_roots=(quant_root, qwen_quant_root),
            output_root=tmp_path / "quant_multi",
            score_summary_fn=_fake_score_summary,
        )


def test_run_quantitative_smoke_multi_llm_rejects_quantitative_sources_with_missing_llm_labels(tmp_path: Path) -> None:
    quant_root = _build_quantitative_root(
        tmp_path,
        root_name="summ_mistral",
        matrix_name="matrix_mistral",
        model="mistral-medium-latest",
        output_name="quant_mistral",
    )
    quant_run_root = Path((quant_root / "latest_run.txt").read_text(encoding="utf-8").strip())
    for record_json_path in quant_run_root.glob("*/records/*/record.json"):
        record_payload = json.loads(record_json_path.read_text(encoding="utf-8"))
        record_payload["llm"] = ""
        record_json_path.write_text(json.dumps(record_payload), encoding="utf-8")

    qwen_quant_root = _build_quantitative_root(
        tmp_path,
        root_name="summ_qwen",
        matrix_name="matrix_qwen",
        model="qwen/qwen3.5-27b",
        output_name="quant_qwen",
    )

    with pytest.raises(ValueError, match="missing llm label"):
        run_quantitative_smoke_multi_llm(
            source_roots=(quant_root, qwen_quant_root),
            output_root=tmp_path / "quant_multi",
            score_summary_fn=_fake_score_summary,
        )
