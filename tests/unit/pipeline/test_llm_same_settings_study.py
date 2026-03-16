from __future__ import annotations

from pathlib import Path

import pandas as pd

from distill_abm.pipeline.llm_same_settings_study import run_llm_same_settings_study


def test_run_llm_same_settings_study_filters_to_anchor_settings_and_shared_rows(tmp_path: Path) -> None:
    gemini_root = tmp_path / "gemini" / "runs" / "run_1"
    kimi_root = tmp_path / "kimi" / "runs" / "run_1"
    qwen_root = tmp_path / "qwen" / "runs" / "run_1"
    opus_root = tmp_path / "opus" / "runs" / "run_1"
    for root in (gemini_root, kimi_root, qwen_root, opus_root):
        (root / "combined").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "record_id": "g1",
                "bundle_id": "b1",
                "case_id": "c1",
                "abm": "fauna",
                "llm": "google/gemini-3.1-pro-preview",
                "evidence": "plot",
                "prompt": "example",
                "role": False,
                "insights": False,
                "example": True,
                "summarizer": "bart",
                "repetition": 1,
                "reference_family": "author",
                "summary_output_path": "x",
                "BLEU": 0.10,
                "METEOR": 0.20,
                "R-1": 0.30,
                "R-2": 0.11,
                "R-L": 0.21,
                "Reading ease": 27.0,
            },
            {
                "record_id": "g2",
                "bundle_id": "b2",
                "case_id": "c2",
                "abm": "grazing",
                "llm": "google/gemini-3.1-pro-preview",
                "evidence": "plot",
                "prompt": "example",
                "role": False,
                "insights": False,
                "example": True,
                "summarizer": "bart",
                "repetition": 1,
                "reference_family": "author",
                "summary_output_path": "y",
                "BLEU": 0.20,
                "METEOR": 0.21,
                "R-1": 0.31,
                "R-2": 0.12,
                "R-L": 0.20,
                "Reading ease": 37.0,
            },
            {
                "record_id": "g3",
                "bundle_id": "b3",
                "case_id": "c3",
                "abm": "milk_consumption",
                "llm": "google/gemini-3.1-pro-preview",
                "evidence": "plot",
                "prompt": "example",
                "role": False,
                "insights": False,
                "example": True,
                "summarizer": "bart",
                "repetition": 1,
                "reference_family": "author",
                "summary_output_path": "z",
                "BLEU": 0.30,
                "METEOR": 0.22,
                "R-1": 0.32,
                "R-2": 0.13,
                "R-L": 0.19,
                "Reading ease": 47.0,
            },
        ]
    ).to_csv(gemini_root / "combined" / "quantitative_rows.csv", index=False)

    pd.DataFrame(
        [
            {
                "record_id": "k1",
                "bundle_id": "b1",
                "case_id": "c1",
                "abm": "fauna",
                "llm": "moonshotai/kimi-k2.5",
                "evidence": "plot",
                "prompt": "example",
                "role": False,
                "insights": False,
                "example": True,
                "summarizer": "bart",
                "repetition": 1,
                "reference_family": "author",
                "summary_output_path": "x",
                "BLEU": 0.30,
                "METEOR": 0.25,
                "R-1": 0.33,
                "R-2": 0.15,
                "R-L": 0.24,
                "Reading ease": 33.0,
            },
            {
                "record_id": "k2",
                "bundle_id": "b2",
                "case_id": "c2",
                "abm": "grazing",
                "llm": "moonshotai/kimi-k2.5",
                "evidence": "plot",
                "prompt": "example",
                "role": False,
                "insights": False,
                "example": True,
                "summarizer": "bart",
                "repetition": 1,
                "reference_family": "author",
                "summary_output_path": "y",
                "BLEU": 0.31,
                "METEOR": 0.26,
                "R-1": 0.34,
                "R-2": 0.16,
                "R-L": 0.36,
                "Reading ease": 36.0,
            },
            {
                "record_id": "k3",
                "bundle_id": "b9",
                "case_id": "c9",
                "abm": "fauna",
                "llm": "moonshotai/kimi-k2.5",
                "evidence": "plot",
                "prompt": "none",
                "role": False,
                "insights": False,
                "example": False,
                "summarizer": "bart",
                "repetition": 1,
                "reference_family": "author",
                "summary_output_path": "skip",
                "BLEU": 0.99,
                "METEOR": 0.99,
                "R-1": 0.99,
                "R-2": 0.99,
                "R-L": 0.99,
                "Reading ease": 99.0,
            },
        ]
    ).to_csv(kimi_root / "combined" / "quantitative_rows.csv", index=False)

    pd.DataFrame(
        [
            {
                "record_id": "q1",
                "bundle_id": "b1",
                "case_id": "c1",
                "abm": "fauna",
                "llm": "qwen/qwen3.5-27b",
                "evidence": "plot",
                "prompt": "example",
                "role": False,
                "insights": False,
                "example": True,
                "summarizer": "bart",
                "repetition": 1,
                "reference_family": "author",
                "summary_output_path": "x",
                "BLEU": 0.40,
                "METEOR": 0.30,
                "R-1": 0.40,
                "R-2": 0.17,
                "R-L": 0.22,
                "Reading ease": 38.0,
            },
            {
                "record_id": "q2",
                "bundle_id": "b2",
                "case_id": "c2",
                "abm": "grazing",
                "llm": "qwen/qwen3.5-27b",
                "evidence": "plot",
                "prompt": "example",
                "role": False,
                "insights": False,
                "example": True,
                "summarizer": "bart",
                "repetition": 1,
                "reference_family": "author",
                "summary_output_path": "y",
                "BLEU": 0.41,
                "METEOR": 0.31,
                "R-1": 0.41,
                "R-2": 0.18,
                "R-L": 0.22,
                "Reading ease": 61.0,
            },
        ]
    ).to_csv(qwen_root / "combined" / "quantitative_rows.csv", index=False)

    pd.DataFrame(
        [
            {
                "record_id": "o1",
                "bundle_id": "b1",
                "case_id": "c1",
                "abm": "fauna",
                "llm": "anthropic/claude-opus-4.6",
                "evidence": "plot",
                "prompt": "example",
                "role": False,
                "insights": False,
                "example": True,
                "summarizer": "bart",
                "repetition": 1,
                "reference_family": "author",
                "summary_output_path": "x",
                "BLEU": 0.35,
                "METEOR": 0.28,
                "R-1": 0.38,
                "R-2": 0.16,
                "R-L": 0.23,
                "Reading ease": 35.0,
            },
            {
                "record_id": "o2",
                "bundle_id": "b2",
                "case_id": "c2",
                "abm": "grazing",
                "llm": "anthropic/claude-opus-4.6",
                "evidence": "plot",
                "prompt": "example",
                "role": False,
                "insights": False,
                "example": True,
                "summarizer": "bart",
                "repetition": 1,
                "reference_family": "author",
                "summary_output_path": "y",
                "BLEU": 0.36,
                "METEOR": 0.29,
                "R-1": 0.39,
                "R-2": 0.17,
                "R-L": 0.23,
                "Reading ease": 42.0,
            },
        ]
    ).to_csv(opus_root / "combined" / "quantitative_rows.csv", index=False)

    result = run_llm_same_settings_study(
        anchor_source_root=gemini_root,
        comparison_source_roots=[kimi_root, qwen_root, opus_root],
        output_root=tmp_path / "study",
    )

    assert result.report_markdown_path.exists() is True
    assert result.same_settings_long_path.exists() is True
    assert result.master_comparison_path.exists() is True
    assert result.metric_summary_path.exists() is True
    assert result.metric_win_summary_path.exists() is True

    long_rows = pd.read_csv(result.same_settings_long_path)
    assert sorted(long_rows["model_label"].unique().tolist()) == ["gemini", "kimi", "opus", "qwen"]
    assert len(long_rows) == 8
    assert sorted(long_rows["abm"].unique().tolist()) == ["fauna", "grazing"]
    assert long_rows["example"].tolist().count(True) == 8

    master = pd.read_csv(result.master_comparison_path)
    assert len(master) == 2
    assert master["winner_reading_ease"].tolist() == ["qwen", "qwen"]
    assert master["gemini_reading_ease"].tolist() == [27.0, 37.0]
    assert master["kimi_r_l"].tolist() == [0.24, 0.36]
    assert master["opus_r_l"].tolist() == [0.23, 0.23]
    assert master["qwen_r_l"].tolist() == [0.22, 0.22]

    summary = pd.read_csv(result.metric_summary_path)
    reading_ease = summary[summary["metric"] == "Reading ease"].set_index("model_label")
    assert reading_ease.loc["gemini", "mean_score"] == 32.0
    assert reading_ease.loc["opus", "mean_score"] == 38.5
    assert reading_ease.loc["qwen", "mean_score"] == 49.5

    wins = pd.read_csv(result.metric_win_summary_path)
    reading_ease_wins = wins[wins["metric"] == "Reading ease"].set_index("model_label")
    assert reading_ease_wins.loc["qwen", "win_count"] == 2
    assert reading_ease_wins.loc["gemini", "win_count"] == 0

    report_text = result.report_markdown_path.read_text(encoding="utf-8")
    assert "Optimization Same-Settings LLM Comparison" in report_text
    assert "master_comparison.csv" in report_text
