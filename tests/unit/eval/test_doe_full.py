from __future__ import annotations

from pathlib import Path

import pandas as pd

from distill_abm.eval.doe_full import analyze_factorial_anova


def test_analyze_factorial_anova_generates_table(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "Case study": ["Fauna", "Fauna", "Grazing", "Grazing"],
            "LLM": ["GPT", "Claude", "GPT", "Claude"],
            "Role": ["Yes", "No", "Yes", "No"],
            "BLEU": [0.4, 0.3, 0.5, 0.45],
            "ROUGE": [0.5, 0.35, 0.55, 0.5],
        }
    )
    source = tmp_path / "anova_input.csv"
    out = tmp_path / "anova_output.csv"
    frame.to_csv(source, index=False)

    result = analyze_factorial_anova(source, out, max_interaction_order=2)

    assert result is not None
    assert out.exists()
    assert "Feature" in result.columns
    assert "BLEU" in result.columns


def test_analyze_factorial_anova_returns_none_for_unreadable_csv(tmp_path: Path) -> None:
    """Test that analyze_factorial_anova returns None for unreadable CSV path."""
    unreadable_path = tmp_path / "nonexistent.csv"
    output_path = tmp_path / "output.csv"

    result = analyze_factorial_anova(unreadable_path, output_path)

    assert result is None
    # Output file should not be created
    assert not output_path.exists()


def test_analyze_factorial_anova_returns_none_for_invalid_csv_content(tmp_path: Path) -> None:
    """Test that analyze_factorial_anova returns None for invalid CSV content."""
    invalid_csv = tmp_path / "invalid.csv"
    invalid_csv.write_text("not,a,valid,csv\n", encoding="utf-8")
    output_path = tmp_path / "output.csv"

    result = analyze_factorial_anova(invalid_csv, output_path)

    assert result is None


def test_analyze_factorial_anova_returns_none_for_empty_dataframe(tmp_path: Path) -> None:
    """Test that analyze_factorial_anova returns None for empty DataFrame."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("Case study,LLM,Role,BLEU\n", encoding="utf-8")
    output_path = tmp_path / "output.csv"

    result = analyze_factorial_anova(empty_csv, output_path)

    assert result is None


def test_analyze_factorial_anova_returns_none_for_missing_factors_or_metrics(tmp_path: Path) -> None:
    """Test that analyze_factorial_anova returns None when no factors or metrics identified."""
    # DataFrame with no numeric columns (metrics)
    frame = pd.DataFrame(
        {
            "Case study": ["Fauna", "Grazing"],
            "LLM": ["GPT", "Claude"],
        }
    )
    source = tmp_path / "no_metrics.csv"
    out = tmp_path / "output.csv"
    frame.to_csv(source, index=False)

    result = analyze_factorial_anova(source, out)

    assert result is None


def test_analyze_factorial_anova_returns_none_for_zero_variance_metric(tmp_path: Path) -> None:
    """Test that analyze_factorial_anova returns None when metric has zero variance."""
    frame = pd.DataFrame(
        {
            "Case study": ["Fauna", "Fauna", "Grazing", "Grazing"],
            "LLM": ["GPT", "Claude", "GPT", "Claude"],
            "BLEU": [0.5, 0.5, 0.5, 0.5],  # All same values - zero variance
        }
    )
    source = tmp_path / "zero_variance.csv"
    out = tmp_path / "output.csv"
    frame.to_csv(source, index=False)

    result = analyze_factorial_anova(source, out)

    # Returns None when total sum_sq is 0
    assert result is None or result.empty
