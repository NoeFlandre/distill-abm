from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from distill_abm.eval.doe_full import _fit_anova, analyze_factorial_anova


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


def test_fit_anova_falls_back_when_statsmodels_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "Role": ["Yes", "No", "Yes", "No"],
            "BLEU": [0.4, 0.3, 0.5, 0.45],
        }
    )
    original_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if name.startswith("statsmodels"):
            raise ImportError("statsmodels unavailable")
        return cast(Any, original_import)(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = _fit_anova(frame, "BLEU", ["Role"], 2)

    assert result is not None
    assert "sum_sq" in result.columns


def test_fit_anova_does_not_hide_runtime_analysis_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "Role": ["Yes", "No", "Yes", "No"],
            "BLEU": [0.4, 0.3, 0.5, 0.45],
        }
    )

    def fake_ols(*args: object, **kwargs: object) -> SimpleNamespace:
        _ = args, kwargs
        return SimpleNamespace(fit=lambda: (_ for _ in ()).throw(ValueError("anova failed")))

    statsmodels_module = ModuleType("statsmodels")
    formula_module = ModuleType("statsmodels.formula")
    formula_api_module = ModuleType("statsmodels.formula.api")
    formula_api_module.ols = fake_ols  # type: ignore[attr-defined]
    stats_module = ModuleType("statsmodels.stats")
    anova_module = ModuleType("statsmodels.stats.anova")
    anova_module.anova_lm = lambda *args, **kwargs: None  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "statsmodels", statsmodels_module)
    monkeypatch.setitem(sys.modules, "statsmodels.formula", formula_module)
    monkeypatch.setitem(sys.modules, "statsmodels.formula.api", formula_api_module)
    monkeypatch.setitem(sys.modules, "statsmodels.stats", stats_module)
    monkeypatch.setitem(sys.modules, "statsmodels.stats.anova", anova_module)

    with pytest.raises(ValueError, match="anova failed"):
        _fit_anova(frame, "BLEU", ["Role"], 2)
