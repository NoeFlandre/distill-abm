"""Design-of-experiments helpers for ANOVA-compatible workflows."""

from __future__ import annotations

import re

import pandas as pd


def clean_name(name: str) -> str:
    """Sanitizes dataframe column names for statsmodels formula compatibility."""
    cleaned = re.sub(r"\s+", "_", str(name))
    cleaned = re.sub(r"[(),-/]+", "_", cleaned)
    return cleaned.strip("_")


def identify_factors_and_metrics(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Splits dataframe columns into categorical factors and numeric metrics."""
    max_unique = 5
    potential_factors = _candidate_factor_columns(df, max_unique)
    identified_factors = _identify_factors(df, potential_factors, max_unique)
    metrics = _identify_numeric_metrics(df, identified_factors)
    factors = [factor for factor in identified_factors if factor not in metrics]
    return factors, metrics


def _candidate_factor_columns(df: pd.DataFrame, max_unique: int) -> list[str]:
    potential = [str(column) for column in df.select_dtypes(include=["object", "category"]).columns.tolist()]
    numeric_columns = [str(column) for column in df.select_dtypes(include=["number"]).columns.tolist()]
    for column in numeric_columns:
        unique_values = set(df[column].dropna().unique())
        is_binary = unique_values.issubset({0, 1}) or unique_values.issubset({-1, 1})
        if is_binary and df[column].nunique(dropna=True) <= max_unique:
            potential.append(column)
    return potential


def _identify_factors(df: pd.DataFrame, candidates: list[str], max_unique: int) -> list[str]:
    known_patterns = [
        {"Yes", "No"},
        {"BART", "BERT"},
        {"none", "bart", "bert", "t5", "longformer_ext"},
        {"plot", "table", "plot+table"},
        {"GPT", "Claude", "DeepSeek"},
    ]
    factors: list[str] = []
    seen: set[str] = set()
    for column in candidates:
        unique_values = set(df[column].dropna().astype(str).unique())
        known = any(unique_values == pattern for pattern in known_patterns)
        is_categorical = isinstance(df[column].dtype, pd.CategoricalDtype)
        is_object = pd.api.types.is_object_dtype(df[column]) or is_categorical
        low_cardinality = is_object and df[column].nunique(dropna=True) <= max_unique
        if (known or low_cardinality) and column not in seen:
            factors.append(column)
            seen.add(column)
    return factors


def _identify_numeric_metrics(df: pd.DataFrame, factor_columns: list[str]) -> list[str]:
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    metric_candidates = [column for column in numeric if column not in factor_columns]
    metrics: list[str] = []
    for column in metric_candidates:
        try:
            pd.to_numeric(df[column])
        except (ValueError, TypeError):
            continue
        metrics.append(column)
    return metrics


def clean_statsmodels_feature_name(feature_name: str) -> str:
    """Normalizes feature names emitted by statsmodels ANOVA tables."""
    cleaned = feature_name.replace("C(", "").replace(")", "")
    return cleaned.replace(":", "_AND_")
