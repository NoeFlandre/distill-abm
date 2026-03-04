"""Full factorial ANOVA analysis for factor contribution inspection."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import cast

import pandas as pd

from distill_abm.eval.doe import (
    clean_name,
    clean_statsmodels_feature_name,
    identify_factors_and_metrics,
)


def analyze_factorial_anova(
    csv_path: Path,
    output_csv: Path,
    max_interaction_order: int = 2,
) -> pd.DataFrame | None:
    """Build ANOVA contribution table for factor and interaction effects."""
    frame = _read_input(csv_path)
    if frame is None:
        return None
    prepared = _prepare_columns(frame)
    if prepared is None:
        return None
    clean_df, clean_factors, clean_metrics, reverse_map = prepared
    contributions = _collect_contributions(clean_df, clean_factors, clean_metrics, reverse_map, max_interaction_order)
    if not contributions:
        return None
    result = _finalize_table(contributions, clean_factors, max_interaction_order)
    result.to_csv(output_csv, index=False, float_format="%.2f")
    return result


def _read_input(csv_path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None


def _prepare_columns(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, str]] | None:
    factors, metrics = identify_factors_and_metrics(frame)
    if not factors or not metrics:
        return None
    mapping = {name: clean_name(name) for name in [*factors, *metrics]}
    reverse_map = {clean: original for original, clean in mapping.items()}
    clean_df = frame.rename(columns=mapping).copy()
    return clean_df, [mapping[name] for name in factors], [mapping[name] for name in metrics], reverse_map


def _collect_contributions(
    frame: pd.DataFrame,
    factors: list[str],
    metrics: list[str],
    reverse_map: dict[str, str],
    max_interaction_order: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for metric in metrics:
        metric_rows = _analyze_metric(frame.copy(), metric, factors, max_interaction_order, reverse_map)
        rows.extend(metric_rows)
    return rows


def _analyze_metric(
    frame: pd.DataFrame,
    metric: str,
    factors: list[str],
    max_interaction_order: int,
    reverse_map: dict[str, str],
) -> list[dict[str, float | str]]:
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    analysis = frame.dropna(subset=[metric, *factors]).copy()
    if analysis.empty:
        return []
    for factor in factors:
        analysis[factor] = analysis[factor].astype(str)
        if analysis[factor].nunique() < 2:
            return []
    anova = _fit_anova(analysis, metric, factors, max_interaction_order)
    if anova is None or anova.empty:
        return []
    total = float(anova["sum_sq"].sum())
    if total <= 0:
        return []
    output_metric = reverse_map.get(metric, metric)
    return _anova_rows(anova, total, output_metric)


def _fit_anova(
    frame: pd.DataFrame,
    metric: str,
    factors: list[str],
    max_interaction_order: int,
) -> pd.DataFrame | None:
    formula = _build_formula(metric, factors, max_interaction_order)
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        model = smf.ols(formula, data=frame).fit()
        return cast(pd.DataFrame, anova_lm(model, typ=2))
    except Exception:
        return _fallback_anova(frame, metric, factors, max_interaction_order)


def _build_formula(metric: str, factors: list[str], max_interaction_order: int) -> str:
    main_effects = [f"C({factor})" for factor in factors]
    terms = [*main_effects]
    for order in range(2, max_interaction_order + 1):
        terms.extend(":".join(combo) for combo in combinations(main_effects, order))
    return f"{metric} ~ {' + '.join(terms)}"


def _fallback_anova(
    frame: pd.DataFrame,
    metric: str,
    factors: list[str],
    max_interaction_order: int,
) -> pd.DataFrame:
    rows: list[tuple[str, float]] = []
    for factor in factors:
        rows.append((f"C({factor})", _sum_squares_effect(frame, metric, [factor])))
    for order in range(2, max_interaction_order + 1):
        for combo in combinations(factors, order):
            term = ":".join(f"C({factor})" for factor in combo)
            rows.append((term, _sum_squares_effect(frame, metric, list(combo))))
    total = float(((frame[metric] - frame[metric].mean()) ** 2).sum())
    explained = sum(value for _, value in rows)
    residual = max(total - explained, 0.0)
    rows.append(("Residual", residual))
    return pd.DataFrame({"sum_sq": [value for _, value in rows]}, index=[name for name, _ in rows])


def _sum_squares_effect(frame: pd.DataFrame, metric: str, groups: list[str]) -> float:
    grouped = frame.groupby(groups, dropna=True)[metric].agg(["mean", "size"])
    global_mean = float(frame[metric].mean())
    return float((grouped["size"] * ((grouped["mean"] - global_mean) ** 2)).sum())


def _anova_rows(anova: pd.DataFrame, total_sum: float, metric_name: str) -> list[dict[str, float | str]]:
    output: list[dict[str, float | str]] = []
    for feature in anova.index:
        if feature == "Residual":
            continue
        sum_sq = float(anova.loc[feature, "sum_sq"])
        contribution = (sum_sq / total_sum) * 100 if total_sum > 0 else 0.0
        output.append(
            {
                "Feature": clean_statsmodels_feature_name(feature),
                "Metric": metric_name,
                "Contribution (%)": contribution,
            }
        )
    return output


def _finalize_table(
    rows: list[dict[str, float | str]],
    factors: list[str],
    max_interaction_order: int,
) -> pd.DataFrame:
    result = pd.DataFrame(rows).pivot(index="Feature", columns="Metric", values="Contribution (%)")
    result = result.fillna(0.0)
    for expected in _expected_features(factors, max_interaction_order):
        if expected not in result.index:
            result.loc[expected] = 0.0
    result = result.sort_index().reset_index().rename_axis(None, axis=1)
    return result


def _expected_features(factors: list[str], max_interaction_order: int) -> set[str]:
    main = [clean_statsmodels_feature_name(f"C({factor})") for factor in factors]
    features = set(main)
    for order in range(2, max_interaction_order + 1):
        for combo in combinations(main, order):
            features.add("_AND_".join(sorted(combo)))
    return features
