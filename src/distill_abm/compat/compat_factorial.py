"""Compatibility helpers for deterministic DoE/factorial helper functions."""

from __future__ import annotations

import csv
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from typing import Any, TypeVar, cast

import pandas as pd

from distill_abm.eval.doe import clean_name, identify_factors_and_metrics

T = TypeVar("T")


def analyze_factorial_contributions(
    csv_path: str | Path,
    output_csv: str | Path,
    repetitions: int = 3,
    max_interaction_order: int = 2,
) -> pd.DataFrame | None:
    return _call_notebook_first(
        "analyze_factorial_contributions",
        _analyze_factorial_contributions_fallback,
        csv_path,
        output_csv,
        repetitions,
        max_interaction_order,
    )


def _analyze_factorial_contributions_fallback(
    csv_path: str | Path,
    output_csv: str | Path,
    repetitions: int = 3,
    max_interaction_order: int = 2,
) -> pd.DataFrame | None:
    def _effect_values(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            return numeric.fillna(0.0)
        lowered = series.astype(str).str.strip().str.lower()
        mapped = lowered.map(
            {
                "yes": 1.0,
                "no": -1.0,
                "bart": 1.0,
                "bert": -1.0,
                "gpt": 1.0,
                "claude": -1.0,
                "deepseek": 0.0,
            }
        )
        return mapped.fillna(0.0)

    frame = pd.read_csv(csv_path)
    factorial = frame.copy()
    factor_columns = get_factor_columns(factorial)
    metric_columns = [
        column
        for column in factorial.columns
        if column not in factor_columns and pd.api.types.is_numeric_dtype(factorial[column])
    ]
    if factor_columns:
        avg_df = factorial.groupby(factor_columns).mean(numeric_only=True).reset_index()
    else:
        group_size = max(int(repetitions), 1)
        avg_df = factorial.groupby(factorial.index // group_size).mean(numeric_only=True).reset_index(drop=True)
    avg_df = add_interactions(avg_df, factor_columns, max_order=max_interaction_order)
    effect_columns = factor_columns + [column for column in avg_df.columns if "_AND_" in column]
    design_size = 1
    for column in factor_columns:
        design_size *= 3 if column == "LLM" else 2
    results: list[dict[str, str | float]] = []
    for metric in metric_columns:
        metric_values = pd.to_numeric(avg_df[metric], errors="coerce").fillna(0.0)
        effects = {
            column: (_effect_values(avg_df[column]) * metric_values).sum() / design_size for column in effect_columns
        }
        sst = sum(value**2 for value in effects.values()) * design_size
        for effect, value in effects.items():
            contribution = (value**2 * design_size * 100) / sst if sst != 0 else 0.0
            results.append({"Feature": effect, "Metric": metric, "Contribution": contribution})
    output_df = (
        pd.DataFrame(results)
        .pivot(index="Feature", columns="Metric", values="Contribution")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    expected_features = set(factor_columns)
    if max_interaction_order >= 2:
        for combination in combinations(factor_columns, 2):
            expected_features.add("_AND_".join(combination))
    for feature in expected_features:
        if feature not in output_df["Feature"].values:
            output_df = pd.concat([output_df, pd.DataFrame({"Feature": [feature]})], ignore_index=True)
    output_df = output_df.fillna(0).sort_values("Feature")
    output_df.to_csv(output_csv, index=False)
    return output_df


def read_csv_to_df(csv_path: str | Path) -> pd.DataFrame:
    def _fallback(path: str | Path) -> pd.DataFrame:
        return pd.read_csv(path)

    return _call_notebook_first("read_csv_to_df", _fallback, csv_path)


def read_and_parse_csv(
    csv_path: str | Path,
    repetitions: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _call_notebook_first("read_and_parse_csv", _read_and_parse_csv_fallback, csv_path, repetitions)


def _read_and_parse_csv_fallback(
    csv_path: str | Path, repetitions: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(csv_path)
    numerical_df = frame.select_dtypes(include=["number"])
    filtered_numerical_df = numerical_df.loc[:, (numerical_df < 2).all()]
    group_size = max(int(repetitions), 1)
    avg_df = filtered_numerical_df.groupby(frame.index // group_size).mean().reset_index(drop=True)
    minus_one_mask = (filtered_numerical_df == -1).any()
    columns_with_minus_one = avg_df.loc[:, minus_one_mask[minus_one_mask].index]
    columns_without_minus_one = avg_df.loc[:, minus_one_mask[~minus_one_mask].index]
    return avg_df, columns_with_minus_one, columns_without_minus_one


def create_factorial_design(
    csv_or_frame: str | Path | pd.DataFrame,
    repetitions_or_columns: int | list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    if isinstance(csv_or_frame, pd.DataFrame):
        if not isinstance(repetitions_or_columns, list):
            raise TypeError("expected columns list when passing a DataFrame")
        return add_interactions(csv_or_frame, repetitions_or_columns, max_order=2)
    if not isinstance(repetitions_or_columns, int):
        raise TypeError("expected repetitions int when passing a CSV path")
    return _call_notebook_first(
        "create_factorial_design",
        _create_factorial_design_fallback,
        csv_or_frame,
        repetitions_or_columns,
    )


def _create_factorial_design_fallback(
    csv_path: str | Path,
    repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    avg_df, columns, specific_columns = _read_and_parse_csv_fallback(csv_path, repetitions)
    factorial_df = avg_df.copy()
    for col in columns:
        for spec_col in specific_columns:
            new_col_name = f"{col}_evaluating_{spec_col}"
            factorial_df[new_col_name] = factorial_df[col] * factorial_df[spec_col]
    for combination in combinations(columns, 2):
        col_name = "_AND_".join(str(part) for part in combination)
        factorial_df[col_name] = factorial_df[list(combination)].prod(axis=1)
        for spec_col in specific_columns:
            new_col_name = f"{col_name}_evaluating_{spec_col}"
            factorial_df[new_col_name] = factorial_df[col_name] * factorial_df[spec_col]
    return factorial_df, columns, specific_columns


def calculate_sums_and_sst(
    frame: pd.DataFrame,
    metric_or_columns: str | list[str],
    num_of_experiments: int,
) -> tuple[list[tuple[str, float]], float]:
    if isinstance(metric_or_columns, list):
        metric_df = frame[metric_or_columns]
        divisor = 2**num_of_experiments
        sums = [(column, metric_df[column].sum() / divisor) for column in metric_df.columns]
        sst = sum(value**2 for _, value in sums[1:]) * divisor
        return sums, sst
    return _call_notebook_first(
        "calculate_sums_and_sst",
        _calculate_sums_and_sst_fallback,
        frame,
        metric_or_columns,
        num_of_experiments,
    )


def _calculate_sums_and_sst_fallback(
    frame: pd.DataFrame,
    metric: str,
    num_of_experiments: int,
) -> tuple[list[tuple[str, float]], float]:
    columns = frame.filter(like=metric).columns
    metric_df = frame[columns]
    divisor = 2**num_of_experiments
    sums = [(column, metric_df[column].sum() / divisor) for column in metric_df.columns]
    sst = sum(value**2 for _, value in sums[1:]) * divisor
    return sums, sst


def calculate_sst(csv_path: str | Path, repetitions: int) -> tuple[list[tuple[list[tuple[str, float]], float]], int]:
    return _call_notebook_first("calculate_sst", _calculate_sst_fallback, csv_path, repetitions)


def _calculate_sst_fallback(
    csv_path: str | Path,
    repetitions: int,
) -> tuple[list[tuple[list[tuple[str, float]], float]], int]:
    list_of_name_and_sst: list[tuple[list[tuple[str, float]], float]] = []
    avg_df, columns, specific_columns = _create_factorial_design_fallback(csv_path, repetitions)
    num_of_experiments = columns.shape[1]
    for column in specific_columns:
        column_sums, sst = _calculate_sums_and_sst_fallback(avg_df, str(column), num_of_experiments)
        list_of_name_and_sst.append((column_sums, sst))
    return list_of_name_and_sst, num_of_experiments


def compute_results(csv_path: str | Path, repetitions: int) -> list[list[tuple[str, float]]]:
    return _call_notebook_first("compute_results", _compute_results_fallback, csv_path, repetitions)


def _compute_results_fallback(csv_path: str | Path, repetitions: int) -> list[list[tuple[str, float]]]:
    list_of_name_and_sst, num_of_experiments = _calculate_sst_fallback(csv_path, repetitions)
    list_with_sst: list[list[tuple[str, float]]] = []
    for column_sums, sst in list_of_name_and_sst:
        metric_contributions: list[tuple[str, float]] = []
        for feature, value in column_sums[1:]:
            contribution = 0.0
            if sst != 0:
                contribution = (value**2) * (2**num_of_experiments) * 100 / sst
            metric_contributions.append((feature, contribution))
        list_with_sst.append(metric_contributions)
    return list_with_sst


def remove_evaluating_suffix(text: str) -> str:
    def _fallback(value: str) -> str:
        marker = "_evaluating"
        index = value.find(marker)
        return value[:index] if index >= 0 else value

    return _call_notebook_first("remove_evaluating_suffix", _fallback, text)


def return_csv(csv_path: str | Path, output_csv: str | Path, repetitions: int) -> Path:
    return _call_notebook_first("return_csv", _return_csv_fallback, csv_path, output_csv, repetitions)


def return_csv_2(csv_path: str | Path, output_csv: str | Path, repetitions: int) -> Path:
    return _call_notebook_first("return_csv_2", _return_csv_2_fallback, csv_path, output_csv, repetitions)


def _normalize_output_csv(path_or_prefix: str | Path) -> Path:
    path = Path(path_or_prefix)
    return path if path.suffix.lower() == ".csv" else path.with_suffix(".csv")


def _return_csv_fallback(csv_path: str | Path, output_csv: str | Path, repetitions: int) -> Path:
    list_with_sst = _compute_results_fallback(csv_path, repetitions)
    filename = _normalize_output_csv(output_csv)
    with filename.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        titles = ["Feature"]
        for metric_results in list_with_sst:
            if metric_results:
                titles.append(metric_results[0][0].split("_")[-1])
        writer.writerow(titles)
        if list_with_sst and list_with_sst[0]:
            for index, (feature, _value) in enumerate(list_with_sst[0]):
                row: list[str | float] = [remove_evaluating_suffix(feature)]
                for metric_results in list_with_sst:
                    row.append(metric_results[index][1] if index < len(metric_results) else "")
                writer.writerow(row)
    return filename


def _return_csv_2_fallback(csv_path: str | Path, output_csv: str | Path, repetitions: int) -> Path:
    data = _compute_results_fallback(csv_path, repetitions)
    filtered_data = data[0] if data else []
    processed = [(feature.replace("_evaluating", ""), result) for feature, result in filtered_data]
    filename = _normalize_output_csv(output_csv)
    with filename.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Feature", "Result"])
        writer.writerows(processed)
    return filename


def preprocess_for_factorial(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = [clean_name(name) for name in cleaned.columns]
    return cleaned


def get_factor_columns(frame: pd.DataFrame) -> list[str]:
    factors, _ = identify_factors_and_metrics(frame)
    return factors


def add_interactions(frame: pd.DataFrame, columns: list[str], max_order: int = 2) -> pd.DataFrame:
    out = frame.copy()
    for order in range(2, max_order + 1):
        for combo in combinations(columns, order):
            name = "_AND_".join(combo)
            out[name] = out[list(combo)].astype(str).agg("_".join, axis=1)
    return out


def _call_notebook_first(name: str, fallback: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    from distill_abm.compat.compat_callables import _call_notebook_first as _dispatch

    return cast(T, _dispatch(name, fallback, *args, **kwargs))
