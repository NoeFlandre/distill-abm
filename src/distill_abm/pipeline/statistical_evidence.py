"""Statistical evidence generation for plot-relevant time-series slices."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from distill_abm.ingest.csv_ingest import matching_columns

DEFAULT_MAX_EXTREMA = 8
DEFAULT_MAX_WINDOWS = 10
DEFAULT_MAX_CHANGEPOINTS = 10
DEFAULT_MAX_OSCILLATION_INDICES = 20
MAX_HEAVY_SIGNAL_POINTS = 512


class StatisticalSeriesSummary(BaseModel):
    """Structured statistics for one matched time series."""

    column: str
    sample_count: int
    valid_sample_count: int = 0
    dropped_non_numeric_count: int = 0
    start_value: float
    end_value: float
    mean: float
    median: float
    std: float
    variance: float
    minimum: float
    minimum_index: int
    maximum: float
    maximum_index: int
    peaks: list[dict[str, float]] = Field(default_factory=list)
    valleys: list[dict[str, float]] = Field(default_factory=list)
    inflection_indices: list[int] = Field(default_factory=list)
    rolling_mann_kendall: dict[str, object] = Field(default_factory=dict)
    change_points: list[int] = Field(default_factory=list)
    oscillation: dict[str, object] = Field(default_factory=dict)


class StatisticalEvidence(BaseModel):
    """Reviewer-friendly statistical evidence derived from plot-relevant series only."""

    reporter_pattern: str
    matched_columns: list[str]
    time_column: str | None = None
    selected_frame_csv: str = ""
    summary_text: str = ""
    summary_payload: dict[str, object] = Field(default_factory=dict)
    compression_tier: int = 0


def preferred_time_column(columns: list[str]) -> str | None:
    for candidate in ("[step]", "time_step", "step"):
        if candidate in columns:
            return candidate
    return None


def select_plot_relevant_frame(*, frame: pd.DataFrame, reporter_pattern: str) -> tuple[pd.DataFrame, str | None]:
    """Return only the matched plot series and optional time column."""
    columns = [str(column) for column in frame.columns]
    matched = matching_columns(columns, include_pattern=reporter_pattern)
    if not matched:
        return pd.DataFrame(), preferred_time_column(columns)
    time_column = preferred_time_column(columns)
    selected = [time_column, *matched] if time_column is not None else [*matched]
    return frame[selected].copy(), time_column


def build_statistical_evidence(
    *,
    frame: pd.DataFrame,
    reporter_pattern: str,
    compression_tier: int = 0,
) -> StatisticalEvidence:
    """Build a compact but comprehensive statistics dump from the matched plot series only."""
    selected_frame, time_column = select_plot_relevant_frame(frame=frame, reporter_pattern=reporter_pattern)
    if selected_frame.empty:
        unmatched_payload: dict[str, object] = {
            "reporter_pattern": reporter_pattern,
            "time_column": time_column,
            "matched_columns": [],
            "status": "unmatched_metric_pattern",
            "series": [],
            "compression_tier": compression_tier,
        }
        return StatisticalEvidence(
            reporter_pattern=reporter_pattern,
            matched_columns=[],
            time_column=time_column,
            selected_frame_csv="",
            summary_text="No simulation series matched the requested metric pattern.\n",
            summary_payload=unmatched_payload,
            compression_tier=compression_tier,
        )

    matched_columns = [str(column) for column in selected_frame.columns if str(column) != time_column]
    series_summaries = _build_series_summaries(
        selected_frame=selected_frame,
        matched_columns=matched_columns,
        time_column=time_column,
        compression_tier=compression_tier,
    )
    matched_payload: dict[str, object] = {
        "reporter_pattern": reporter_pattern,
        "time_column": time_column,
        "matched_columns": matched_columns,
        "matched_series_count": len(matched_columns),
        "series_summary_mode": "aggregate_mean" if len(matched_columns) > 1 else "single_series",
        "compression_tier": compression_tier,
        "series": [summary.model_dump(mode="json") for summary in series_summaries],
    }
    return StatisticalEvidence(
        reporter_pattern=reporter_pattern,
        matched_columns=matched_columns,
        time_column=time_column,
        selected_frame_csv=selected_frame.to_csv(index=False),
        summary_text=_render_summary_text(
            reporter_pattern=reporter_pattern,
            time_column=time_column,
            matched_series_count=len(matched_columns),
            summaries=series_summaries,
            compression_tier=compression_tier,
        ),
        summary_payload=matched_payload,
        compression_tier=compression_tier,
    )


def _build_series_summaries(
    *,
    selected_frame: pd.DataFrame,
    matched_columns: list[str],
    time_column: str | None,
    compression_tier: int,
) -> list[StatisticalSeriesSummary]:
    time_values = selected_frame[time_column] if time_column is not None else None
    if len(matched_columns) == 1:
        column = matched_columns[0]
        return [
            _summarize_series(
                column=column,
                raw_values=selected_frame[column],
                time_values=time_values,
                compression_tier=compression_tier,
            )
        ]

    numeric_frame = selected_frame[matched_columns].apply(pd.to_numeric, errors="coerce")
    aggregate_series = numeric_frame.mean(axis=1, skipna=True)
    return [
        _summarize_series(
            column=f"aggregate_mean_of_{len(matched_columns)}_matched_series",
            raw_values=aggregate_series,
            time_values=time_values,
            compression_tier=compression_tier,
        )
    ]


def render_evidence_artifacts(*, evidence: StatisticalEvidence, output_dir: Path, stem: str) -> tuple[Path, Path, Path]:
    """Persist table evidence text, structured payload, and selected input series."""
    summary_path = output_dir / f"{stem}.txt"
    payload_path = output_dir / f"{stem}.json"
    series_path = output_dir / f"{stem}_series.csv"
    summary_path.write_text(evidence.summary_text, encoding="utf-8")
    payload_path.write_text(json.dumps(evidence.summary_payload, indent=2), encoding="utf-8")
    series_path.write_text(evidence.selected_frame_csv, encoding="utf-8")
    return summary_path, payload_path, series_path


def _summarize_series(
    *,
    column: str,
    raw_values: pd.Series,
    time_values: pd.Series | None,
    compression_tier: int,
) -> StatisticalSeriesSummary:
    total_sample_count = len(raw_values)
    values, resolved_time_values, resolved_indices, dropped_non_numeric_count = _coerce_numeric_series(
        raw_values=raw_values,
        time_values=time_values,
    )
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        zero = 0.0
        return StatisticalSeriesSummary(
            column=column,
            sample_count=total_sample_count,
            valid_sample_count=0,
            dropped_non_numeric_count=dropped_non_numeric_count,
            start_value=zero,
            end_value=zero,
            mean=zero,
            median=zero,
            std=zero,
            variance=zero,
            minimum=zero,
            minimum_index=0,
            maximum=zero,
            maximum_index=0,
            rolling_mann_kendall={"status": "empty"},
            oscillation={"status": "empty"},
        )
    peaks, valleys = _detect_extrema(
        values=values,
        index_values=resolved_indices,
        time_values=resolved_time_values,
        compression_tier=compression_tier,
    )
    return StatisticalSeriesSummary(
        column=column,
        sample_count=total_sample_count,
        valid_sample_count=len(values),
        dropped_non_numeric_count=dropped_non_numeric_count,
        start_value=float(values[0]),
        end_value=float(values[-1]),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        std=float(np.std(values)),
        variance=float(np.var(values)),
        minimum=float(np.min(values)),
        minimum_index=int(resolved_indices[int(np.argmin(values))]),
        maximum=float(np.max(values)),
        maximum_index=int(resolved_indices[int(np.argmax(values))]),
        peaks=peaks,
        valleys=valleys,
        inflection_indices=_detect_inflections(
            values=values, index_values=resolved_indices, compression_tier=compression_tier
        ),
        rolling_mann_kendall=_rolling_mann_kendall(
            values=values, index_values=resolved_indices, compression_tier=compression_tier
        ),
        change_points=_detect_change_points(
            values=values,
            index_values=resolved_indices,
            compression_tier=compression_tier,
        ),
        oscillation=_detect_oscillation(
            values=values,
            index_values=resolved_indices,
            compression_tier=compression_tier,
        ),
    )


def _coerce_numeric_series(
    *,
    raw_values: pd.Series,
    time_values: pd.Series | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, int]:
    numeric_values = pd.to_numeric(raw_values, errors="coerce")
    mask = np.isfinite(numeric_values.to_numpy(dtype=float))
    index_values = np.arange(len(raw_values), dtype=int)[mask]
    dropped_non_numeric_count = int(len(raw_values) - int(mask.sum()))
    filtered_values = numeric_values.to_numpy(dtype=float)[mask]
    resolved_time_values = None
    if time_values is not None:
        numeric_time = pd.to_numeric(time_values, errors="coerce").to_numpy(dtype=float)
        resolved_time_values = numeric_time[mask]
    return filtered_values, resolved_time_values, index_values, dropped_non_numeric_count


def _detect_extrema(
    *,
    values: np.ndarray,
    index_values: np.ndarray,
    time_values: np.ndarray | None,
    compression_tier: int,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    max_items = max(DEFAULT_MAX_EXTREMA - (compression_tier * 2), 2)
    try:
        from scipy.signal import find_peaks

        peak_indices, peak_props = find_peaks(values)
        valley_indices, valley_props = find_peaks(-values)
        peaks = _extrema_rows(
            values=values,
            index_values=index_values,
            indices=peak_indices,
            properties=peak_props,
            time_values=time_values,
            limit=max_items,
        )
        valleys = _extrema_rows(
            values=values,
            index_values=index_values,
            indices=valley_indices,
            properties=valley_props,
            time_values=time_values,
            limit=max_items,
        )
    except Exception:
        peaks = _fallback_extrema(
            values=values, index_values=index_values, time_values=time_values, limit=max_items, valleys=False
        )
        valleys = _fallback_extrema(
            values=values, index_values=index_values, time_values=time_values, limit=max_items, valleys=True
        )
    return peaks, valleys


def _extrema_rows(
    *,
    values: np.ndarray,
    index_values: np.ndarray,
    indices: np.ndarray,
    properties: dict[str, np.ndarray],
    time_values: np.ndarray | None,
    limit: int,
) -> list[dict[str, float]]:
    prominences = properties.get("prominences")
    if prominences is None:
        prominences = np.zeros(len(indices), dtype=float)
    rows = [
        {
            "index": float(index),
            "source_index": float(index_values[index]),
            "time": float(time_values[index]) if time_values is not None else float(index),
            "value": float(values[index]),
            "prominence": float(prominence),
        }
        for index, prominence in zip(indices, prominences, strict=False)
    ]
    rows.sort(key=lambda row: abs(float(row["prominence"])) or abs(float(row["value"])), reverse=True)
    return rows[:limit]


def _fallback_extrema(
    *,
    values: np.ndarray,
    index_values: np.ndarray,
    time_values: np.ndarray | None,
    limit: int,
    valleys: bool,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for index in range(1, len(values) - 1):
        center = values[index]
        left = values[index - 1]
        right = values[index + 1]
        if valleys:
            is_extreme = center < left and center < right
        else:
            is_extreme = center > left and center > right
        if not is_extreme:
            continue
        rows.append(
            {
                "index": float(index),
                "source_index": float(index_values[index]),
                "time": float(time_values[index]) if time_values is not None else float(index),
                "value": float(center),
                "prominence": float(abs(center - ((left + right) / 2))),
            }
        )
    rows.sort(key=lambda row: abs(float(row["prominence"])), reverse=True)
    return rows[:limit]


def _detect_inflections(*, values: np.ndarray, index_values: np.ndarray, compression_tier: int) -> list[int]:
    try:
        from scipy.signal import savgol_filter

        window_length = _savgol_window_length(len(values))
        polyorder = 3 if window_length >= 5 else 2
        second = savgol_filter(values, window_length=window_length, polyorder=polyorder, deriv=2)
        change_indices = np.where(np.diff(np.sign(second)) != 0)[0]
    except Exception:
        second = np.gradient(np.gradient(values))
        change_indices = np.where(np.diff(np.sign(second)) != 0)[0]
    limit = max(24 - (compression_tier * 6), 6)
    return [int(index_values[int(value)]) for value in change_indices[:limit]]


def _rolling_mann_kendall(*, values: np.ndarray, index_values: np.ndarray, compression_tier: int) -> dict[str, object]:
    try:
        import pymannkendall as mk
    except Exception:
        return {"status": "unavailable", "reason": "pymannkendall not installed"}

    window = max(5, math.ceil(len(values) * 0.1))
    if window >= len(values):
        window = max(3, len(values) // 2)
    if window < 3:
        return {"status": "too_short"}

    rows: list[dict[str, object]] = []
    for start in range(0, len(values) - window + 1, window):
        segment = values[start : start + window]
        if len(segment) < 3:
            continue
        if np.unique(segment).size < 2:
            continue
        try:
            result = mk.original_test(segment)
        except Exception:
            continue
        rows.append(
            {
                "start_index": int(index_values[start]),
                "end_index": int(index_values[start + len(segment) - 1]),
                "trend": str(result.trend),
                "tau": float(result.Tau),
                "slope": float(result.slope),
                "p": float(result.p),
                "significant": bool(result.h),
            }
        )
    limit = max(DEFAULT_MAX_WINDOWS - (compression_tier * 2), 3)
    return {"status": "ok", "window_length": window, "windows": rows[:limit]}


def _detect_change_points(*, values: np.ndarray, index_values: np.ndarray, compression_tier: int) -> list[int]:
    reduced_values, reduced_indices = _downsample_heavy_signal_inputs(values=values, index_values=index_values)
    try:
        import ruptures as rpt
    except Exception:
        return []
    try:
        algo = rpt.Pelt(model="rbf").fit(reduced_values.reshape(-1, 1))
        result = [int(reduced_indices[min(point, len(reduced_indices) - 1)]) for point in algo.predict(pen=10)]
    except Exception:
        return []
    limit = max(DEFAULT_MAX_CHANGEPOINTS - (compression_tier * 2), 3)
    return result[:limit]


def _detect_oscillation(*, values: np.ndarray, index_values: np.ndarray, compression_tier: int) -> dict[str, object]:
    reduced_values, reduced_indices = _downsample_heavy_signal_inputs(values=values, index_values=index_values)
    try:
        import pywt
    except Exception:
        return {"status": "unavailable", "reason": "PyWavelets not installed"}
    scales = np.arange(1, min(64, max(4, len(reduced_values) // 2)))
    if scales.size == 0:
        return {"status": "too_short"}
    try:
        coefficients, _frequencies = pywt.cwt(reduced_values, scales, "morl", sampling_period=1.0)
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}
    power = np.abs(coefficients) ** 2
    mean_power = power.mean(axis=1)
    dominant_idx = int(np.argmax(mean_power))
    dominant_scale = int(scales[dominant_idx])
    dominant_power = power[dominant_idx]
    threshold = 0.5 * float(np.max(dominant_power))
    oscillation_indices = np.where(dominant_power > threshold)[0]
    limit = max(DEFAULT_MAX_OSCILLATION_INDICES - (compression_tier * 5), 5)
    return {
        "status": "ok",
        "dominant_period": float(dominant_scale),
        "oscillation_indices": [int(reduced_indices[int(index)]) for index in oscillation_indices[:limit]],
    }


def _downsample_heavy_signal_inputs(
    *,
    values: np.ndarray,
    index_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= MAX_HEAVY_SIGNAL_POINTS:
        return values, index_values
    step = math.ceil(len(values) / MAX_HEAVY_SIGNAL_POINTS)
    return values[::step], index_values[::step]


def _savgol_window_length(series_length: int) -> int:
    proposed = max(5, math.ceil(series_length * 0.1))
    if proposed % 2 == 0:
        proposed += 1
    return min(proposed, series_length if series_length % 2 == 1 else series_length - 1)


def _render_summary_text(
    *,
    reporter_pattern: str,
    time_column: str | None,
    matched_series_count: int,
    summaries: Iterable[StatisticalSeriesSummary],
    compression_tier: int,
) -> str:
    lines = [
        f"Statistical evidence for simulation series matching `{reporter_pattern}`.",
        f"Compression tier: {compression_tier}.",
    ]
    if time_column is not None:
        lines.append(f"Time axis column: {time_column}.")
    if matched_series_count > 1:
        lines.append(
            "Matched series count: "
            f"{matched_series_count}. Detailed signal analysis below is computed on the tick-wise "
            "mean of the matched series."
        )
    for summary in summaries:
        lines.extend(
            [
                "",
                f"Series: {summary.column}",
                f"- samples: {summary.sample_count}",
                f"- valid numeric samples: {summary.valid_sample_count}",
                f"- dropped non-numeric samples: {summary.dropped_non_numeric_count}",
                f"- start value: {summary.start_value:.4f}",
                f"- end value: {summary.end_value:.4f}",
                f"- mean: {summary.mean:.4f}",
                f"- median: {summary.median:.4f}",
                f"- std: {summary.std:.4f}",
                f"- variance: {summary.variance:.4f}",
                f"- minimum value: {summary.minimum:.4f}",
                f"- minimum index: {summary.minimum_index}",
                f"- maximum value: {summary.maximum:.4f}",
                f"- maximum index: {summary.maximum_index}",
                "- peaks:",
                *_render_extrema(summary.peaks, "peak"),
                "- valleys:",
                *_render_extrema(summary.valleys, "valley"),
                *_render_inflections(summary.inflection_indices),
                *_render_mann_kendall(summary.rolling_mann_kendall),
                *_render_int_items(summary.change_points, "change point"),
                *_render_oscillation(summary.oscillation),
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _render_extrema(rows: list[dict[str, float]], label: str) -> list[str]:
    if not rows:
        return [f"  - {label} none detected"]
    return [
        f"  - {label} {index}: index {int(row['index'])} value {row['value']:.4f} prominence {row['prominence']:.4f}"
        for index, row in enumerate(rows, start=1)
    ]


def _render_inflections(values: list[int]) -> list[str]:
    if not values:
        return ["- inflection index none detected"]
    return [f"- inflection index {index}: {value}" for index, value in enumerate(values, start=1)]


def _render_int_items(values: list[int], label: str) -> list[str]:
    if not values:
        return [f"- {label} none detected"]
    return [f"- {label} {index}: {value}" for index, value in enumerate(values, start=1)]


def _render_mann_kendall(payload: dict[str, object]) -> list[str]:
    status = str(payload.get("status", "unknown"))
    if status != "ok":
        return [f"- rolling Mann-Kendall: {status}"]
    windows = payload.get("windows")
    if not isinstance(windows, list) or not windows:
        return ["- rolling Mann-Kendall: no windows analysed"]
    lines: list[str] = ["- rolling Mann-Kendall status: ok"]
    for index, window in enumerate(windows, start=1):
        if not isinstance(window, dict):
            continue
        lines.append(
            f"- rolling Mann-Kendall window {index}: start={int(window['start_index'])} "
            f"end={int(window['end_index'])} trend={window['trend']} "
            f"tau={float(window['tau']):.3f} slope={float(window['slope']):.3f} "
            f"p={float(window['p']):.3f} significant={window['significant']}"
        )
    return lines


def _render_oscillation(payload: dict[str, object]) -> list[str]:
    status = str(payload.get("status", "unknown"))
    if status != "ok":
        return [f"- oscillation status: {status}"]
    dominant_period_value = payload.get("dominant_period", 0.0)
    dominant_period = float(dominant_period_value) if isinstance(dominant_period_value, (int, float)) else 0.0
    lines: list[str] = [f"- oscillation dominant period: {dominant_period:.2f}"]
    indices = payload.get("oscillation_indices")
    if not isinstance(indices, list) or not indices:
        lines.append("- oscillation indices: none detected")
        return lines
    for index, value in enumerate(indices, start=1):
        if isinstance(value, (int, float)):
            lines.append(f"- oscillation index {index}: {int(value)}")
    return lines
