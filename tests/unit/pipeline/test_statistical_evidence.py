from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from distill_abm.pipeline.statistical_evidence import (
    build_statistical_evidence,
    render_evidence_artifacts,
    select_plot_relevant_frame,
)


def test_select_plot_relevant_frame_keeps_only_matched_columns_and_step() -> None:
    frame = pd.DataFrame(
        {
            "[step]": [0, 1, 2],
            "metric-a": [1.0, 2.0, 3.0],
            "metric-b": [4.0, 5.0, 6.0],
            "other": [7.0, 8.0, 9.0],
        }
    )

    selected, time_column = select_plot_relevant_frame(frame=frame, reporter_pattern="metric-a")

    assert time_column == "[step]"
    assert list(selected.columns) == ["[step]", "metric-a"]


def test_build_statistical_evidence_reports_only_matched_series() -> None:
    frame = pd.DataFrame(
        {
            "[step]": [0, 1, 2, 3, 4],
            "metric-a": [1.0, 3.0, 2.0, 4.0, 3.5],
            "other": [100, 100, 100, 100, 100],
        }
    )

    evidence = build_statistical_evidence(frame=frame, reporter_pattern="metric-a")

    assert evidence.matched_columns == ["metric-a"]
    assert "other" not in evidence.selected_frame_csv
    assert "Statistical evidence for simulation series matching `metric-a`." in evidence.summary_text
    assert "rolling Mann-Kendall:" in evidence.summary_text
    assert "change points:" in evidence.summary_text
    assert "oscillation summary:" in evidence.summary_text


def test_build_statistical_evidence_aggregates_multi_series_matches() -> None:
    frame = pd.DataFrame(
        {
            "[step]": [0, 1, 2, 3],
            "metric-a": [1.0, 2.0, 3.0, 4.0],
            "metric-a.1": [2.0, 3.0, 4.0, 5.0],
            "metric-a.2": [3.0, 4.0, 5.0, 6.0],
            "other": [100.0, 100.0, 100.0, 100.0],
        }
    )

    evidence = build_statistical_evidence(frame=frame, reporter_pattern="metric-a")

    assert evidence.matched_columns == ["metric-a", "metric-a.1", "metric-a.2"]
    assert evidence.summary_payload["matched_series_count"] == 3
    assert evidence.summary_payload["series_summary_mode"] == "aggregate_mean"
    series_payload = cast(list[dict[str, Any]], evidence.summary_payload["series"])
    assert len(series_payload) == 1
    assert series_payload[0]["column"] == "aggregate_mean_of_3_matched_series"
    assert "Matched series count: 3." in evidence.summary_text
    assert "other" not in evidence.selected_frame_csv


def test_build_statistical_evidence_handles_unmatched_metric_pattern() -> None:
    frame = pd.DataFrame({"[step]": [0, 1], "x": [1.0, 2.0]})

    evidence = build_statistical_evidence(frame=frame, reporter_pattern="missing")

    assert evidence.matched_columns == []
    assert evidence.summary_payload["status"] == "unmatched_metric_pattern"
    assert "No simulation series matched" in evidence.summary_text


def test_render_evidence_artifacts_writes_text_json_and_series_csv(tmp_path: Path) -> None:
    frame = pd.DataFrame({"[step]": [0, 1, 2], "metric-a": [1.0, 2.0, 3.0]})
    evidence = build_statistical_evidence(frame=frame, reporter_pattern="metric-a")

    summary_path, payload_path, series_path = render_evidence_artifacts(
        evidence=evidence,
        output_dir=tmp_path,
        stem="trend_evidence_table",
    )

    assert summary_path.read_text(encoding="utf-8") == evidence.summary_text
    assert json.loads(payload_path.read_text(encoding="utf-8"))["matched_columns"] == ["metric-a"]
    assert "metric-a" in series_path.read_text(encoding="utf-8")


def test_build_statistical_evidence_skips_degenerate_mann_kendall_windows() -> None:
    frame = pd.DataFrame(
        {
            "[step]": list(range(20)),
            "metric-a": [5.0] * 20,
        }
    )

    evidence = build_statistical_evidence(frame=frame, reporter_pattern="metric-a")

    series_payload = cast(list[dict[str, Any]], evidence.summary_payload["series"])
    first_series = series_payload[0]
    rolling = cast(dict[str, Any], first_series["rolling_mann_kendall"])
    assert rolling["status"] == "ok"
    assert isinstance(rolling["windows"], list)


def test_build_statistical_evidence_handles_mann_kendall_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "[step]": list(range(20)),
            "metric-a": [float(index) for index in range(20)],
        }
    )

    class FakeMk:
        @staticmethod
        def original_test(_segment):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "pymannkendall", SimpleNamespace(original_test=FakeMk.original_test))

    evidence = build_statistical_evidence(frame=frame, reporter_pattern="metric-a")

    series_payload = cast(list[dict[str, Any]], evidence.summary_payload["series"])
    first_series = series_payload[0]
    rolling = cast(dict[str, Any], first_series["rolling_mann_kendall"])
    assert rolling["status"] == "ok"
    assert rolling["windows"] == []


def test_build_statistical_evidence_coerces_non_numeric_runtime_markers() -> None:
    frame = pd.DataFrame(
        {
            "[step]": [0, 1, 2, 3],
            "metric-a": [1.0, "<RuntimePrimitiveException>", 3.0, 4.0],
            "other": [9.0, 9.0, 9.0, 9.0],
        }
    )

    evidence = build_statistical_evidence(frame=frame, reporter_pattern="metric-a")

    series_payload = cast(list[dict[str, Any]], evidence.summary_payload["series"])
    first_series = series_payload[0]
    assert first_series["sample_count"] == 4
    assert first_series["valid_sample_count"] == 3
    assert first_series["dropped_non_numeric_count"] == 1
    assert "dropped non-numeric samples: 1" in evidence.summary_text
    assert "other" not in evidence.selected_frame_csv


def test_build_statistical_evidence_downsamples_heavy_signal_routines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "[step]": list(range(2000)),
            "metric-a": [float(index % 17) for index in range(2000)],
        }
    )
    seen: dict[str, int] = {}

    class FakeAlgo:
        def fit(self, values):  # type: ignore[no-untyped-def]
            seen["ruptures_len"] = len(values)
            return self

        def predict(self, pen):  # type: ignore[no-untyped-def]
            assert pen == 10
            return [1]

    class FakeRuptures:
        @staticmethod
        def pelt(model):  # type: ignore[no-untyped-def]
            assert model == "rbf"
            return FakeAlgo()

    class FakePywt:
        @staticmethod
        def cwt(values, scales, wavelet, sampling_period=1.0):  # type: ignore[no-untyped-def]
            seen["wavelet_len"] = len(values)
            assert wavelet == "morl"
            coefficients = np.ones((len(scales), len(values)), dtype=float)
            frequencies = np.ones(len(scales), dtype=float)
            return coefficients, frequencies

    monkeypatch.setitem(sys.modules, "ruptures", SimpleNamespace(Pelt=FakeRuptures.pelt))
    monkeypatch.setitem(sys.modules, "pywt", FakePywt())

    build_statistical_evidence(frame=frame, reporter_pattern="metric-a")

    assert seen["ruptures_len"] <= 512
    assert seen["wavelet_len"] <= 512
