"""Small, reusable building blocks for notebook-equivalent NetLogo execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import pandas as pd

ParameterScalar = bool | int | float | str


class NetLogoLinkProtocol(Protocol):
    """Subset of NetLogoLink used by the workflow helpers."""

    def load_model(self, model_path: str) -> None: ...

    def command(self, command: str) -> None: ...

    def report(self, reporter: str) -> Any: ...


def coerce_parameter_value_for_netlogo(value: ParameterScalar) -> ParameterScalar:
    """Normalize booleans to NetLogo literals while keeping other values unchanged."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def collect_reporter_rows(
    *,
    netlogo: NetLogoLinkProtocol,
    reporters: Sequence[str],
    max_ticks: int,
    interval: int,
) -> list[dict[str, Any]]:
    """Run reporters for all requested ticks and return row dictionaries."""
    rows: list[dict[str, Any]] = []
    for tick in range(0, max_ticks, interval):
        netlogo.command("go")
        row = {reporter: netlogo.report(reporter) for reporter in reporters}
        row["tick"] = tick
        rows.append(row)
    return rows


def run_single_repetition(
    *,
    netlogo: NetLogoLinkProtocol,
    reporters: Sequence[str],
    max_ticks: int,
    interval: int,
    experiment_parameters: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run a single notebook-style simulation repetition and return a frame."""
    if experiment_parameters:
        for param, value in experiment_parameters.items():
            netlogo.command(f"set {param} {coerce_parameter_value_for_netlogo(value)}")
    netlogo.command("setup")
    rows = collect_reporter_rows(netlogo=netlogo, reporters=reporters, max_ticks=max_ticks, interval=interval)
    frame = pd.DataFrame(rows)
    frame.columns = [f"{column}" for column in frame.columns]
    return frame
