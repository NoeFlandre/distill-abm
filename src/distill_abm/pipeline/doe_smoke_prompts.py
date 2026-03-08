"""Legacy-aligned DOE smoke prompt builders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from distill_abm.ingest.csv_ingest import matching_columns

CONTEXT_PLACEHOLDER = "<<context_response_from_context_llm>>"
LEGACY_EXAMPLE_TEXT = (
    "Here is an example of the style of the report: “ The total number of earthquakes recorded in the region "
    "starts at 5,000 then it declines rapidly over the first 400 time steps. This initial decline could be "
    "attributed to the depletion of immediate stress points within the fault lines, as the most susceptible areas "
    "release their pent-up energy early in the simulation. It increases briefly at around 500 steps, likely due "
    "to the aftershocks and secondary stress points being triggered as the system seeks a new equilibrium. The "
    "simulation ends with the number of earthquakes near zero, indicating that the majority of the stress has been "
    "released and the system has stabilized.\n\nThe total seismic activity follows the same pattern, starting at "
    "10,000 events and declining steadily. This steady decline reflects a systematic reduction in seismic energy "
    "over time, as the energy distribution within the tectonic plates becomes more uniform. There is low variance "
    "across simulation runs in the first 100 steps, but deviations become noticeable afterward. This suggests that "
    "while the initial reactions to stress are highly predictable, as time progresses, the system's complexity "
    "introduces more variability. This variability could be due to differences in secondary stress points and the "
    "non-linear nature of seismic energy release over time.“"
)
LEGACY_INSIGHTS_TEXT = "When summarizing trends, provide brief insights about their implications for decision makers."
LEGACY_CONTEXT_PROMPT_TEMPLATE = (
    "Your goal is to explain an agent-based model. Your explanation must include the context of the model, its goals, "
    "and key parameters. For each parameter, include the range of values (if provided) and the value that we have set. "
    "Do not write any summary or conclusion. {parameters}\n\n"
    "The context and goals of the model are as follows. {documentation}"
)
LEGACY_TREND_PROMPT_TEMPLATE = (
    "We have a plot from repeated simulations of an agent based model. Your goal is to describe the trends in details "
    "from the plot, mentioning key time steps and values taken by the metric in the plot, and interpreting them based "
    "on the context of the model. The report must objectively describe the trends in the data without addressing the "
    "quality of the simulation. Do not refer to the plot or any visual in your description. If a plot has very simple "
    "dynamics, simply state them without expanding."
)
LEGACY_ABM_ROLE_TEXTS: dict[str, tuple[str, str]] = {
    "fauna": (
        "You are an expert in megaherbivore extinction without any statistics background.",
        "You are an expert in megaherbivore extinction with a statistic background.",
    ),
    "grazing": (
        "You are an expert in grazing systems without any statistics background.",
        "You are an expert in grazing systems with a statistic background.",
    ),
    "milk_consumption": (
        "You are an expert in Consumer Behavior without any statistics background.",
        "You are an expert in Consumer Behavior with a statistic background.",
    ),
}


def legacy_role_texts(abm: str) -> tuple[str, str]:
    try:
        return LEGACY_ABM_ROLE_TEXTS[abm]
    except KeyError as exc:
        raise ValueError(f"missing legacy DOE role text for ABM '{abm}'") from exc


def build_legacy_doe_context_prompt(
    *,
    abm: str,
    inputs_csv_path: Path,
    inputs_doc_path: Path,
    enabled: set[str] | None,
) -> str:
    parameters = inputs_csv_path.read_text(encoding="utf-8")
    documentation = inputs_doc_path.read_text(encoding="utf-8")
    base = LEGACY_CONTEXT_PROMPT_TEMPLATE.format(parameters=parameters, documentation=documentation)
    context_role, _trend_role = legacy_role_texts(abm)
    if enabled is not None and "role" not in enabled:
        return base
    return f"{context_role}\n\n{base}"


def build_legacy_doe_trend_prompt(
    *,
    abm: str,
    context_response: str,
    plot_description: str,
    evidence_mode: str,
    table_csv: str,
    enabled: set[str] | None,
) -> str:
    parts: list[str] = []
    active = enabled or set()
    _context_role, trend_role = legacy_role_texts(abm)
    if enabled is None or "role" in active:
        parts.append(trend_role)
    parts.append(legacy_trend_prompt_for_evidence_mode(evidence_mode))
    parts.append(f"The context and goals of the model are as follows. {context_response}")
    if enabled is None or "example" in active:
        parts.append(LEGACY_EXAMPLE_TEXT)
    if enabled is None or "insights" in active:
        parts.append(LEGACY_INSIGHTS_TEXT)
    if plot_description.strip():
        parts.append(
            legacy_plot_description_for_evidence_mode(
                plot_description=plot_description,
                evidence_mode=evidence_mode,
            )
        )
    if evidence_mode in {"table", "plot+table"}:
        parts.append(f"Relevant simulation columns (CSV):\n{table_csv}")
    return "\n\n".join(parts)


def build_raw_table_csv(*, frame: pd.DataFrame, reporter_pattern: str) -> str:
    columns = [str(column) for column in frame.columns]
    matched = matching_columns(columns, include_pattern=reporter_pattern)
    if not matched:
        return "unmatched metric pattern\n"
    leading_column = preferred_time_column(columns)
    selected = [leading_column, *matched] if leading_column is not None else matched
    return frame[selected].to_csv(index=False)


def preferred_time_column(columns: list[str]) -> str | None:
    for candidate in ("[step]", "time_step", "step"):
        if candidate in columns:
            return candidate
    return None


def legacy_trend_prompt_for_evidence_mode(evidence_mode: str) -> str:
    if evidence_mode == "plot":
        return LEGACY_TREND_PROMPT_TEMPLATE
    if evidence_mode == "table":
        return (
            LEGACY_TREND_PROMPT_TEMPLATE.replace("a plot", "a data table", 1)
            .replace("from the plot", "from the data table", 1)
            .replace("the plot", "the data table")
            .replace("If a plot has very simple dynamics", "If a data table has very simple dynamics")
        )
    if evidence_mode == "plot+table":
        return LEGACY_TREND_PROMPT_TEMPLATE.replace(
            (
                "We have a plot from repeated simulations of an agent based model. "
                "Your goal is to describe the trends in details from the plot, "
                "mentioning key time steps and values taken by the metric in the plot, "
                "and interpreting them based on the context of the model. "
                "The report must objectively describe the trends in the data without "
                "addressing the quality of the simulation. "
                "Do not refer to the plot or any visual in your description. "
                "If a plot has very simple dynamics, simply state them without expanding."
            ),
            (
                "We have a plot and a data table from repeated simulations of an agent based model. "
                "Your goal is to describe the trends in details from the plot and the data table, "
                "mentioning key time steps and values taken by the metric in the plot and the data table, "
                "and interpreting them based on the context of the model. "
                "The report must objectively describe the trends in the data without "
                "addressing the quality of the simulation. "
                "Do not refer to the plot and the data table or any visual in your description. "
                "If the plot and data table have very simple dynamics, simply state them without expanding."
            ),
        )
    raise ValueError(f"unsupported DOE evidence mode: {evidence_mode}")


def legacy_plot_description_for_evidence_mode(*, plot_description: str, evidence_mode: str) -> str:
    stripped = plot_description.strip()
    if evidence_mode == "plot":
        return stripped
    if evidence_mode == "table":
        return (
            stripped.replace("The attachment is the plot representing", "The data table represents", 1)
            .replace("This plot represents", "This data table represents", 1)
        )
    if evidence_mode == "plot+table":
        return (
            stripped.replace(
                "The attachment is the plot representing",
                "The attachment includes the plot, and the data table represents",
                1,
            ).replace(
                "This plot represents",
                "This plot and data table represent",
                1,
            )
        )
    raise ValueError(f"unsupported DOE evidence mode: {evidence_mode}")
