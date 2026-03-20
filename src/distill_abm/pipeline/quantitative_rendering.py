"""Pure rendering helpers for quantitative smoke publication tables."""

from __future__ import annotations

import math

import pandas as pd

ABSENT_MARKER = "—"
METRIC_COLUMN_NAMES: tuple[str, ...] = (
    "BLEU",
    "METEOR",
    "R-1",
    "R-2",
    "R-L",
    "Reading ease",
)


def render_anova_markdown_table(rows: list[dict[str, float | str | None]]) -> str:
    header = (
        "| Variable / metric | BLEU | METEOR | R-1 | R-2 | R-L | Reading ease |\n"
        "| --- | --- | --- | --- | --- | --- | --- |"
    )
    body = [
        "| "
        + " | ".join(
            [
                str(row["label"]),
                *(_format_pvalue_cell(_lookup_metric_value(row, metric)) for metric in METRIC_COLUMN_NAMES),
            ]
        )
        + " |"
        for row in rows
    ]
    return "# ANOVA Table\n\n" + "\n".join([header, *body]) + "\n"


def render_anova_latex_table(rows: list[dict[str, float | str | None]]) -> str:
    latex_rows = [
        " \\hline\n"
        + " {} & {} \\\\".format(
            _latex_escape(str(row["label"])),
            " & ".join(
                _latex_escape(_format_pvalue_cell(_lookup_metric_value(row, metric))) for metric in METRIC_COLUMN_NAMES
            ),
        )
        for row in rows
    ]
    return (
        "\\begin{tabular}{|l|l|l|l|l|l|l|}\n\\hline\n"
        "\\textit{$\\downarrow$Variable / metric $\\rightarrow$} & BLEU & METEOR & R-1 & R-2 & R-L"
        " & Reading ease \\\\\n" + "\n".join(latex_rows) + "\n\\hline\n\\end{tabular}\n"
    )


def render_factorial_markdown_table(frame: pd.DataFrame) -> str:
    header = "| Feature | BLEU | METEOR | R-1 | R-2 | R-L | Reading ease |\n| --- | --- | --- | --- | --- | --- | --- |"
    body = []
    for row in frame.to_dict(orient="records"):
        values = [_format_contribution_cell(row[column]) for column in METRIC_COLUMN_NAMES]
        body.append("| " + " | ".join([str(row["Feature"]), *values]) + " |")
    return "# Factorial Contributions\n\n" + "\n".join([header, *body]) + "\n"


def render_factorial_latex_table(frame: pd.DataFrame) -> str:
    latex_rows = []
    for row in frame.to_dict(orient="records"):
        formatted_values = [_latex_format_contribution(row[column]) for column in METRIC_COLUMN_NAMES]
        latex_rows.append(
            " \\hline\n"
            + " {} & {} \\\\".format(
                _latex_escape(str(row["Feature"]).replace("_AND_", " and ")),
                " & ".join(formatted_values),
            )
        )
    return (
        "\\begin{tabular}{|l|l|l|l|l|l|l|}\n\\hline\n"
        "\\textbf{Feature} & BLEU & METEOR & R-1 & R-2 & R-L & Reading ease \\\\\n"
        + "\n".join(latex_rows)
        + "\n\\hline\n\\end{tabular}\n"
    )


def render_optimal_markdown_table(rows: list[dict[str, str]]) -> str:
    header = (
        "| Reference family | ABM | Summary | LLM | BLEU | METEOR | R-1 | R-2 | R-L | Reading ease |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    body = [
        "| "
        + " | ".join(
            [
                row["Reference family"],
                row["ABM"],
                row["Summary"],
                row["LLM"],
                *(row[metric] for metric in METRIC_COLUMN_NAMES),
            ]
        )
        + " |"
        for row in rows
    ]
    return "# Best Score Across Dynamic Prompt Elements\n\n" + "\n".join([header, *body]) + "\n"


def render_optimal_latex_table(rows: list[dict[str, str]]) -> str:
    latex_rows = [
        " \\hline\n"
        + " {} & {} & {} & {} & {} \\\\".format(
            _latex_escape(row["Reference family"]),
            _latex_escape(row["ABM"]),
            _latex_escape(row["Summary"]),
            _latex_escape(row["LLM"]),
            " & ".join(_latex_escape(row[metric]) for metric in METRIC_COLUMN_NAMES),
        )
        for row in rows
    ]
    return (
        "\\begin{tabular}{|l|l|l|l|l|l|l|l|l|l|}\n\\hline\n"
        "\\textit{\\textbf{Reference family}} & \\textit{\\textbf{ABM}} & \\textit{\\textbf{Summary}}"
        " & \\textit{\\textbf{LLM}} & BLEU & METEOR & R-1 & R-2 & R-L & Reading ease \\\\\n"
        + "\n".join(latex_rows)
        + "\n\\hline\n\\end{tabular}\n"
    )


def render_evidence_summary_markdown_table(rows: list[dict[str, str]]) -> str:
    header = (
        "| Evidence | ABM | Avg BLEU | Avg METEOR | Avg R-1 | Avg R-2 | Avg R-L | Avg Reading ease |"
        " Best BLEU | Best METEOR | Best R-1 | Best R-2 | Best R-L | Best Reading ease |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    sections = ["# Evidence Mode Summary", ""]
    for reference_family in sorted({row["Reference family"] for row in rows}):
        sections.append(f"## {reference_family}")
        sections.append("")
        sections.append(header)
        for row in rows:
            if row["Reference family"] != reference_family:
                continue
            sections.append(
                "| "
                + " | ".join(
                    [
                        row["Evidence"],
                        row["ABM"],
                        row["Avg BLEU"],
                        row["Avg METEOR"],
                        row["Avg R-1"],
                        row["Avg R-2"],
                        row["Avg R-L"],
                        row["Avg Reading ease"],
                        row["Best BLEU"],
                        row["Best METEOR"],
                        row["Best R-1"],
                        row["Best R-2"],
                        row["Best R-L"],
                        row["Best Reading ease"],
                    ]
                )
                + " |"
            )
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def _format_pvalue_cell(value: float | None) -> str:
    if value is None:
        return ABSENT_MARKER
    if value < 0.01:
        return "<0.01"
    return f"{value:.2f}"


def _lookup_metric_value(row: dict[str, float | str | None], metric: str) -> float | None:
    aliases = {
        "R-1": ("R-1", "ROUGE-1"),
        "R-2": ("R-2", "ROUGE-2"),
        "R-L": ("R-L", "ROUGE-L"),
    }
    for key in aliases.get(metric, (metric,)):
        if key in row:
            value = row[key]
            return value if isinstance(value, float) or value is None else None
    return None


def _format_contribution_cell(value: float | int | None) -> str:
    numeric_value = _coerce_optional_float(value)
    if numeric_value is None:
        return ABSENT_MARKER
    if 0 < abs(numeric_value) < 0.01:
        rendered = "<0.01"
        return f"**{rendered}**" if numeric_value > 5 else rendered
    rendered = f"{numeric_value:.2f}"
    return f"**{rendered}**" if numeric_value > 5 else rendered


def _latex_format_contribution(value: float | int | None) -> str:
    numeric_value = _coerce_optional_float(value)
    if numeric_value is None:
        return _latex_escape(ABSENT_MARKER)
    if 0 < abs(numeric_value) < 0.01:
        rendered = _latex_escape("<0.01")
        return f"\\textbf{{{rendered}}}" if numeric_value > 5 else rendered
    rendered = f"{numeric_value:.2f}"
    return f"\\textbf{{{rendered}}}" if numeric_value > 5 else rendered


def _coerce_optional_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    numeric_value = float(value)
    if math.isnan(numeric_value):
        return None
    return numeric_value


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("<", "\\textless{}")
    )
