"""Pure rendering helpers for quantitative smoke publication tables."""

from __future__ import annotations

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
        values = [_format_contribution_cell(float(row[column])) for column in METRIC_COLUMN_NAMES]
        body.append("| " + " | ".join([str(row["Feature"]), *values]) + " |")
    return "# Factorial Contributions\n\n" + "\n".join([header, *body]) + "\n"


def render_factorial_latex_table(frame: pd.DataFrame) -> str:
    latex_rows = []
    for row in frame.to_dict(orient="records"):
        formatted_values = [_latex_format_contribution(float(row[column])) for column in METRIC_COLUMN_NAMES]
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
        "| ABM | Summary | LLM | BLEU | METEOR | R-1 | R-2 | R-L | Reading ease |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    body = [
        "| "
        + " | ".join([row["ABM"], row["Summary"], row["LLM"], *(row[metric] for metric in METRIC_COLUMN_NAMES)])
        + " |"
        for row in rows
    ]
    return "# Best Score Across Dynamic Prompt Elements\n\n" + "\n".join([header, *body]) + "\n"


def render_optimal_latex_table(rows: list[dict[str, str]]) -> str:
    latex_rows = [
        " \\hline\n"
        + " {} & {} & {} & {} \\\\".format(
            _latex_escape(row["ABM"]),
            _latex_escape(row["Summary"]),
            _latex_escape(row["LLM"]),
            " & ".join(_latex_escape(row[metric]) for metric in METRIC_COLUMN_NAMES),
        )
        for row in rows
    ]
    return (
        "\\begin{tabular}{|l|l|l|l|l|l|l|l|l|}\n\\hline\n"
        "\\textit{\\textbf{ABM}} & \\textit{\\textbf{Summary}} & \\textit{\\textbf{LLM}}"
        " & BLEU & METEOR & R-1 & R-2 & R-L & Reading ease \\\\\n"
        + "\n".join(latex_rows)
        + "\n\\hline\n\\end{tabular}\n"
    )


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


def _format_contribution_cell(value: float) -> str:
    rendered = f"{value:.2f}"
    return f"**{rendered}**" if value > 5 else rendered


def _latex_format_contribution(value: float) -> str:
    rendered = f"{value:.2f}"
    return f"\\textbf{{{rendered}}}" if value > 5 else rendered


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("<", "\\textless{}")
    )
