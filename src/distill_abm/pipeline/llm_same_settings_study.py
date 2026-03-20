"""Side-study analysis for comparing multiple LLMs on one anchor model's exact settings."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pandas as pd
from pydantic import BaseModel

from distill_abm.pipeline.quantitative_rendering import METRIC_COLUMN_NAMES
from distill_abm.pipeline.report_writers import write_model_report_files
from distill_abm.pipeline.run_artifact_contracts import latest_run_pointer_path, resolve_run_root

__all__ = [
    "LlmSameSettingsStudyResult",
    "run_llm_same_settings_study",
]

STUDY_REPORT_FILENAME = "llm_same_settings_study_report.json"
STUDY_SUMMARIZERS = frozenset({"bart", "bert", "t5"})
COMPARISON_KEY_COLUMNS = (
    "abm",
    "reference_family",
    "evidence",
    "role",
    "insights",
    "example",
    "summarizer",
    "repetition",
)
DISPLAY_MODEL_ORDER = ("gemini", "kimi", "qwen", "opus")
MODEL_LABEL_ALIASES = {
    "google/gemini-3.1-pro-preview": "gemini",
    "moonshotai/kimi-k2.5": "kimi",
    "qwen/qwen3.5-27b": "qwen",
    "anthropic/claude-opus-4.6": "opus",
}
METRIC_SLUGS = {
    "BLEU": "bleu",
    "METEOR": "meteor",
    "R-1": "r_1",
    "R-2": "r_2",
    "R-L": "r_l",
    "Reading ease": "reading_ease",
}


class LlmSameSettingsStudyResult(BaseModel):
    """Structured result for the same-settings LLM comparison side study."""

    started_at_utc: str
    finished_at_utc: str
    output_root: Path
    run_id: str
    run_root: Path
    report_json_path: Path
    report_markdown_path: Path
    source_run_roots: dict[str, Path]
    comparable_case_count: int
    same_settings_long_path: Path
    master_comparison_path: Path
    metric_summary_path: Path
    metric_win_summary_path: Path


def run_llm_same_settings_study(
    *,
    anchor_source_root: Path,
    comparison_source_roots: list[Path],
    output_root: Path,
) -> LlmSameSettingsStudyResult:
    """Compare multiple LLMs after restricting all of them to the anchor model's settings."""
    started_at = datetime.now(UTC)
    run_id = started_at.strftime("run_%Y%m%d_%H%M%S_%f")
    run_root = output_root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    latest_run_pointer_path(output_root).write_text(str(run_root), encoding="utf-8")

    anchor_rows, anchor_run_root, anchor_label = _load_quantitative_rows(anchor_source_root)
    filtered_anchor = _filter_candidate_rows(anchor_rows)
    allowed_keys = cast(
        pd.DataFrame,
        filtered_anchor.loc[:, COMPARISON_KEY_COLUMNS].drop_duplicates().reset_index(drop=True),
    )

    filtered_frames: list[pd.DataFrame] = [_restrict_to_allowed_keys(filtered_anchor, allowed_keys)]
    source_run_roots = {anchor_label: anchor_run_root}

    for source_root in comparison_source_roots:
        frame, run_root_path, model_label = _load_quantitative_rows(source_root)
        restricted = _restrict_to_allowed_keys(_filter_candidate_rows(frame), allowed_keys)
        filtered_frames.append(restricted)
        source_run_roots[model_label] = run_root_path

    shared_keys = _build_shared_keys(filtered_frames)
    comparable_rows = pd.concat(
        [_restrict_to_allowed_keys(frame, shared_keys) for frame in filtered_frames],
        ignore_index=True,
    ).sort_values(["abm", "reference_family", "summarizer", "repetition", "model_label"]).reset_index(drop=True)

    master_comparison = _build_master_comparison(comparable_rows)
    metric_summary = _build_metric_summary(comparable_rows)
    metric_win_summary = _build_metric_win_summary(comparable_rows)

    same_settings_long_path = run_root / "same_settings_rows.csv"
    master_comparison_path = run_root / "master_comparison.csv"
    metric_summary_path = run_root / "metric_summary.csv"
    metric_win_summary_path = run_root / "metric_win_summary.csv"

    comparable_rows.to_csv(same_settings_long_path, index=False)
    master_comparison.to_csv(master_comparison_path, index=False)
    metric_summary.to_csv(metric_summary_path, index=False)
    metric_win_summary.to_csv(metric_win_summary_path, index=False)

    result = LlmSameSettingsStudyResult(
        started_at_utc=started_at.isoformat(),
        finished_at_utc=datetime.now(UTC).isoformat(),
        output_root=output_root,
        run_id=run_id,
        run_root=run_root,
        report_json_path=run_root / STUDY_REPORT_FILENAME,
        report_markdown_path=run_root / "llm_same_settings_study_report.md",
        source_run_roots=dict(sorted(source_run_roots.items())),
        comparable_case_count=len(master_comparison),
        same_settings_long_path=same_settings_long_path,
        master_comparison_path=master_comparison_path,
        metric_summary_path=metric_summary_path,
        metric_win_summary_path=metric_win_summary_path,
    )
    write_model_report_files(
        result=result,
        report_json_path=result.report_json_path,
        report_markdown_path=result.report_markdown_path,
        markdown=_render_markdown_report(
            result=result,
            metric_summary=metric_summary,
            metric_win_summary=metric_win_summary,
        ),
    )
    return result


def _load_quantitative_rows(source_root: Path) -> tuple[pd.DataFrame, Path, str]:
    resolved_root = resolve_run_root(source_root)
    frame = pd.read_csv(resolved_root / "combined" / "quantitative_rows.csv").copy()
    model_label = _infer_model_label(frame=frame, source_root=source_root, run_root=resolved_root)
    frame.insert(0, "model_label", model_label)
    return frame, resolved_root, model_label


def _infer_model_label(*, frame: pd.DataFrame, source_root: Path, run_root: Path) -> str:
    llm_values = frame["llm"].dropna().astype(str).unique().tolist()
    if len(llm_values) == 1 and llm_values[0] in MODEL_LABEL_ALIASES:
        return MODEL_LABEL_ALIASES[llm_values[0]]
    joined = " ".join([str(source_root), str(run_root), *llm_values]).lower()
    for token in DISPLAY_MODEL_ORDER:
        if token in joined:
            return token
    return run_root.parent.parent.parent.name if run_root.name.startswith("run_") else run_root.name


def _filter_candidate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    filtered = rows[rows["summarizer"].isin(STUDY_SUMMARIZERS)].copy()
    return filtered.reset_index(drop=True)


def _restrict_to_allowed_keys(rows: pd.DataFrame, allowed_keys: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or allowed_keys.empty:
        return rows.iloc[0:0].copy()
    merged = rows.merge(allowed_keys, on=list(COMPARISON_KEY_COLUMNS), how="inner")
    return merged.drop_duplicates(subset=["model_label", *COMPARISON_KEY_COLUMNS]).reset_index(drop=True)


def _build_shared_keys(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames or any(frame.empty for frame in frames):
        return pd.DataFrame(columns=COMPARISON_KEY_COLUMNS)
    shared = frames[0].loc[:, COMPARISON_KEY_COLUMNS].drop_duplicates()
    for frame in frames[1:]:
        shared = shared.merge(
            frame.loc[:, COMPARISON_KEY_COLUMNS].drop_duplicates(),
            on=list(COMPARISON_KEY_COLUMNS),
            how="inner",
        )
    return cast(pd.DataFrame, shared.reset_index(drop=True))


def _build_master_comparison(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        columns = list(COMPARISON_KEY_COLUMNS)
        for metric in METRIC_COLUMN_NAMES:
            metric_slug = METRIC_SLUGS[metric]
            for model_label in DISPLAY_MODEL_ORDER:
                columns.append(f"{model_label}_{metric_slug}")
            columns.append(f"winner_{metric_slug}")
        return pd.DataFrame(columns=columns)

    master = rows.loc[:, ["prompt", "model_label", *COMPARISON_KEY_COLUMNS, *METRIC_COLUMN_NAMES]].copy()
    prompt_frame = cast(
        pd.DataFrame,
        master.groupby(list(COMPARISON_KEY_COLUMNS), as_index=False)["prompt"]
        .first()
        .reset_index(drop=True),
    )
    result: pd.DataFrame = prompt_frame.copy()
    for metric in METRIC_COLUMN_NAMES:
        metric_slug = METRIC_SLUGS[metric]
        pivoted = master.pivot_table(
            index=list(COMPARISON_KEY_COLUMNS),
            columns="model_label",
            values=metric,
            aggfunc="first",
        ).reset_index()
        ordered_metric_columns: list[str] = []
        for model_label in DISPLAY_MODEL_ORDER:
            column_name = f"{model_label}_{metric_slug}"
            if model_label in pivoted.columns:
                pivoted = pivoted.rename(columns={model_label: column_name})
            else:
                pivoted[column_name] = pd.NA
            ordered_metric_columns.append(column_name)
        pivoted[f"winner_{metric_slug}"] = pivoted[ordered_metric_columns].apply(_winner_label, axis=1)
        result = result.merge(
            pivoted.loc[:, [*COMPARISON_KEY_COLUMNS, *ordered_metric_columns, f"winner_{metric_slug}"]],
            on=list(COMPARISON_KEY_COLUMNS),
            how="left",
        )
    return result.sort_values(["abm", "reference_family", "summarizer", "repetition"]).reset_index(drop=True)


def _build_metric_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "model_label",
                "mean_score",
                "median_score",
                "min_score",
                "max_score",
                "observations",
            ]
        )
    melted = rows.melt(
        id_vars=["model_label", *COMPARISON_KEY_COLUMNS],
        value_vars=list(METRIC_COLUMN_NAMES),
        var_name="metric",
        value_name="score",
    )
    summary = (
        melted.groupby(["metric", "model_label"], as_index=False)["score"]
        .agg(["mean", "median", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_score",
                "median": "median_score",
                "min": "min_score",
                "max": "max_score",
                "count": "observations",
            }
        )
    )
    summary = summary.loc[:, ~summary.columns.astype(str).str.startswith("index")]
    return summary.sort_values(["metric", "model_label"]).reset_index(drop=True)


def _build_metric_win_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["metric", "model_label", "win_count", "contests", "win_rate"])
    records: list[dict[str, object]] = []
    for contest_key, contest_frame in rows.groupby(list(COMPARISON_KEY_COLUMNS), sort=True):
        contest_identity = dict(zip(COMPARISON_KEY_COLUMNS, contest_key, strict=False))
        for metric in METRIC_COLUMN_NAMES:
            best_score = contest_frame[metric].max()
            winners = contest_frame[contest_frame[metric] == best_score]["model_label"].tolist()
            if not winners:
                continue
            credit = 1.0 / len(winners)
            for model_label in DISPLAY_MODEL_ORDER:
                records.append(
                    {
                        **contest_identity,
                        "metric": metric,
                        "model_label": model_label,
                        "win_count": credit if model_label in winners else 0.0,
                        "contests": 1,
                    }
                )
    summary = (
        pd.DataFrame.from_records(records)
        .groupby(["metric", "model_label"], as_index=False)[["win_count", "contests"]]
        .sum()
    )
    summary["win_rate"] = summary["win_count"] / summary["contests"]
    return summary.sort_values(["metric", "model_label"]).reset_index(drop=True)


def _winner_label(row: pd.Series) -> str:
    winner_labels = [str(column).rsplit("_", 2)[0] for column, value in row.items() if pd.notna(value)]
    if not winner_labels:
        return ""
    best_value = row.max()
    winners = [
        str(column).rsplit("_", 2)[0]
        for column, value in row.items()
        if pd.notna(value) and value == best_value
    ]
    return ",".join(winners)


def _render_markdown_report(
    *,
    result: LlmSameSettingsStudyResult,
    metric_summary: pd.DataFrame,
    metric_win_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Optimization Same-Settings LLM Comparison",
        "",
        "This side study reuses existing quantitative artifacts only.",
        (
            "Gemini defines the allowed settings slice; all comparison models are filtered "
            "to that same slice before comparison."
        ),
        "",
        "## Inputs",
        "",
        *[f"- `{model_label}`: `{path}`" for model_label, path in sorted(result.source_run_roots.items())],
        "",
        "## Fixed settings",
        "",
        "- `evidence=plot`",
        "- `role=False`",
        "- `insights=False`",
        "- `example=True`",
        "- `summarizer in {bart, bert, t5}`",
        "",
        f"Comparable cases present in all models: `{result.comparable_case_count}`",
        "",
        "## Metric summary",
        "",
        _render_markdown_table(metric_summary),
        "",
        "## Metric win summary",
        "",
        _render_markdown_table(metric_win_summary),
        "",
        "## Artifacts",
        "",
        f"- `same_settings_rows.csv`: `{result.same_settings_long_path}`",
        f"- `master_comparison.csv`: `{result.master_comparison_path}`",
        f"- `metric_summary.csv`: `{result.metric_summary_path}`",
        f"- `metric_win_summary.csv`: `{result.metric_win_summary_path}`",
    ]
    return "\n".join(lines)


def _render_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.2f}")
    header = "| " + " | ".join(str(column) for column in display.columns) + " |"
    divider = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False)]
    return "\n".join([header, divider, *rows])
