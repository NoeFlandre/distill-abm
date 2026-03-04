"""Compatibility scoring and LLM CSV post-processing helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import pandas as pd

from distill_abm.eval.legacy_scores import compute_scores
from distill_abm.eval.qualitative import extract_coverage_score, extract_faithfulness_score
from distill_abm.summarize.models import summarize_with_bart, summarize_with_bert

_SUMMARY_METRICS_MAPPING: dict[str, dict[str, str]] = {
    "Summary (BART) Reduced": {
        "BLEU (BART)": "BLEU",
        "METEOR (BART)": "METEOR",
        "ROUGE-1 (BART)": "ROUGE-1",
        "ROUGE-2 (BART)": "ROUGE-2",
        "ROUGE-L (BART)": "ROUGE-L",
        "Flesch Reading Ease (BART)": "Flesch Reading Ease",
    },
    "Summary (BERT) Reduced": {
        "BLEU (BERT)": "BLEU",
        "METEOR (BERT)": "METEOR",
        "ROUGE-1 (BERT)": "ROUGE-1",
        "ROUGE-2 (BERT)": "ROUGE-2",
        "ROUGE-L (BERT)": "ROUGE-L",
        "Flesch Reading Ease (BERT)": "Flesch Reading Ease",
    },
}


T = TypeVar("T")


def append_faithfulness_score(response_or_input_csv: str | Path, output_csv: str | Path | None = None) -> str | Path:
    if output_csv is None:
        return extract_faithfulness_score(str(response_or_input_csv))
    frame = pd.read_csv(response_or_input_csv)
    frame["LLM Faithfulness Score"] = frame["LLM Faithfulness Raw Response"].apply(extract_faithfulness_score)
    output = Path(output_csv)
    frame.to_csv(output, index=False)
    return output


def append_coverage_score(response_or_input_csv: str | Path, output_csv: str | Path | None = None) -> str | Path:
    if output_csv is None:
        return extract_coverage_score(str(response_or_input_csv))
    frame = pd.read_csv(response_or_input_csv, sep=";")
    frame["LLM Coverage Score"] = frame["LLM Coverage Raw Response"].apply(extract_coverage_score)
    output = Path(output_csv)
    frame.to_csv(output, index=False)
    return output


def increment_score(score: int | float | str) -> int:
    try:
        return int(float(score)) + 1
    except Exception:
        return 1


def modify_scores(values: list[int | float | str]) -> list[int]:
    return [increment_score(value) for value in values]


def update_structured_df(
    input_df: pd.DataFrame,
    structured_df: pd.DataFrame,
    case_study: str,
    llm: str,
) -> pd.DataFrame:
    return _call_notebook_first(
        "update_structured_df",
        _update_structured_df_fallback,
        input_df,
        structured_df,
        case_study,
        llm,
    )


def _update_structured_df_fallback(
    input_df: pd.DataFrame,
    structured_df: pd.DataFrame,
    case_study: str,
    llm: str,
) -> pd.DataFrame:
    output = structured_df.copy()
    for summary_type, metric_mapping in _SUMMARY_METRICS_MAPPING.items():
        if summary_type not in input_df.columns:
            continue
        summary_model = summary_type.split(" ")[1].strip("()")
        for _, row in input_df.iterrows():
            combination_desc = str(row.get("Combination Description", ""))
            lowered = combination_desc.lower()
            role = "Yes" if "role" in lowered else "No"
            example = "Yes" if "example" in lowered else "No"
            insight = "Yes" if "insights" in lowered else "No"
            filter_condition = (
                (output["Case study"] == case_study)
                & (output["Summary"] == summary_model)
                & (output["LLM"] == llm)
                & (output["Role"] == role)
                & (output["Example"] == example)
                & (output["Insight"] == insight)
            )
            output.loc[filter_condition, "Output"] = row.get(summary_type, "")
            if "Context Prompt" in row:
                output.loc[filter_condition, "Input"] = row.get("Context Prompt", "")
            for metric_column, target_column in metric_mapping.items():
                if metric_column in row and target_column in output.columns:
                    output.loc[filter_condition, target_column] = row[metric_column]
    return output


def fill_faithfulness_scores(
    frame: pd.DataFrame | None = None,
    source_column: str = "Faithfulness (GPT)",
    structured_data_path: str | Path = "Final Sheet/updated_structured_data.csv",
    yes_no_path: str | Path = "Results/Yes-No Format.csv",
    output_path: str | Path = "Final Sheet/updated_structured_data_filled.csv",
) -> pd.DataFrame:
    if frame is not None:
        return _fill_faithfulness_scores_frame_fallback(frame, source_column=source_column)
    return _fill_faithfulness_scores_files_fallback(
        structured_data_path=structured_data_path,
        yes_no_path=yes_no_path,
        output_path=output_path,
    )


def _fill_faithfulness_scores_frame_fallback(
    frame: pd.DataFrame,
    source_column: str = "Faithfulness (GPT)",
) -> pd.DataFrame:
    out = frame.copy()
    if source_column in out.columns:
        out[source_column] = out[source_column].fillna("")
    return out


def _fill_faithfulness_scores_files_fallback(
    structured_data_path: str | Path,
    yes_no_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    structured = pd.read_csv(structured_data_path, sep=",")
    yes_no = pd.read_csv(yes_no_path, sep=";")
    required_columns = [
        "Case study",
        "Summary",
        "LLM",
        "Role",
        "Example",
        "Insight",
        "Final LLM Faithfulness Score",
    ]
    missing = [column for column in required_columns if column not in yes_no.columns]
    if missing:
        raise ValueError(f"Columns {missing} not found in Yes-No CSV.")
    merged = pd.merge(
        structured,
        yes_no[required_columns],
        on=required_columns[:-1],
        how="left",
    )
    merged["Faithfulness (GPT)"] = merged["Final LLM Faithfulness Score"]
    merged.drop(columns=["Final LLM Faithfulness Score"], inplace=True)
    merged.to_csv(output_path, index=False, sep=",")
    return merged


def compute_scores_wrapper(ground_truth: str, summary: str) -> tuple[float, float, float, float, float, float]:
    score = compute_scores(ground_truth, summary)
    return score.bleu, score.meteor, score.rouge1, score.rouge2, score.rouge_l, score.flesch_reading_ease


def summarize_text_with_models(text: str) -> dict[str, str]:
    return {"bart": summarize_with_bart(text), "bert": summarize_with_bert(text)}


def _call_notebook_first(name: str, fallback: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    from distill_abm.legacy.compat_callables import _call_notebook_first as _dispatch

    return cast(T, _dispatch(name, fallback, *args, **kwargs))
