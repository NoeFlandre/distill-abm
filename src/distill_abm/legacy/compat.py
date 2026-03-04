"""Compatibility wrappers that preserve notebook-era function names."""

from __future__ import annotations

import base64
import csv
import json
import re
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from typing import Any, TypeVar, cast

import pandas as pd

import distill_abm.legacy.notebook_loader as notebook_loader
from distill_abm.eval.doe import clean_name, clean_statsmodels_feature_name, identify_factors_and_metrics
from distill_abm.eval.doe_full import analyze_factorial_anova
from distill_abm.eval.legacy_scores import compute_scores
from distill_abm.eval.qualitative import extract_coverage_score, extract_faithfulness_score
from distill_abm.ingest.netlogo import (
    clean_json_content,
    extract_code,
    extract_documentation,
    extract_parameters,
    format_json_oneline,
    remove_default_elements,
    remove_urls,
    remove_urls_from_data,
    update_parameters,
)
from distill_abm.llm.adapters.base import LLMMessage, LLMRequest
from distill_abm.llm.adapters.echo_adapter import EchoAdapter
from distill_abm.pipeline.run import run_pipeline
from distill_abm.summarize.legacy import (
    chunk_text,
    summarize_text,
)
from distill_abm.summarize.legacy import (
    clean_context_response as _clean_context_response_refactored,
)
from distill_abm.summarize.legacy import (
    clean_symbols as _clean_symbols_refactored,
)
from distill_abm.summarize.legacy import (
    process_csv_context as _process_csv_context_refactored,
)
from distill_abm.summarize.models import summarize_with_bart, summarize_with_bert
from distill_abm.summarize.postprocess import (
    capitalize_sentences,
    clean_non_unicode,
    remove_hyphens_after_punctuation,
    remove_sentences_with_www,
    remove_space_before_dot,
    remove_unnecessary_punctuation,
    remove_unnecessary_spaces_in_parentheses,
)
from distill_abm.viz.plots import plot_metric_bundle

T = TypeVar("T")

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


def _call_notebook_first(name: str, fallback: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    if not notebook_loader.should_dispatch_notebook(name):
        return fallback(*args, **kwargs)
    try:
        notebook_callable = notebook_loader.get_notebook_function(name)
    except KeyError:
        return fallback(*args, **kwargs)
    try:
        return cast(T, notebook_callable(*args, **kwargs))
    except Exception:
        return fallback(*args, **kwargs)


def encode_image(image_path: str | Path) -> str | None:
    try:
        return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    except Exception:
        return None


def get_llm_response(prompt: str, base64_image: str | None = None) -> str:
    adapter = EchoAdapter(model="legacy-echo")
    req = LLMRequest(model="legacy-echo", messages=[LLMMessage(role="user", content=prompt)], image_b64=base64_image)
    return adapter.complete(req).text


def get_llm_response2(prompt: str) -> str:
    return get_llm_response(prompt)


def get_llm_response_no_image(prompt: str) -> str:
    return get_llm_response(prompt)


def analyze_image_with_janus(prompt: str, base64_image: str) -> str:
    return get_llm_response(prompt, base64_image)


def setup_janus_model() -> str:
    """Compatibility stub for notebook Janus model initialization."""
    return "janus-model-initialized"


def append_to_csv(
    file_name: str | Path,
    combination_desc: str,
    context_prompt: str,
    context_response: str,
    trend_analysis_responses: list[str],
) -> None:
    _append_row(file_name, [combination_desc, context_prompt, context_response, *trend_analysis_responses])


def append_to_csv2(
    file_name: str | Path,
    combination_desc: str,
    context_prompt: str,
    context_response: str,
    trend_analysis_prompts: list[str],
    trend_analysis_responses: list[str],
) -> None:
    row: list[str] = [combination_desc, context_prompt, context_response]
    for prompt, response in zip(trend_analysis_prompts, trend_analysis_responses, strict=False):
        row.extend([prompt, response])
    _append_row(file_name, row)


def append_analysis_to_csv(file_name: str | Path, values: list[str]) -> None:
    _append_row(file_name, values)


def _append_row(file_name: str | Path, row: list[str]) -> None:
    with Path(file_name).open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(row)


def load_existing_rows(file_name: str | Path) -> list[list[str]]:
    path = Path(file_name)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [row for row in csv.reader(handle)]


def process_documentation(input_json: str | Path, output_json: str | Path) -> None:
    payload = json.loads(Path(input_json).read_text(encoding="utf-8"))
    cleaned = remove_urls_from_data(payload)
    if "documentation" in cleaned:
        cleaned["documentation"] = remove_default_elements(str(cleaned["documentation"]))
    Path(output_json).write_text(json.dumps(cleaned, indent=2), encoding="utf-8")


def clean_context_response(text: str) -> str:
    return _call_notebook_first("clean_context_response", _clean_context_response_refactored, text)


def clean_symbols(text: str | float) -> str | float:
    return _call_notebook_first("clean_symbols", _clean_symbols_refactored, text)


def process_csv(input_file: str | Path, output_file: str | Path) -> None:
    _process_csv_context_refactored(Path(input_file), Path(output_file))


def remove_non_unicode(text: str) -> str:
    temp = Path("/tmp/.legacy_non_unicode_in.csv")
    out = Path("/tmp/.legacy_non_unicode_out.csv")
    temp.write_text(text, encoding="utf-8")
    clean_non_unicode(temp, out)
    return out.read_text(encoding="utf-8")


def capitalize(match: re.Match[str]) -> str:
    return match.group(1) + match.group(2).upper()


def plot_columns(
    data: pd.DataFrame,
    column_pattern: str,
    y_label: str,
    title: str,
    exclude_pattern: str | None = None,
) -> Path:
    return plot_metric_bundle(
        frame=data,
        include_pattern=column_pattern,
        exclude_pattern=exclude_pattern,
        output_dir=Path("results/plots"),
        title=title,
        y_label=y_label,
        show_mean_line=False,
    )


def should_skip_row(row: dict[str, Any], column_name: str) -> bool:
    value = row.get(column_name)
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value > 0
    return isinstance(value, str) and bool(value.strip())


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


def load_example_files(base_dir: str | Path = "Examples/Text") -> dict[str, str]:
    base = Path(base_dir)
    output: dict[str, str] = {}
    for path in sorted(base.glob("*.txt")):
        output[path.stem] = path.read_text(encoding="utf-8")
    return output


def get_response_with_images(question: str, image_paths: list[str | Path]) -> str:
    encoded = [encode_image(path) for path in image_paths]
    images = [item for item in encoded if item]
    return get_llm_response(question, images[0] if images else None)


def get_response_with_pdf_and_images(question: str, pdf_path: str | Path, image_paths: list[str | Path]) -> str:
    text = extract_text_from_pdf(pdf_path)
    return get_response_with_images(f"{question}\n{text}", image_paths)


def prepare_conversation(history: list[dict[str, str]]) -> list[dict[str, str]]:
    return [dict(item) for item in history]


def generate_response(prompt: str) -> str:
    return get_llm_response(prompt)


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    try:
        import pypdf
    except Exception:
        return ""
    reader = pypdf.PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def create_collage(image_paths: list[str | Path], output_path: str | Path, columns: int = 2) -> Path:
    try:
        from PIL import Image
    except Exception:
        Path(output_path).write_text("PIL not available", encoding="utf-8")
        return Path(output_path)
    images = [Image.open(path) for path in image_paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * height), color=(255, 255, 255))
    for index, image in enumerate(images):
        x = (index % columns) * width
        y = (index // columns) * height
        canvas.paste(image, (x, y))
    canvas.save(output_path)
    return Path(output_path)


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


def read_csv_to_df(csv_path: str | Path) -> pd.DataFrame:
    def _fallback(path: str | Path) -> pd.DataFrame:
        return pd.read_csv(path)

    return _call_notebook_first("read_csv_to_df", _fallback, csv_path)


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


def read_and_parse_csv(
    csv_path: str | Path,
    repetitions: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _call_notebook_first("read_and_parse_csv", _read_and_parse_csv_fallback, csv_path, repetitions)


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


def calculate_sst(csv_path: str | Path, repetitions: int) -> tuple[list[tuple[list[tuple[str, float]], float]], int]:
    return _call_notebook_first("calculate_sst", _calculate_sst_fallback, csv_path, repetitions)


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


def compute_results(csv_path: str | Path, repetitions: int) -> list[list[tuple[str, float]]]:
    return _call_notebook_first("compute_results", _compute_results_fallback, csv_path, repetitions)


def remove_evaluating_suffix(text: str) -> str:
    def _fallback(value: str) -> str:
        marker = "_evaluating"
        index = value.find(marker)
        return value[:index] if index >= 0 else value

    return _call_notebook_first("remove_evaluating_suffix", _fallback, text)


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


def return_csv(csv_path: str | Path, output_csv: str | Path, repetitions: int) -> Path:
    return _call_notebook_first("return_csv", _return_csv_fallback, csv_path, output_csv, repetitions)


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


def return_csv_2(csv_path: str | Path, output_csv: str | Path, repetitions: int) -> Path:
    return _call_notebook_first("return_csv_2", _return_csv_2_fallback, csv_path, output_csv, repetitions)


def compute_scores_wrapper(ground_truth: str, summary: str) -> tuple[float, float, float, float, float, float]:
    score = compute_scores(ground_truth, summary)
    return score.bleu, score.meteor, score.rouge1, score.rouge2, score.rouge_l, score.flesch_reading_ease


def summarize_text_with_models(text: str) -> dict[str, str]:
    return {"bart": summarize_with_bart(text), "bert": summarize_with_bert(text)}


__all__ = [
    "add_interactions",
    "analyze_factorial_anova",
    "analyze_factorial_contributions",
    "analyze_image_with_janus",
    "append_analysis_to_csv",
    "append_coverage_score",
    "append_faithfulness_score",
    "append_to_csv",
    "append_to_csv2",
    "calculate_sst",
    "calculate_sums_and_sst",
    "capitalize",
    "capitalize_sentences",
    "chunk_text",
    "clean_context_response",
    "clean_json_content",
    "clean_name",
    "clean_non_unicode",
    "clean_statsmodels_feature_name",
    "clean_symbols",
    "compute_results",
    "compute_scores",
    "compute_scores_wrapper",
    "create_collage",
    "create_factorial_design",
    "encode_image",
    "extract_code",
    "extract_coverage_score",
    "extract_documentation",
    "extract_faithfulness_score",
    "extract_parameters",
    "extract_text_from_pdf",
    "fill_faithfulness_scores",
    "format_json_oneline",
    "generate_response",
    "get_factor_columns",
    "get_llm_response",
    "get_llm_response2",
    "get_llm_response_no_image",
    "get_response_with_images",
    "get_response_with_pdf_and_images",
    "identify_factors_and_metrics",
    "increment_score",
    "load_example_files",
    "load_existing_rows",
    "modify_scores",
    "plot_columns",
    "prepare_conversation",
    "preprocess_for_factorial",
    "process_csv",
    "process_documentation",
    "read_and_parse_csv",
    "read_csv_to_df",
    "remove_default_elements",
    "remove_evaluating_suffix",
    "remove_hyphens_after_punctuation",
    "remove_non_unicode",
    "remove_sentences_with_www",
    "remove_space_before_dot",
    "remove_unnecessary_punctuation",
    "remove_unnecessary_spaces_in_parentheses",
    "remove_urls",
    "remove_urls_from_data",
    "return_csv",
    "return_csv_2",
    "run_pipeline",
    "setup_janus_model",
    "should_skip_row",
    "summarize_text",
    "summarize_text_with_models",
    "update_parameters",
    "update_structured_df",
]
