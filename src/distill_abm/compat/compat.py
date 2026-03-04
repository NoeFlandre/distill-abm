"""Compatibility wrappers for historical callable surface."""

from __future__ import annotations

from pathlib import Path

from distill_abm.compat.compat_callables import (
    analyze_image_with_janus as _analyze_image_with_janus,
)
from distill_abm.compat.compat_callables import (
    encode_image,
    prepare_conversation,
    setup_janus_model,
)
from distill_abm.compat.compat_callables import (
    extract_text_from_pdf as _extract_text_from_pdf,
)
from distill_abm.compat.compat_callables import (
    get_llm_response as _get_llm_response,
)
from distill_abm.compat.compat_callables import (
    get_llm_response2 as _get_llm_response2,
)
from distill_abm.compat.compat_callables import (
    get_llm_response_no_image as _get_llm_response_no_image,
)
from distill_abm.compat.compat_factorial import (
    add_interactions,
    analyze_factorial_contributions,
    calculate_sst,
    calculate_sums_and_sst,
    compute_results,
    create_factorial_design,
    get_factor_columns,
    preprocess_for_factorial,
    read_and_parse_csv,
    read_csv_to_df,
    remove_evaluating_suffix,
    return_csv,
    return_csv_2,
)
from distill_abm.compat.compat_io import (
    append_analysis_to_csv,
    append_to_csv,
    append_to_csv2,
    create_collage,
    load_example_files,
    load_existing_rows,
    process_csv,
    process_documentation,
    remove_non_unicode,
)
from distill_abm.compat.compat_plot import plot_columns
from distill_abm.compat.compat_scoring import (
    append_coverage_score,
    append_faithfulness_score,
    compute_scores_wrapper,
    fill_faithfulness_scores,
    increment_score,
    modify_scores,
    update_structured_df,
)
from distill_abm.compat.compat_text import capitalize, clean_context_response, clean_symbols, should_skip_row
from distill_abm.eval.doe import clean_name, clean_statsmodels_feature_name, identify_factors_and_metrics
from distill_abm.eval.doe_full import analyze_factorial_anova
from distill_abm.eval.qualitative import extract_coverage_score, extract_faithfulness_score
from distill_abm.eval.reference_scores import compute_scores
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
from distill_abm.pipeline.run import run_pipeline
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
from distill_abm.summarize.reference_text import chunk_text, summarize_text


def get_llm_response(prompt: str, base64_image: str | None = None) -> str:
    return _get_llm_response(prompt, base64_image)


def get_llm_response2(prompt: str) -> str:
    return _get_llm_response2(prompt)


def get_llm_response_no_image(prompt: str) -> str:
    return _get_llm_response_no_image(prompt)


def analyze_image_with_janus(prompt: str, base64_image: str) -> str:
    return _analyze_image_with_janus(prompt, base64_image)


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    return _extract_text_from_pdf(pdf_path)


def get_response_with_images(question: str, image_paths: list[str | Path]) -> str:
    encoded = [encode_image(path) for path in image_paths]
    images = [item for item in encoded if item]
    return get_llm_response(question, images[0] if images else None)


def get_response_with_pdf_and_images(question: str, pdf_path: str | Path, image_paths: list[str | Path]) -> str:
    text = extract_text_from_pdf(pdf_path)
    return get_response_with_images(f"{question}\n{text}", image_paths)


def generate_response(prompt: str) -> str:
    return get_llm_response(prompt)


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
    "summarize_with_bart",
    "summarize_with_bert",
    "update_parameters",
    "update_structured_df",
]
