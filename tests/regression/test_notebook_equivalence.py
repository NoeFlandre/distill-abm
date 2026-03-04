from __future__ import annotations

import ast
from pathlib import Path
from types import FunctionType
from typing import Any, cast

import nbformat
import pandas as pd
import pytest

import distill_abm.legacy.compat as compat
from distill_abm.eval.doe import clean_name, clean_statsmodels_feature_name, identify_factors_and_metrics
from distill_abm.eval.qualitative import (
    extract_coverage_score,
    extract_faithfulness_score,
    should_skip_row,
)
from distill_abm.ingest.netlogo import remove_default_elements, remove_urls
from distill_abm.legacy.notebook_loader import get_notebook_function
from distill_abm.summarize.legacy import clean_context_response, clean_symbols
from distill_abm.summarize.postprocess import (
    capitalize_sentences,
    remove_hyphens_after_punctuation,
    remove_space_before_dot,
    remove_unnecessary_punctuation,
    remove_unnecessary_spaces_in_parentheses,
)

RE_CONTEXT = {"re": __import__("re")}


def _load_notebook_function(path: Path, func_name: str, context: dict[str, Any]) -> FunctionType:
    notebook = cast(Any, nbformat).read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                namespace: dict[str, Any] = {}
                namespace.update(context)
                segment = ast.get_source_segment(source, node)
                if segment is None:
                    continue
                exec(segment, namespace)
                return cast(FunctionType, namespace[func_name])
    raise ValueError(f"function {func_name} not found in {path}")


def test_notebook_parity_for_core_text_utilities() -> None:
    root = Path("archive/legacy_repo/Code")
    model_nb = root / "Models/Milk Consumption/1. (Cleaned) From NetLogo to CSV.ipynb"
    summary_nb = root / "Models/Milk Consumption/4. Generating the summary.ipynb"
    post_nb = root / "Models/Milk Consumption/5. Postprocessing.ipynb"

    defaults = {"## WHAT IS IT?": "(a general understanding of what the model is trying to show or explain)"}
    notebook_remove_urls = _load_notebook_function(model_nb, "remove_urls", RE_CONTEXT)
    notebook_remove_defaults = _load_notebook_function(
        model_nb,
        "remove_default_elements",
        {"default_elements": defaults},
    )
    notebook_clean_context = _load_notebook_function(summary_nb, "clean_context_response", {})
    notebook_clean_symbols = _load_notebook_function(summary_nb, "clean_symbols", {"pd": pd})
    notebook_hyphens = _load_notebook_function(post_nb, "remove_hyphens_after_punctuation", RE_CONTEXT)
    notebook_punct = _load_notebook_function(post_nb, "remove_unnecessary_punctuation", RE_CONTEXT)
    notebook_parens = _load_notebook_function(
        post_nb,
        "remove_unnecessary_spaces_in_parentheses",
        RE_CONTEXT,
    )
    notebook_dot = _load_notebook_function(post_nb, "remove_space_before_dot", RE_CONTEXT)
    notebook_cap = _load_notebook_function(post_nb, "capitalize_sentences", RE_CONTEXT)

    assert remove_urls("a www.site.com") == notebook_remove_urls("a www.site.com")
    doc = "## WHAT IS IT?\n\n(a general understanding of what the model is trying to show or explain)"
    assert remove_default_elements(doc, defaults) == notebook_remove_defaults(doc)
    assert clean_context_response("x</think>y") == notebook_clean_context("x</think>y")
    assert clean_symbols("#x*") == notebook_clean_symbols("#x*")
    assert remove_hyphens_after_punctuation("A. - b") == notebook_hyphens("A. - b")
    assert remove_unnecessary_punctuation("A. , b") == notebook_punct("A. , b")
    assert remove_unnecessary_spaces_in_parentheses("( a )") == notebook_parens("( a )")
    assert remove_space_before_dot("a .") == notebook_dot("a .")
    assert capitalize_sentences("hello. world") == notebook_cap("hello. world")


def test_notebook_parity_for_evaluation_helpers() -> None:
    root = Path("archive/legacy_repo/Code")
    gpt_nb = root / "Evaluation/Qualitative Assessment using LLMs/1. GPTFaithfulness.ipynb"
    claude_nb = root / "Evaluation/Qualitative Assessment using LLMs/2.ClaudeCoverage.ipynb"
    doe_nb = root / "Evaluation/DOE/DoE.ipynb"

    notebook_faith = _load_notebook_function(gpt_nb, "extract_faithfulness_score", {})
    notebook_skip = _load_notebook_function(gpt_nb, "should_skip_row", {"pd": pd})
    notebook_cov = _load_notebook_function(claude_nb, "extract_coverage_score", {})
    notebook_clean_name = _load_notebook_function(doe_nb, "clean_name", RE_CONTEXT)
    notebook_clean_feature = _load_notebook_function(doe_nb, "clean_statsmodels_feature_name", {})
    notebook_split = _load_notebook_function(doe_nb, "identify_factors_and_metrics", {"pd": pd})

    frame = pd.DataFrame({"Model": ["BART", "BERT"], "Metric": [0.1, 0.2], "Flag": [1, -1]})
    notebook_factors, notebook_metrics = notebook_split(frame)
    factors, metrics = identify_factors_and_metrics(frame)

    assert extract_faithfulness_score("Faithfulness score: 5") == notebook_faith("Faithfulness score: 5")
    assert extract_coverage_score("Coverage score: 4") == notebook_cov("Coverage score: 4")
    assert should_skip_row({"score": 1}, "score") == notebook_skip({"score": 1}, "score")
    assert clean_name("A Column(1)") == notebook_clean_name("A Column(1)")
    assert clean_statsmodels_feature_name("C(A):C(B)") == notebook_clean_feature("C(A):C(B)")
    assert set(factors) == set(notebook_factors)
    assert set(metrics) == set(notebook_metrics)


def test_notebook_parity_for_remaining_deterministic_doe_helpers(tmp_path: Path) -> None:
    csv_path = tmp_path / "doe.csv"
    frame = pd.DataFrame(
        {
            "PromptA": [-1, 1, -1, 1],
            "PromptB": [-1, -1, 1, 1],
            "BLEU": [0.1, 0.2, 0.3, 0.4],
            "ROUGE": [0.2, 0.3, 0.4, 0.5],
            "IgnoreMe": [2.0, 2.0, 2.0, 2.0],
        }
    )
    frame.to_csv(csv_path, index=False)

    notebook_read = get_notebook_function("read_and_parse_csv")
    notebook_design = get_notebook_function("create_factorial_design")
    notebook_compute = get_notebook_function("compute_results")

    compat_avg, compat_factors, compat_metrics = compat.read_and_parse_csv(csv_path, 2)
    notebook_avg, notebook_factors, notebook_metrics = notebook_read(csv_path, 2)
    assert compat_avg.equals(notebook_avg)
    assert compat_factors.equals(notebook_factors)
    assert compat_metrics.equals(notebook_metrics)

    compat_design_bundle = compat.create_factorial_design(csv_path, 2)
    assert isinstance(compat_design_bundle, tuple)
    compat_design_avg, compat_design_factors, compat_design_metrics = compat_design_bundle
    notebook_design_avg, notebook_design_factors, notebook_design_metrics = notebook_design(csv_path, 2)
    assert compat_design_avg.equals(notebook_design_avg)
    assert compat_design_factors.equals(notebook_design_factors)
    assert compat_design_metrics.equals(notebook_design_metrics)

    compat_results = compat.compute_results(csv_path, 2)
    notebook_results = notebook_compute(csv_path, 2)
    assert len(compat_results) == len(notebook_results)
    for compat_metric, notebook_metric in zip(compat_results, notebook_results, strict=True):
        assert [name for name, _ in compat_metric] == [name for name, _ in notebook_metric]
        for (_compat_name, compat_value), (_notebook_name, notebook_value) in zip(
            compat_metric, notebook_metric, strict=True
        ):
            assert compat_value == pytest.approx(notebook_value)


def test_notebook_parity_for_remove_evaluating_suffix() -> None:
    notebook_remove_suffix = get_notebook_function("remove_evaluating_suffix")
    value = "Role_AND_Example_evaluating_BLEU"
    assert compat.remove_evaluating_suffix(value) == notebook_remove_suffix(value)


def test_notebook_parity_for_read_csv_to_df(tmp_path: Path) -> None:
    csv_path = tmp_path / "simple.csv"
    frame = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    frame.to_csv(csv_path, index=False)
    notebook_read = get_notebook_function("read_csv_to_df")
    assert compat.read_csv_to_df(csv_path).equals(notebook_read(csv_path))


def test_notebook_parity_for_calculate_sums_and_sst_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        {
            "Intercept_evaluating_BLEU": [1.0, 1.0, 1.0, 1.0],
            "PromptA_evaluating_BLEU": [-1.0, 1.0, -1.0, 1.0],
            "PromptB_evaluating_BLEU": [-1.0, -1.0, 1.0, 1.0],
        }
    )
    notebook_fn = get_notebook_function("calculate_sums_and_sst")
    notebook_sums, notebook_sst = notebook_fn(frame, "BLEU", 2)

    def raise_missing(_name: str) -> FunctionType:
        raise KeyError("missing")

    monkeypatch.setattr("distill_abm.legacy.notebook_loader.get_notebook_function", raise_missing)
    fallback_sums, fallback_sst = compat.calculate_sums_and_sst(frame, "BLEU", 2)

    assert [name for name, _ in fallback_sums] == [name for name, _ in notebook_sums]
    for (_name_a, value_a), (_name_b, value_b) in zip(fallback_sums, notebook_sums, strict=True):
        assert value_a == pytest.approx(value_b)
    assert fallback_sst == pytest.approx(notebook_sst)


def test_notebook_parity_for_calculate_sst_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "sst.csv"
    pd.DataFrame(
        {
            "PromptA": [-1, 1, -1, 1],
            "PromptB": [-1, -1, 1, 1],
            "BLEU": [0.1, 0.2, 0.3, 0.4],
        }
    ).to_csv(csv_path, index=False)
    notebook_fn = get_notebook_function("calculate_sst")
    notebook_list, notebook_experiments = notebook_fn(csv_path, 2)

    def raise_missing(_name: str) -> FunctionType:
        raise KeyError("missing")

    monkeypatch.setattr("distill_abm.legacy.notebook_loader.get_notebook_function", raise_missing)
    fallback_list, fallback_experiments = compat.calculate_sst(csv_path, 2)

    assert fallback_experiments == notebook_experiments
    assert len(fallback_list) == len(notebook_list)
    for (fallback_sums, fallback_sst), (notebook_sums, notebook_sst) in zip(fallback_list, notebook_list, strict=True):
        assert [name for name, _ in fallback_sums] == [name for name, _ in notebook_sums]
        for (_name_a, value_a), (_name_b, value_b) in zip(fallback_sums, notebook_sums, strict=True):
            assert value_a == pytest.approx(value_b)
        assert fallback_sst == pytest.approx(notebook_sst)


def test_external_wrapper_paths_are_mockable(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_llm(prompt: str, base64_image: str | None = None) -> str:
        calls.append((prompt, base64_image))
        return "ok"

    monkeypatch.setattr(compat, "get_llm_response", fake_llm)
    monkeypatch.setattr(compat, "encode_image", lambda _path: "img-b64")
    monkeypatch.setattr(compat, "extract_text_from_pdf", lambda _path: "pdf-body")
    monkeypatch.setattr(compat, "summarize_with_bart", lambda text: f"bart::{text}")
    monkeypatch.setattr(compat, "summarize_with_bert", lambda text: f"bert::{text}")

    assert compat.get_response_with_images("question", ["x.png"]) == "ok"
    assert calls[-1] == ("question", "img-b64")
    assert compat.get_response_with_pdf_and_images("question", "paper.pdf", ["x.png"]) == "ok"
    assert calls[-1] == ("question\npdf-body", "img-b64")
    assert compat.summarize_text_with_models("hello") == {"bart": "bart::hello", "bert": "bert::hello"}


def test_notebook_parity_for_qualitative_csv_appenders(tmp_path: Path) -> None:
    notebook_append_faith = get_notebook_function("append_faithfulness_score")
    notebook_append_cov = get_notebook_function("append_coverage_score")

    faith_input = tmp_path / "faith_input.csv"
    faith_notebook_out = tmp_path / "faith_notebook.csv"
    faith_compat_out = tmp_path / "faith_compat.csv"
    pd.DataFrame({"LLM Faithfulness Raw Response": ["Faithfulness score: 5", "Faithfulness score: 3"]}).to_csv(
        faith_input, index=False
    )

    notebook_append_faith(faith_input, faith_notebook_out)
    compat.append_faithfulness_score(faith_input, faith_compat_out)

    notebook_faith_df = pd.read_csv(faith_notebook_out)
    compat_faith_df = pd.read_csv(faith_compat_out)
    assert compat_faith_df.equals(notebook_faith_df)

    cov_input = tmp_path / "cov_input.csv"
    cov_notebook_out = tmp_path / "cov_notebook.csv"
    cov_compat_out = tmp_path / "cov_compat.csv"
    pd.DataFrame({"LLM Coverage Raw Response": ["Coverage score: 4", "Coverage score: 2"]}).to_csv(
        cov_input, index=False, sep=";"
    )

    notebook_append_cov(cov_input, cov_notebook_out)
    compat.append_coverage_score(cov_input, cov_compat_out)

    notebook_cov_df = pd.read_csv(cov_notebook_out)
    compat_cov_df = pd.read_csv(cov_compat_out)
    assert compat_cov_df.equals(notebook_cov_df)


def test_notebook_parity_for_analyze_factorial_contributions_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "anova.csv"
    pd.DataFrame(
        {
            "Role": ["Yes", "Yes", "No", "No"],
            "Example": ["Yes", "No", "Yes", "No"],
            "BLEU": [0.1, 0.2, 0.3, 0.4],
            "ROUGE-1": [0.2, 0.4, 0.6, 0.8],
        }
    ).to_csv(csv_path, index=False)
    notebook_out = tmp_path / "nb_anova.csv"
    fallback_out = tmp_path / "fallback_anova.csv"

    notebook_analyze = get_notebook_function("analyze_factorial_contributions")
    notebook_df = notebook_analyze(csv_path, notebook_out, 2, 2)

    def raise_missing(_name: str) -> FunctionType:
        raise KeyError("missing")

    monkeypatch.setattr("distill_abm.legacy.notebook_loader.get_notebook_function", raise_missing)
    fallback_df = compat.analyze_factorial_contributions(csv_path, fallback_out, repetitions=2, max_interaction_order=2)
    assert fallback_df is not None
    assert notebook_df is not None

    notebook_sorted = notebook_df.sort_values("Feature").reset_index(drop=True)
    fallback_sorted = fallback_df.sort_values("Feature").reset_index(drop=True)
    assert list(fallback_sorted.columns) == list(notebook_sorted.columns)
    for column in fallback_sorted.columns:
        if column == "Feature":
            assert fallback_sorted[column].tolist() == notebook_sorted[column].tolist()
        else:
            assert fallback_sorted[column].tolist() == pytest.approx(notebook_sorted[column].tolist())


def test_notebook_parity_for_sheet_helpers_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_df = pd.DataFrame(
        {
            "Combination Description": ["role + example + insights"],
            "Summary (BART) Reduced": ["bart output"],
            "Summary (BERT) Reduced": ["bert output"],
            "Context Prompt": ["context prompt"],
            "BLEU (BART)": [0.1],
            "METEOR (BART)": [0.2],
            "ROUGE-1 (BART)": [0.3],
            "ROUGE-2 (BART)": [0.4],
            "ROUGE-L (BART)": [0.5],
            "Flesch Reading Ease (BART)": [40.0],
            "BLEU (BERT)": [0.6],
            "METEOR (BERT)": [0.7],
            "ROUGE-1 (BERT)": [0.8],
            "ROUGE-2 (BERT)": [0.9],
            "ROUGE-L (BERT)": [1.0],
            "Flesch Reading Ease (BERT)": [41.0],
        }
    )
    structured_df = pd.DataFrame(
        {
            "Case study": ["Milk", "Milk"],
            "Summary": ["BART", "BERT"],
            "LLM": ["GPT", "GPT"],
            "Role": ["Yes", "Yes"],
            "Example": ["Yes", "Yes"],
            "Insight": ["Yes", "Yes"],
            "Output": ["", ""],
            "Input": ["", ""],
            "BLEU": [0.0, 0.0],
            "METEOR": [0.0, 0.0],
            "ROUGE-1": [0.0, 0.0],
            "ROUGE-2": [0.0, 0.0],
            "ROUGE-L": [0.0, 0.0],
            "Flesch Reading Ease": [0.0, 0.0],
            "Faithfulness (GPT)": ["", ""],
        }
    )

    notebook_update = get_notebook_function("update_structured_df")
    notebook_structured = structured_df.copy()
    notebook_updated = notebook_update(input_df.copy(), notebook_structured, "Milk", "GPT")
    assert notebook_updated is None

    def raise_missing(_name: str) -> FunctionType:
        raise KeyError("missing")

    monkeypatch.setattr("distill_abm.legacy.notebook_loader.get_notebook_function", raise_missing)
    fallback_updated = compat.update_structured_df(input_df.copy(), structured_df.copy(), "Milk", "GPT")
    assert fallback_updated.equals(notebook_structured)

    structured_path = tmp_path / "structured.csv"
    yes_no_path = tmp_path / "yes_no.csv"
    notebook_out = tmp_path / "notebook_filled.csv"
    fallback_out = tmp_path / "fallback_filled.csv"
    fallback_updated.to_csv(structured_path, index=False)
    pd.DataFrame(
        {
            "Case study": ["Milk"],
            "Summary": ["BART"],
            "LLM": ["GPT"],
            "Role": ["Yes"],
            "Example": ["Yes"],
            "Insight": ["Yes"],
            "Final LLM Faithfulness Score": [4],
        }
    ).to_csv(yes_no_path, index=False, sep=";")

    notebook_fill = get_notebook_function("fill_faithfulness_scores")
    notebook_fill(str(structured_path), str(yes_no_path), str(notebook_out))
    fallback_df = compat.fill_faithfulness_scores(
        frame=None,
        structured_data_path=structured_path,
        yes_no_path=yes_no_path,
        output_path=fallback_out,
    )

    notebook_filled = pd.read_csv(notebook_out)
    fallback_filled = pd.read_csv(fallback_out)
    assert fallback_filled.equals(notebook_filled)
    assert fallback_df.equals(notebook_filled)
