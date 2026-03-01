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
