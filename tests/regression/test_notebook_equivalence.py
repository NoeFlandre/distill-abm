from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import distill_abm.legacy.compat as compat
from distill_abm.eval.doe import clean_name, clean_statsmodels_feature_name, identify_factors_and_metrics
from distill_abm.eval.qualitative import extract_coverage_score, extract_faithfulness_score, should_skip_row
from distill_abm.ingest.netlogo import remove_default_elements, remove_urls
from distill_abm.summarize.legacy import clean_context_response, clean_symbols
from distill_abm.summarize.postprocess import (
    capitalize_sentences,
    remove_hyphens_after_punctuation,
    remove_space_before_dot,
    remove_unnecessary_punctuation,
    remove_unnecessary_spaces_in_parentheses,
)


def test_core_text_utility_regressions() -> None:
    defaults = {"## WHAT IS IT?": "(a general understanding of what the model is trying to show or explain)"}
    doc = "## WHAT IS IT?\n\n(a general understanding of what the model is trying to show or explain)"

    assert remove_urls("a www.site.com") == "a "
    assert remove_default_elements(doc, defaults) == (
        "## ## WHAT IS IT?\n\n(a general understanding of what the model is trying to show or explain)"
    )
    assert clean_context_response("x</think>y") == "y"
    assert clean_symbols("#x*") == "x"
    assert remove_hyphens_after_punctuation("A. - b") == "A. b"
    assert remove_unnecessary_punctuation("A. , b") == "A. b"
    assert remove_unnecessary_spaces_in_parentheses("( a )") == "(a)"
    assert remove_space_before_dot("a .") == "a."
    assert capitalize_sentences("hello. world") == "Hello. World"


def test_evaluation_helper_regressions() -> None:
    frame = pd.DataFrame({"Model": ["BART", "BERT"], "Metric": [0.1, 0.2], "Flag": [1, -1]})
    factors, metrics = identify_factors_and_metrics(frame)

    assert extract_faithfulness_score("Faithfulness score: 5") == "5"
    assert extract_coverage_score("Coverage score: 4") == "4"
    assert should_skip_row({"score": 1}, "score") is True
    assert clean_name("A Column(1)") == "A_Column_1"
    assert clean_statsmodels_feature_name("C(A):C(B)") == "A_AND_B"
    assert set(factors) == {"Model"}
    assert set(metrics) == {"Metric", "Flag"}


def test_deterministic_doe_helper_regressions(tmp_path: Path) -> None:
    csv_path = tmp_path / "doe.csv"
    pd.DataFrame(
        {
            "PromptA": [-1, 1, -1, 1],
            "PromptB": [-1, -1, 1, 1],
            "BLEU": [0.1, 0.2, 0.3, 0.4],
            "ROUGE": [0.2, 0.3, 0.4, 0.5],
            "IgnoreMe": [2.0, 2.0, 2.0, 2.0],
        }
    ).to_csv(csv_path, index=False)

    avg_df, factor_df, metric_df = compat.read_and_parse_csv(csv_path, 2)
    assert avg_df.to_dict("list") == {
        "PromptA": [0.0, 0.0],
        "PromptB": [-1.0, 1.0],
        "BLEU": [0.15000000000000002, 0.35],
        "ROUGE": [0.25, 0.45],
    }
    assert factor_df.to_dict("list") == {"PromptA": [0.0, 0.0], "PromptB": [-1.0, 1.0]}
    assert metric_df.to_dict("list") == {"BLEU": [0.15000000000000002, 0.35], "ROUGE": [0.25, 0.45]}

    design_bundle = compat.create_factorial_design(csv_path, 2)
    assert isinstance(design_bundle, tuple)
    design_df, columns, specific_columns = design_bundle
    assert columns.columns.tolist() == ["PromptA", "PromptB"]
    assert specific_columns.columns.tolist() == ["BLEU", "ROUGE"]
    assert design_df.columns.tolist() == [
        "PromptA",
        "PromptB",
        "BLEU",
        "ROUGE",
        "PromptA_evaluating_BLEU",
        "PromptA_evaluating_ROUGE",
        "PromptB_evaluating_BLEU",
        "PromptB_evaluating_ROUGE",
        "PromptA_AND_PromptB",
        "PromptA_AND_PromptB_evaluating_BLEU",
        "PromptA_AND_PromptB_evaluating_ROUGE",
    ]

    results = compat.compute_results(csv_path, 2)
    assert len(results) == 2
    for metric_results in results:
        assert [name for name, _ in metric_results] in [
            [
                "PromptA_evaluating_BLEU",
                "PromptB_evaluating_BLEU",
                "PromptA_AND_PromptB_evaluating_BLEU",
            ],
            [
                "PromptA_evaluating_ROUGE",
                "PromptB_evaluating_ROUGE",
                "PromptA_AND_PromptB_evaluating_ROUGE",
            ],
        ]
        assert [value for _, value in metric_results] == pytest.approx([0.0, 100.0, 0.0])


def test_remove_evaluating_suffix_regression() -> None:
    assert compat.remove_evaluating_suffix("Role_AND_Example_evaluating_BLEU") == "Role_AND_Example"


def test_read_csv_to_df_regression(tmp_path: Path) -> None:
    csv_path = tmp_path / "simple.csv"
    frame = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    frame.to_csv(csv_path, index=False)
    loaded = compat.read_csv_to_df(csv_path)
    assert loaded.equals(frame)


def test_calculate_sums_and_sst_regression() -> None:
    frame = pd.DataFrame(
        {
            "Intercept_evaluating_BLEU": [1.0, 1.0, 1.0, 1.0],
            "PromptA_evaluating_BLEU": [-1.0, 1.0, -1.0, 1.0],
            "PromptB_evaluating_BLEU": [-1.0, -1.0, 1.0, 1.0],
        }
    )
    sums, sst = compat.calculate_sums_and_sst(frame, "BLEU", 2)
    assert [name for name, _ in sums] == [
        "Intercept_evaluating_BLEU",
        "PromptA_evaluating_BLEU",
        "PromptB_evaluating_BLEU",
    ]
    assert [value for _, value in sums] == pytest.approx([1.0, 0.0, 0.0])
    assert sst == pytest.approx(0.0)


def test_calculate_sst_regression(tmp_path: Path) -> None:
    csv_path = tmp_path / "sst.csv"
    pd.DataFrame(
        {
            "PromptA": [-1, 1, -1, 1],
            "PromptB": [-1, -1, 1, 1],
            "BLEU": [0.1, 0.2, 0.3, 0.4],
            "ROUGE": [0.2, 0.3, 0.4, 0.5],
        }
    ).to_csv(csv_path, index=False)
    rows, experiments = compat.calculate_sst(csv_path, 2)
    assert experiments == 2
    assert len(rows) == 2
    assert [name for name, _ in rows[0][0]] == [
        "BLEU",
        "PromptA_evaluating_BLEU",
        "PromptB_evaluating_BLEU",
        "PromptA_AND_PromptB_evaluating_BLEU",
    ]
    assert [name for name, _ in rows[1][0]] == [
        "ROUGE",
        "PromptA_evaluating_ROUGE",
        "PromptB_evaluating_ROUGE",
        "PromptA_AND_PromptB_evaluating_ROUGE",
    ]
    assert rows[0][1] == pytest.approx(0.009999999999999995)
    assert rows[1][1] == pytest.approx(0.010000000000000002)


def test_return_csv_regression(tmp_path: Path) -> None:
    csv_path = tmp_path / "return_input.csv"
    out_prefix = str(tmp_path / "return_out")
    pd.DataFrame(
        {
            "PromptA": [-1, 1, -1, 1],
            "PromptB": [-1, -1, 1, 1],
            "BLEU": [0.1, 0.2, 0.3, 0.4],
            "ROUGE": [0.2, 0.3, 0.4, 0.5],
        }
    ).to_csv(csv_path, index=False)
    compat.return_csv(csv_path, out_prefix, 2)
    output = pd.read_csv(Path(out_prefix + ".csv"))
    assert output.to_dict("list") == {
        "Feature": ["PromptA", "PromptB", "PromptA_AND_PromptB"],
        "BLEU": [0.0, 100.0, 0.0],
        "ROUGE": [0.0, 100.0, 0.0],
    }


def test_return_csv_2_regression(tmp_path: Path) -> None:
    csv_path = tmp_path / "return2_input.csv"
    out_prefix = str(tmp_path / "return2_out")
    pd.DataFrame(
        {
            "PromptA": [-1, 1, -1, 1],
            "PromptB": [-1, -1, 1, 1],
            "BLEU": [0.1, 0.2, 0.3, 0.4],
        }
    ).to_csv(csv_path, index=False)
    compat.return_csv_2(csv_path, out_prefix, 2)
    output = pd.read_csv(Path(out_prefix + ".csv"))
    assert output.to_dict("list") == {
        "Feature": ["PromptA_BLEU", "PromptB_BLEU", "PromptA_AND_PromptB_BLEU"],
        "Result": [0.0, 100.0, 0.0],
    }


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


def test_qualitative_csv_appenders_regression(tmp_path: Path) -> None:
    faith_input = tmp_path / "faith_input.csv"
    faith_out = tmp_path / "faith_out.csv"
    pd.DataFrame({"LLM Faithfulness Raw Response": ["Faithfulness score: 5", "Faithfulness score: 3"]}).to_csv(
        faith_input, index=False
    )
    compat.append_faithfulness_score(faith_input, faith_out)
    assert pd.read_csv(faith_out).to_dict("list") == {
        "LLM Faithfulness Raw Response": ["Faithfulness score: 5", "Faithfulness score: 3"],
        "LLM Faithfulness Score": [5, 3],
    }

    cov_input = tmp_path / "cov_input.csv"
    cov_out = tmp_path / "cov_out.csv"
    pd.DataFrame({"LLM Coverage Raw Response": ["Coverage score: 4", "Coverage score: 2"]}).to_csv(
        cov_input, index=False, sep=";"
    )
    compat.append_coverage_score(cov_input, cov_out)
    assert pd.read_csv(cov_out).to_dict("list") == {
        "LLM Coverage Raw Response": ["Coverage score: 4", "Coverage score: 2"],
        "LLM Coverage Score": [4, 2],
    }


def test_analyze_factorial_contributions_regression(tmp_path: Path) -> None:
    csv_path = tmp_path / "anova.csv"
    out_path = tmp_path / "anova_out.csv"
    pd.DataFrame(
        {
            "Role": ["Yes", "Yes", "No", "No"],
            "Example": ["Yes", "No", "Yes", "No"],
            "BLEU": [0.1, 0.2, 0.3, 0.4],
            "ROUGE-1": [0.2, 0.4, 0.6, 0.8],
        }
    ).to_csv(csv_path, index=False)
    frame = compat.analyze_factorial_contributions(csv_path, out_path, repetitions=2, max_interaction_order=2)
    assert frame is not None
    sorted_frame = frame.sort_values("Feature").reset_index(drop=True)
    assert sorted_frame["Feature"].tolist() == ["Example", "Role", "Role_AND_Example"]
    assert sorted_frame["BLEU"].tolist() == pytest.approx([20.000000000000018, 80.0, 0.0])
    assert sorted_frame["ROUGE-1"].tolist() == pytest.approx([20.000000000000018, 80.0, 0.0])


def test_sheet_helper_regressions(tmp_path: Path) -> None:
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

    updated = compat.update_structured_df(input_df.copy(), structured_df.copy(), "Milk", "GPT")
    assert updated.to_dict("list") == {
        "Case study": ["Milk", "Milk"],
        "Summary": ["BART", "BERT"],
        "LLM": ["GPT", "GPT"],
        "Role": ["Yes", "Yes"],
        "Example": ["Yes", "Yes"],
        "Insight": ["Yes", "Yes"],
        "Output": ["bart output", "bert output"],
        "Input": ["context prompt", "context prompt"],
        "BLEU": [0.1, 0.6],
        "METEOR": [0.2, 0.7],
        "ROUGE-1": [0.3, 0.8],
        "ROUGE-2": [0.4, 0.9],
        "ROUGE-L": [0.5, 1.0],
        "Flesch Reading Ease": [40.0, 41.0],
        "Faithfulness (GPT)": ["", ""],
    }

    structured_path = tmp_path / "structured.csv"
    yes_no_path = tmp_path / "yes_no.csv"
    output_path = tmp_path / "filled.csv"
    updated.to_csv(structured_path, index=False)
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

    filled = compat.fill_faithfulness_scores(
        frame=None,
        structured_data_path=structured_path,
        yes_no_path=yes_no_path,
        output_path=output_path,
    )
    assert filled["Faithfulness (GPT)"].tolist()[0] == pytest.approx(4.0)
    assert pd.isna(filled["Faithfulness (GPT)"].tolist()[1])
