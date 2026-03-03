from __future__ import annotations

import pandas as pd

from distill_abm.eval.doe import (
    clean_name,
    clean_statsmodels_feature_name,
    identify_factors_and_metrics,
)


def test_clean_name_and_feature_name() -> None:
    assert clean_name("My Column(1)") == "My_Column_1"
    assert clean_statsmodels_feature_name("C(Model):C(LLM)") == "Model_AND_LLM"


def test_identify_factors_and_metrics() -> None:
    frame = pd.DataFrame(
        {
            "Model": ["BART", "BERT", "BART"],
            "Role": ["Yes", "No", "Yes"],
            "Score": [0.1, 0.2, 0.3],
            "BLEU": [0.4, 0.5, 0.6],
            "Flag": [1, -1, 1],
        }
    )
    factors, metrics = identify_factors_and_metrics(frame)
    assert "Model" in factors
    assert "Score" in metrics


def test_identify_factors_preserves_column_order() -> None:
    frame = pd.DataFrame(
        {
            "LLM": ["GPT", "Claude", "DeepSeek"],
            "Role": ["Yes", "No", "Yes"],
            "Example": ["No", "Yes", "No"],
            "BLEU": [0.1, 0.2, 0.3],
        }
    )
    factors, _ = identify_factors_and_metrics(frame)
    assert factors == ["LLM", "Role", "Example"]
