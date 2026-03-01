from __future__ import annotations

from pathlib import Path

import pandas as pd

from distill_abm.eval.doe_full import analyze_factorial_anova


def test_analyze_factorial_anova_generates_table(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "Case study": ["Fauna", "Fauna", "Grazing", "Grazing"],
            "LLM": ["GPT", "Claude", "GPT", "Claude"],
            "Role": ["Yes", "No", "Yes", "No"],
            "BLEU": [0.4, 0.3, 0.5, 0.45],
            "ROUGE": [0.5, 0.35, 0.55, 0.5],
        }
    )
    source = tmp_path / "anova_input.csv"
    out = tmp_path / "anova_output.csv"
    frame.to_csv(source, index=False)

    result = analyze_factorial_anova(source, out, max_interaction_order=2)

    assert result is not None
    assert out.exists()
    assert "Feature" in result.columns
    assert "BLEU" in result.columns
