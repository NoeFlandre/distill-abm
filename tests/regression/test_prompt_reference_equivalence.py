from __future__ import annotations

from pathlib import Path

import yaml


def test_runtime_prompts_match_notebook_prompt_reference() -> None:
    runtime = yaml.safe_load(Path("configs/prompts.yaml").read_text(encoding="utf-8"))
    reference = yaml.safe_load(Path("configs/notebook_prompt_reference.yaml").read_text(encoding="utf-8"))
    reference_prompts = reference["prompts"]

    for key in [
        "context_prompt",
        "trend_prompt",
        "coverage_eval_prompt",
        "faithfulness_eval_prompt",
    ]:
        assert runtime[key].strip() == reference_prompts[key].strip()


def test_prompt_reference_assets_exist() -> None:
    expected = [
        "configs/notebook_prompt_assets/Evaluation/Qualitative Assessment using LLMs/Examples/Text/input_text.txt",
        "configs/notebook_prompt_assets/Evaluation/Qualitative Assessment using LLMs/Examples/Text/Example1.txt",
        "configs/notebook_prompt_assets/Evaluation/Qualitative Assessment using LLMs/Examples/Text/Example2.txt",
        "configs/notebook_prompt_assets/Evaluation/Qualitative Assessment using LLMs/Examples/Text/Example3.txt",
        "configs/notebook_prompt_assets/Evaluation/Qualitative Assessment using LLMs/Examples/Text/Example4.txt",
    ]
    for path in expected:
        assert Path(path).exists()
