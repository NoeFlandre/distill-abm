from __future__ import annotations

from distill_abm.pipeline.doe_smoke_models import (
    CANONICAL_DOE_MODEL_IDS,
    CANONICAL_EVIDENCE_MODES,
    CANONICAL_REPETITIONS,
    canonical_prompt_variants,
    canonical_summarization_specs,
)


def test_canonical_doe_model_and_evidence_defaults_are_stable() -> None:
    assert CANONICAL_DOE_MODEL_IDS == ("qwen3_5_27b", "kimi_k2_5", "gemini_3_1_pro_preview", "claude_opus_4_6")
    assert CANONICAL_EVIDENCE_MODES == ("plot", "table", "plot+table")
    assert CANONICAL_REPETITIONS == (1, 2, 3)


def test_canonical_prompt_and_summarization_specs_preserve_benchmark_contract() -> None:
    prompt_variants = canonical_prompt_variants()
    summarization_specs = canonical_summarization_specs()

    assert [variant.variant_id for variant in prompt_variants] == [
        "none",
        "role",
        "insights",
        "example",
        "role+example",
        "role+insights",
        "insights+example",
        "all_three",
    ]
    assert [spec.summarization_mode for spec in summarization_specs] == [
        "none",
        "bart",
        "bert",
        "t5",
        "longformer_ext",
    ]
