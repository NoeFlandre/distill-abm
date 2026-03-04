from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from distill_abm.configs.models import PromptsConfig
from distill_abm.eval.metrics import SummaryScores
from distill_abm.llm.adapters.base import LLMAdapter, LLMProviderError, LLMRequest, LLMResponse
from distill_abm.pipeline import helpers


def test_build_context_prompt_includes_role_only_when_enabled(tmp_path: Path) -> None:
    prompts = PromptsConfig(
        context_prompt="Context {parameters} {documentation}",
        trend_prompt="Trend",
        style_features={"role": "ROLE", "example": "EXAMPLE"},
    )
    params = tmp_path / "params.txt"
    docs = tmp_path / "docs.txt"
    params.write_text("p=1", encoding="utf-8")
    docs.write_text("d=1", encoding="utf-8")
    include_role = helpers.build_context_prompt(
        inputs_csv_path=params,
        inputs_doc_path=docs,
        prompts=prompts,
        enabled={"example"},
    )
    assert "ROLE" not in include_role
    assert "Context" in include_role


def test_build_context_prompt_includes_role_when_enabled(tmp_path: Path) -> None:
    prompts = PromptsConfig(
        context_prompt="Context {parameters} {documentation}",
        trend_prompt="Trend",
        style_features={"role": "ROLE"},
    )
    params = tmp_path / "params.txt"
    docs = tmp_path / "docs.txt"
    params.write_text("p=1", encoding="utf-8")
    docs.write_text("d=1", encoding="utf-8")
    prompts_text = helpers.build_context_prompt(
        inputs_csv_path=params,
        inputs_doc_path=docs,
        prompts=prompts,
        enabled=None,
    )
    assert prompts_text.startswith("ROLE")


def test_build_trend_prompt_includes_optional_stats_csv_and_plot_description() -> None:
    prompts = PromptsConfig(
        context_prompt="Context {parameters} {documentation}",
        trend_prompt="Trend {description} {context}",
        style_features={"role": "ROLE", "example": "EXAMPLE", "insights": "INSIGHTS"},
    )
    prompt = helpers.build_trend_prompt(
        prompts=prompts,
        metric_description="coverage",
        context="context",
        plot_description="Line A",
        evidence_mode="table-csv",
        stats_table_csv="time_step,mean,std,min,max,median\n0,1.0,0.1,0.8,1.2,1.0\n",
        enabled={"example"},
    )
    assert prompt.startswith("Trend coverage context")
    assert "\n\nEXAMPLE" in prompt
    assert "INSIGHTS" not in prompt
    assert "time_step,mean,std,min,max,median" in prompt


def test_invoke_adapter_retries_transient_provider_errors() -> None:
    class FlakyAdapter(LLMAdapter):
        provider = "flaky"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, request: LLMRequest) -> LLMResponse:
            self.calls += 1
            if self.calls < 3:
                raise LLMProviderError("temporary timeout")
            return LLMResponse(provider="flaky", model=request.model, text="final answer", raw={})

    adapter = FlakyAdapter()
    text = helpers.invoke_adapter(
        adapter=adapter,
        model="test-model",
        prompt="hello",
        max_retries=2,
        retry_backoff_seconds=0.0,
    )
    assert text == "final answer"
    assert adapter.calls == 3


def test_invoke_adapter_raises_after_retry_budget_exhausted() -> None:
    class AlwaysFailAdapter(LLMAdapter):
        provider = "always-fail"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, request: LLMRequest) -> LLMResponse:
            _ = request
            self.calls += 1
            raise LLMProviderError("still failing")

    adapter = AlwaysFailAdapter()
    try:
        helpers.invoke_adapter(
            adapter=adapter,
            model="test-model",
            prompt="hello",
            max_retries=1,
            retry_backoff_seconds=0.0,
        )
    except LLMProviderError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected invoke_adapter to raise after retries are exhausted")
    assert "after 2 attempt(s)" in message
    assert adapter.calls == 2


def test_build_stats_csv_uses_expected_column_order() -> None:
    stats_table = pd.DataFrame(
        {
            "time_step": [0],
            "mean": [1.0],
            "std": [0.1],
            "min": [0.8],
            "max": [1.2],
            "median": [1.0],
        }
    )
    csv_text = helpers.build_stats_csv(stats_table)
    assert csv_text.startswith("time_step,mean,std,min,max,median\n")


def test_resolve_evidence_mode_maps_compatibility_aliases() -> None:
    assert helpers.resolve_evidence_mode("stats-markdown") == "table-csv"
    assert helpers.resolve_evidence_mode("stats-image") == "table-csv"
    assert helpers.resolve_evidence_mode("plot+stats") == "plot+table"


def test_load_existing_rows_if_compatible_rejects_schema_mismatch() -> None:
    path = Path("/tmp/mismatch.csv")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["a", "b"])
        writer.writerow(["x", "y"])
    rows = helpers.load_existing_rows_if_compatible(path, ["a"])
    assert rows == {}


def test_summarize_report_text_prefers_summary_when_enabled_with_mounted_summarizers() -> None:
    def fake_bart(text: str) -> str:
        return f"bart:{text}"

    def fake_bert(text: str) -> str:
        return "bert::abc"

    assert (
        helpers.summarize_report_text(
            text="raw",
            skip_summarization=False,
            summarize_with_bart_fn=fake_bart,
            summarize_with_bert_fn=fake_bert,
        )
        == "bart:raw\nbert::abc"
    )


def test_summarize_report_text_pair_returns_raw_and_summary() -> None:
    def fake_bart(text: str) -> str:
        return f"bart:{text}"

    def fake_bert(text: str) -> str:
        return "bert::abc"

    raw, summary = helpers.summarize_report_text_pair(
        text="raw",
        skip_summarization=False,
        summarize_with_bart_fn=fake_bart,
        summarize_with_bert_fn=fake_bert,
    )
    assert raw == "raw"
    assert summary == "bart:raw\nbert::abc"


def test_summarize_report_text_pair_includes_additional_summarizers() -> None:
    def fake_bart(text: str) -> str:
        return f"bart:{text}"

    def fake_bert(text: str) -> str:
        return f"bert:{text}"

    def fake_t5(text: str) -> str:
        return f"t5:{text}"

    def fake_longformer(text: str) -> str:
        return f"longformer:{text}"

    _raw, summary = helpers.summarize_report_text_pair(
        text="raw",
        skip_summarization=False,
        summarize_with_bart_fn=fake_bart,
        summarize_with_bert_fn=fake_bert,
        additional_summarizers=[
            ("t5", fake_t5),
            ("longformer_ext", fake_longformer),
        ],
    )

    assert summary == "bart:raw\nbert:raw\nt5:raw\nlongformer:raw"


def test_summarize_report_text_pair_skips_failing_additional_summarizer() -> None:
    def fake_bart(text: str) -> str:
        return f"bart:{text}"

    def fake_bert(text: str) -> str:
        return f"bert:{text}"

    def fake_t5(_text: str) -> str:
        raise RuntimeError("t5 unavailable")

    _raw, summary = helpers.summarize_report_text_pair(
        text="raw",
        skip_summarization=False,
        summarize_with_bart_fn=fake_bart,
        summarize_with_bert_fn=fake_bert,
        additional_summarizers=[("t5", fake_t5)],
    )

    assert summary == "bart:raw\nbert:raw"


def test_write_report_with_both_full_and_summary_scores(tmp_path: Path) -> None:
    def fake_scores(token_f1: float) -> SummaryScores:
        return SummaryScores(
            token_f1=token_f1,
            precision=0.1,
            recall=0.2,
            bleu=0.3,
            meteor=0.4,
            rouge1=0.5,
            rouge2=0.6,
            rouge_l=0.7,
            flesch_reading_ease=75.0,
            reference_length=10,
            candidate_length=5,
        )

    report_path = helpers.write_report(
        output_dir=tmp_path,
        context="context",
        trend_full="full trend",
        trend_summary="summary trend",
        scores=fake_scores(0.9),
        full_scores=fake_scores(0.7),
        summary_scores=fake_scores(0.8),
        include_extended_columns=True,
    )

    with report_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0][0] == "context_response"
    assert "trend_full_response" in rows[0]
    assert "summary_full_response" not in rows[0]
    assert "full_token_f1" in rows[0]
    assert "summary_token_f1" in rows[0]
    assert rows[1][1] == "summary trend"
