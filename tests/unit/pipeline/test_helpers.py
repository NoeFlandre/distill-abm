from __future__ import annotations

import csv
from pathlib import Path

from distill_abm.configs.models import PromptsConfig
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


def test_build_trend_prompt_includes_optional_stats_markdown_and_plot_description() -> None:
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
        evidence_mode="stats-markdown",
        stats_markdown="| time_step | mean | std | min | max | median |",
        enabled={"example"},
    )
    assert prompt.startswith("Trend coverage context")
    assert "\n\nEXAMPLE" in prompt
    assert "INSIGHTS" not in prompt
    assert "| time_step | mean | std | min | max | median |" in prompt


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
