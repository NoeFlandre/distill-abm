from __future__ import annotations

from typing import Any, cast

from distill_abm.summarize.models import BartSummarizerRunner, BertSummarizerRunner


class FakeTokenizer:
    def encode(self, text: str, truncation: bool = False, add_special_tokens: bool = True) -> list[int]:
        _ = (truncation, add_special_tokens)
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        _ = skip_special_tokens
        return "".join(chr(i) for i in ids)


class FakeBartPipeline:
    tokenizer = FakeTokenizer()

    def __call__(
        self,
        text: str,
        min_length: int,
        max_length: int,
        truncation: bool,
    ) -> list[dict[str, str]]:
        _ = (min_length, max_length, truncation)
        return [{"summary_text": text.upper()}]


class FakeBertModel:
    def __call__(self, text: str, min_length: int, max_length: int) -> str:
        _ = (min_length, max_length)
        return text.lower()


def test_bart_runner_summarizes_text() -> None:
    runner = BartSummarizerRunner(summarizer=cast(Any, FakeBartPipeline()))
    out = runner.summarize("abcde", max_input_length=2)
    assert out == "AB CD E"


def test_bert_runner_summarizes_text() -> None:
    runner = BertSummarizerRunner(model=FakeBertModel(), tokenizer=FakeTokenizer())
    out = runner.summarize("ABCDE", max_input_length=2)
    assert out == "ab cd e"
