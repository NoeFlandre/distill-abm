from __future__ import annotations

from typing import Any, cast

from distill_abm.summarize.models import (
    BartSummarizerRunner,
    BertSummarizerRunner,
    LongformerExtSummarizerRunner,
    T5SummarizerRunner,
    summarize_with_longformer_ext,
    summarize_with_t5,
)


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


class FakeT5Pipeline:
    tokenizer = FakeTokenizer()

    def __call__(
        self,
        text: str,
        min_length: int,
        max_length: int,
        truncation: bool,
    ) -> list[dict[str, str]]:
        _ = (min_length, max_length, truncation)
        return [{"summary_text": text.title()}]


class FakeLongformerPipeline:
    tokenizer = FakeTokenizer()

    def __call__(
        self,
        text: str,
        min_length: int,
        max_length: int,
        truncation: bool,
    ) -> list[dict[str, str]]:
        _ = (min_length, max_length, truncation)
        return [{"summary_text": f"{text[::-1]}"}]


def test_bart_runner_summarizes_text() -> None:
    runner = BartSummarizerRunner(summarizer=cast(Any, FakeBartPipeline()))
    out = runner.summarize("abcde", max_input_length=2)
    assert out == "AB CD E"


def test_bert_runner_summarizes_text() -> None:
    runner = BertSummarizerRunner(model=FakeBertModel(), tokenizer=FakeTokenizer())
    out = runner.summarize("ABCDE", max_input_length=2)
    assert out == "ab cd e"


def test_t5_runner_summarizes_text() -> None:
    runner = T5SummarizerRunner(summarizer=cast(Any, FakeT5Pipeline()))
    out = runner.summarize("abcde", max_input_length=2)
    assert out == "Ab Cd E"


def test_longformer_ext_runner_summarizes_text() -> None:
    runner = LongformerExtSummarizerRunner(summarizer=cast(Any, FakeLongformerPipeline()))
    out = runner.summarize("abc", max_input_length=1)
    assert out == "a b c"


def test_t5_and_longformer_ext_convenience_functions() -> None:
    assert (
        summarize_with_t5(
            "abc",
            summarizer=cast(Any, FakeT5Pipeline()),
            min_summary_length=2,
            max_summary_length=3,
        )
        == "Abc"
    )
    assert (
        summarize_with_longformer_ext(
            "abc",
            summarizer=cast(Any, FakeLongformerPipeline()),
            min_summary_length=2,
            max_summary_length=3,
            max_input_length=1,
        )
        == "a b c"
    )
