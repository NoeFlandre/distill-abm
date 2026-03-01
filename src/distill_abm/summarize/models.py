"""BART and BERT summarization runners extracted from notebook workflows."""

from __future__ import annotations

from typing import Protocol

from distill_abm.summarize.legacy import chunk_text


class TokenizerLike(Protocol):
    """Tokenizer protocol used for chunked summarization workflows."""

    def encode(
        self,
        text: str,
        truncation: bool = False,
        add_special_tokens: bool = True,
    ) -> list[int]: ...

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str: ...


class BartPipelineLike(Protocol):
    """Protocol for HuggingFace summarization pipeline-compatible callables."""

    tokenizer: TokenizerLike

    def __call__(
        self,
        text: str,
        min_length: int,
        max_length: int,
        truncation: bool,
    ) -> list[dict[str, str]]: ...


class BertModelLike(Protocol):
    """Protocol for bert-extractive summarizer-like model callables."""

    def __call__(self, text: str, min_length: int, max_length: int) -> str: ...


class SummarizationError(RuntimeError):
    """Raised when external summarization runtimes fail."""


class BartSummarizerRunner:
    """Runs notebook-equivalent BART chunked abstractive summarization."""

    def __init__(
        self, summarizer: BartPipelineLike | None = None, model_name: str = "sshleifer/distilbart-cnn-12-6"
    ) -> None:
        self._summarizer = summarizer
        self.model_name = model_name

    def summarize(
        self,
        text: str,
        max_input_length: int = 1024,
        min_summary_length: int = 50,
        max_summary_length: int = 100,
    ) -> str:
        summarizer = self._get_summarizer()
        chunks = chunk_text(text, summarizer.tokenizer, max_input_length)
        parts: list[str] = []
        for chunk in chunks:
            response = summarizer(
                chunk,
                min_length=min_summary_length,
                max_length=max_summary_length,
                truncation=True,
            )
            parts.append(response[0]["summary_text"])
        return " ".join(parts)

    def _get_summarizer(self) -> BartPipelineLike:
        if self._summarizer is not None:
            return self._summarizer
        try:
            from transformers import pipeline
        except Exception as exc:
            raise SummarizationError(f"transformers not available: {exc}") from exc
        self._summarizer = pipeline("summarization", model=self.model_name)
        return self._summarizer


class BertSummarizerRunner:
    """Runs notebook-equivalent BERT chunked extractive summarization."""

    def __init__(
        self,
        model: BertModelLike | None = None,
        tokenizer: TokenizerLike | None = None,
        model_name: str = "bert-base-uncased",
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.model_name = model_name

    def summarize(
        self,
        text: str,
        max_input_length: int = 512,
        min_summary_length: int = 100,
        max_summary_length: int = 150,
    ) -> str:
        model, tokenizer = self._runtime()
        chunks = self._chunk_text(text, tokenizer, max_input_length)
        parts: list[str] = []
        for chunk in chunks:
            parts.append(model(chunk, min_length=min_summary_length, max_length=max_summary_length))
        return " ".join(parts)

    def _runtime(self) -> tuple[BertModelLike, TokenizerLike]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        try:
            from summarizer.bert import Summarizer
            from transformers import BertTokenizer
        except Exception as exc:
            raise SummarizationError(f"BERT summarization dependencies unavailable: {exc}") from exc
        self._model = Summarizer()
        self._tokenizer = BertTokenizer.from_pretrained(self.model_name)
        return self._model, self._tokenizer

    def _chunk_text(self, text: str, tokenizer: TokenizerLike, max_input_length: int) -> list[str]:
        tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
        chunks: list[str] = []
        for index in range(0, len(tokens), max_input_length):
            chunk_ids = tokens[index : index + max_input_length]
            chunks.append(tokenizer.decode(chunk_ids))
        return chunks


def summarize_with_bart(text: str, summarizer: BartPipelineLike | None = None) -> str:
    """Convenience wrapper for notebook-compatible BART summaries."""
    return BartSummarizerRunner(summarizer=summarizer).summarize(text)


def summarize_with_bert(
    text: str,
    model: BertModelLike | None = None,
    tokenizer: TokenizerLike | None = None,
) -> str:
    """Convenience wrapper for notebook-compatible BERT summaries."""
    return BertSummarizerRunner(model=model, tokenizer=tokenizer).summarize(text)
