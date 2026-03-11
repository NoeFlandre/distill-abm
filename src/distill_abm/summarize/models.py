"""BART, BERT, T5, and Longformer-like summarization runners."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from distill_abm.summarize.reference_text import chunk_text

if TYPE_CHECKING:
    import torch


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


class TransformerPipelineLike(Protocol):
    """Protocol shared across Hugging Face summarization pipelines."""

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


@dataclass
class _TransformerBertExtractiveModel:
    """Compatibility fallback for Python runtimes where bert-extractive-summarizer breaks."""

    model: object
    tokenizer: object
    device: str

    def __call__(self, text: str, min_length: int, max_length: int) -> str:
        import torch

        sentences = _split_sentences(text)
        if not sentences:
            return text.strip()
        if len(sentences) == 1:
            return sentences[0]

        sentence_embeddings = self._embed(sentences)
        document_embedding = self._embed([" ".join(sentences)])[0]
        document_norm = torch.norm(document_embedding) + 1e-12
        scores = torch.mv(sentence_embeddings, document_embedding) / (
            torch.norm(sentence_embeddings, dim=1) * document_norm + 1e-12
        )

        ranked_indexes = sorted(range(len(sentences)), key=lambda index: float(scores[index]), reverse=True)
        selected_indexes: list[int] = []
        selected_word_count = 0
        for index in ranked_indexes:
            sentence_word_count = len(sentences[index].split())
            if selected_indexes and selected_word_count + sentence_word_count > max_length:
                continue
            selected_indexes.append(index)
            selected_word_count += sentence_word_count
            if selected_word_count >= min_length:
                break

        if not selected_indexes:
            return sentences[0]

        selected_indexes.sort()
        summary = " ".join(sentences[index] for index in selected_indexes).strip()
        return summary or sentences[0]

    def _embed(self, texts: list[str]) -> torch.Tensor:
        import torch

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
        hidden_state = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        masked_hidden_state = hidden_state * attention_mask
        token_counts = attention_mask.sum(dim=1).clamp(min=1)
        pooled = masked_hidden_state.sum(dim=1) / token_counts
        return pooled.cpu()


def _split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


class BartSummarizerRunner:
    """Run chunked BART summarization."""

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
        self._summarizer = cast(BartPipelineLike, pipeline("summarization", model=self.model_name))
        return self._summarizer


class BertSummarizerRunner:
    """Run chunked BERT summarization."""

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
            try:
                self._model, self._tokenizer = self._build_transformers_runtime()
                return self._model, self._tokenizer
            except Exception as fallback_exc:
                raise SummarizationError(
                    f"BERT summarization dependencies unavailable: {exc}; fallback failed: {fallback_exc}"
                ) from fallback_exc
        self._model = cast(BertModelLike, Summarizer())
        self._tokenizer = cast(TokenizerLike, BertTokenizer.from_pretrained(self.model_name))
        return self._model, self._tokenizer

    def _chunk_text(self, text: str, tokenizer: TokenizerLike, max_input_length: int) -> list[str]:
        tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
        chunks: list[str] = []
        for index in range(0, len(tokens), max_input_length):
            chunk_ids = tokens[index : index + max_input_length]
            chunks.append(tokenizer.decode(chunk_ids))
        return chunks

    def _build_transformers_runtime(self) -> tuple[BertModelLike, TokenizerLike]:
        try:
            import torch
            from transformers import BertModel, BertTokenizer
        except Exception as exc:
            raise SummarizationError(f"transformers not available: {exc}") from exc

        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        device = (
            "mps"
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
            else "cpu"
        )
        model = BertModel.from_pretrained(self.model_name)
        model.to(device)
        model.eval()
        return (
            cast(BertModelLike, _TransformerBertExtractiveModel(model=model, tokenizer=tokenizer, device=device)),
            cast(TokenizerLike, tokenizer),
        )


class T5SummarizerRunner:
    """Run chunked T5 summarization."""

    def __init__(
        self,
        summarizer: TransformerPipelineLike | None = None,
        model_name: str = "t5-small",
    ) -> None:
        self._summarizer = summarizer
        self.model_name = model_name

    def summarize(
        self,
        text: str,
        max_input_length: int = 1024,
        min_summary_length: int = 40,
        max_summary_length: int = 120,
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

    def _get_summarizer(self) -> TransformerPipelineLike:
        if self._summarizer is not None:
            return self._summarizer
        try:
            from transformers import pipeline
        except Exception as exc:
            raise SummarizationError(f"transformers not available: {exc}") from exc
        self._summarizer = cast(TransformerPipelineLike, pipeline("summarization", model=self.model_name))
        return self._summarizer


class LongformerExtSummarizerRunner:
    """Run chunked long-document transformer summarization."""

    def __init__(
        self,
        summarizer: TransformerPipelineLike | None = None,
        model_name: str = "allenai/led-base-16384",
    ) -> None:
        self._summarizer = summarizer
        self.model_name = model_name

    def summarize(
        self,
        text: str,
        max_input_length: int = 2048,
        min_summary_length: int = 64,
        max_summary_length: int = 180,
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

    def _get_summarizer(self) -> TransformerPipelineLike:
        if self._summarizer is not None:
            return self._summarizer
        try:
            from transformers import pipeline
        except Exception as exc:
            raise SummarizationError(f"transformers not available: {exc}") from exc
        self._summarizer = cast(TransformerPipelineLike, pipeline("summarization", model=self.model_name))
        return self._summarizer


def summarize_with_bart(text: str, summarizer: BartPipelineLike | None = None) -> str:
    """Convenience wrapper for BART summaries."""
    return BartSummarizerRunner(summarizer=summarizer).summarize(text)


def summarize_with_bert(
    text: str,
    model: BertModelLike | None = None,
    tokenizer: TokenizerLike | None = None,
) -> str:
    """Convenience wrapper for BERT summaries."""
    return BertSummarizerRunner(model=model, tokenizer=tokenizer).summarize(text)


def summarize_with_t5(
    text: str,
    summarizer: TransformerPipelineLike | None = None,
    min_summary_length: int = 40,
    max_summary_length: int = 120,
    max_input_length: int = 1024,
) -> str:
    """Convenience wrapper for T5 summaries."""
    return T5SummarizerRunner(summarizer=summarizer).summarize(
        text,
        min_summary_length=min_summary_length,
        max_summary_length=max_summary_length,
        max_input_length=max_input_length,
    )


def summarize_with_longformer_ext(
    text: str,
    summarizer: TransformerPipelineLike | None = None,
    min_summary_length: int = 64,
    max_summary_length: int = 180,
    max_input_length: int = 2048,
) -> str:
    """Convenience wrapper for Longformer-like summaries."""
    return LongformerExtSummarizerRunner(summarizer=summarizer).summarize(
        text,
        min_summary_length=min_summary_length,
        max_summary_length=max_summary_length,
        max_input_length=max_input_length,
    )
