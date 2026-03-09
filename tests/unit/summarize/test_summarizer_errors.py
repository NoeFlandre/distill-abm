"""Tests for summarizer error paths when dependencies are missing."""

import pytest

from distill_abm.summarize.models import (
    BartSummarizerRunner,
    BertSummarizerRunner,
    LongformerExtSummarizerRunner,
    SummarizationError,
    T5SummarizerRunner,
)


def test_bart_runner_raises_summarization_error_on_missing_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that BartSummarizerRunner raises SummarizationError when transformers is missing."""

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "transformers":
            raise ImportError("simulated missing transformers")
        return __import__(name)

    monkeypatch.setattr("builtins.__import__", mock_import)
    runner = BartSummarizerRunner()

    with pytest.raises(SummarizationError) as exc_info:
        runner.summarize("abc")

    assert "transformers not available" in str(exc_info.value).lower()


def test_bert_runner_raises_summarization_error_on_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that BertSummarizerRunner raises SummarizationError when dependencies are missing."""

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name in ("summarizer.bert", "transformers"):
            raise ImportError("simulated missing dependency")
        return __import__(name)

    monkeypatch.setattr("builtins.__import__", mock_import)
    runner = BertSummarizerRunner()

    with pytest.raises(SummarizationError) as exc_info:
        runner.summarize("abc")

    assert "bert summarization dependencies unavailable" in str(exc_info.value).lower()


def test_t5_runner_raises_summarization_error_on_missing_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that T5SummarizerRunner raises SummarizationError when transformers is missing."""

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "transformers":
            raise ImportError("simulated missing transformers")
        return __import__(name)

    monkeypatch.setattr("builtins.__import__", mock_import)
    runner = T5SummarizerRunner()

    with pytest.raises(SummarizationError) as exc_info:
        runner.summarize("abc")

    assert "transformers not available" in str(exc_info.value).lower()


def test_longformer_ext_runner_raises_on_missing_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that LongformerExtSummarizerRunner raises SummarizationError when transformers is missing."""

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "transformers":
            raise ImportError("simulated missing transformers")
        return __import__(name)

    monkeypatch.setattr("builtins.__import__", mock_import)
    runner = LongformerExtSummarizerRunner()

    with pytest.raises(SummarizationError) as exc_info:
        runner.summarize("abc")

    assert "transformers not available" in str(exc_info.value).lower()
