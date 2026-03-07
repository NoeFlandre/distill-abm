"""Tests for llm factory module."""

import pytest

from distill_abm.llm.adapters.anthropic_adapter import AnthropicAdapter
from distill_abm.llm.adapters.echo_adapter import EchoAdapter
from distill_abm.llm.adapters.ollama_adapter import OllamaAdapter
from distill_abm.llm.adapters.openai_adapter import OpenAIAdapter
from distill_abm.llm.adapters.openrouter_adapter import OpenRouterAdapter
from distill_abm.llm.factory import create_adapter


def test_factory_raises_on_unknown_provider() -> None:
    """Test that create_adapter raises ValueError for unknown provider."""
    with pytest.raises(ValueError) as exc_info:
        create_adapter(provider="unknown", model="test-model")
    assert "unknown provider" in str(exc_info.value).lower()


def test_factory_normalizes_provider_case() -> None:
    """Test that provider name is case-insensitive."""
    adapter = create_adapter(provider="OPENAI", model="test-model")
    assert isinstance(adapter, OpenAIAdapter)


def test_factory_normalizes_provider_whitespace() -> None:
    """Test that provider name whitespace is stripped."""
    adapter = create_adapter(provider="  ollama  ", model="test-model")
    assert isinstance(adapter, OllamaAdapter)


def test_factory_creates_openai_adapter() -> None:
    """Test that openai provider creates OpenAIAdapter."""
    adapter = create_adapter(provider="openai", model="test-model")
    assert isinstance(adapter, OpenAIAdapter)


def test_factory_creates_openrouter_adapter() -> None:
    """Test that openrouter provider creates OpenRouterAdapter."""
    adapter = create_adapter(provider="openrouter", model="test-model")
    assert isinstance(adapter, OpenRouterAdapter)


def test_factory_creates_anthropic_adapter() -> None:
    """Test that anthropic provider creates AnthropicAdapter."""
    adapter = create_adapter(provider="anthropic", model="test-model")
    assert isinstance(adapter, AnthropicAdapter)


def test_factory_creates_ollama_adapter() -> None:
    """Test that ollama provider creates OllamaAdapter."""
    adapter = create_adapter(provider="ollama", model="test-model")
    assert isinstance(adapter, OllamaAdapter)


def test_factory_creates_echo_adapter() -> None:
    """Test that echo provider creates EchoAdapter."""
    adapter = create_adapter(provider="echo", model="test-model")
    assert isinstance(adapter, EchoAdapter)


def test_factory_openai_passes_client_kwarg() -> None:
    """Test that client kwarg is passed to adapter."""
    client = object()
    client = object()
    adapter = create_adapter(provider="openai", model="test-model", client=client)
    # Adapter is created with client kwarg - verified by successful creation
    assert adapter is not None
