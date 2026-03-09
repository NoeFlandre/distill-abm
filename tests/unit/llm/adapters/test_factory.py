"""Tests for llm factory module."""

import pytest

from distill_abm.llm.adapters.echo_adapter import EchoAdapter
from distill_abm.llm.adapters.mistral_adapter import MistralAdapter
from distill_abm.llm.adapters.openrouter_adapter import OpenRouterAdapter
from distill_abm.llm.factory import create_adapter


def test_factory_raises_on_unknown_provider() -> None:
    """Test that create_adapter raises ValueError for unknown provider."""
    with pytest.raises(ValueError) as exc_info:
        create_adapter(provider="unknown", model="test-model")
    assert "unknown provider" in str(exc_info.value).lower()


def test_factory_normalizes_provider_case() -> None:
    """Test that provider name is case-insensitive."""
    adapter = create_adapter(provider="OPENROUTER", model="test-model")
    assert isinstance(adapter, OpenRouterAdapter)


def test_factory_normalizes_provider_whitespace() -> None:
    """Test that provider name whitespace is stripped."""
    adapter = create_adapter(provider="  mistral  ", model="mistral-medium-latest")
    assert isinstance(adapter, MistralAdapter)


def test_factory_creates_openrouter_adapter() -> None:
    """Test that openrouter provider creates OpenRouterAdapter."""
    adapter = create_adapter(provider="openrouter", model="test-model")
    assert isinstance(adapter, OpenRouterAdapter)
    assert adapter.timeout_seconds == 120.0


def test_factory_passes_openrouter_timeout_kwarg() -> None:
    """Test that openrouter timeout kwarg is passed through."""
    adapter = create_adapter(provider="openrouter", model="test-model", timeout_seconds=900.0)
    assert isinstance(adapter, OpenRouterAdapter)
    assert adapter.timeout_seconds == 900.0


def test_factory_creates_echo_adapter() -> None:
    """Test that echo provider creates EchoAdapter."""
    adapter = create_adapter(provider="echo", model="test-model")
    assert isinstance(adapter, EchoAdapter)


def test_factory_creates_mistral_adapter() -> None:
    """Test that mistral provider creates MistralAdapter."""
    adapter = create_adapter(provider="mistral", model="mistral-medium-latest")
    assert isinstance(adapter, MistralAdapter)


def test_factory_openrouter_passes_client_kwarg() -> None:
    """Test that client kwarg is passed to the OpenRouter adapter."""
    client = object()
    adapter = create_adapter(provider="openrouter", model="test-model", client=client)
    assert adapter is not None
