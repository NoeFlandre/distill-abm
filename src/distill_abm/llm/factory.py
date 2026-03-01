"""Factory that instantiates provider adapters from config or CLI inputs."""

from __future__ import annotations

from typing import Any

from distill_abm.llm.adapters.anthropic_adapter import AnthropicAdapter
from distill_abm.llm.adapters.base import LLMAdapter
from distill_abm.llm.adapters.echo_adapter import EchoAdapter
from distill_abm.llm.adapters.janus_adapter import JanusAdapter
from distill_abm.llm.adapters.ollama_adapter import OllamaAdapter
from distill_abm.llm.adapters.openai_adapter import OpenAIAdapter


def create_adapter(provider: str, model: str, **kwargs: Any) -> LLMAdapter:
    """Creates an adapter instance using a normalized provider key."""
    key = provider.strip().lower()
    if key == "openai":
        return OpenAIAdapter(model=model, client=kwargs.get("client"))
    if key == "anthropic":
        return AnthropicAdapter(model=model, client=kwargs.get("client"))
    if key == "ollama":
        return OllamaAdapter(model=model, client=kwargs.get("client"), host=kwargs.get("host"))
    if key == "janus":
        return JanusAdapter(model=model, client=kwargs.get("client"))
    if key == "echo":
        return EchoAdapter(model=model)
    raise ValueError(f"unknown provider: {provider}")
