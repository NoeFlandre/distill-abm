"""Factory that instantiates provider adapters from config or CLI inputs."""

from __future__ import annotations

from typing import Any

from distill_abm.llm.adapters.anthropic_adapter import AnthropicAdapter
from distill_abm.llm.adapters.base import LLMAdapter
from distill_abm.llm.adapters.echo_adapter import EchoAdapter
from distill_abm.llm.adapters.mistral_adapter import MistralAdapter
from distill_abm.llm.adapters.ollama_adapter import OllamaAdapter
from distill_abm.llm.adapters.openai_adapter import OpenAIAdapter
from distill_abm.llm.adapters.openrouter_adapter import OpenRouterAdapter


def create_adapter(provider: str, model: str, **kwargs: Any) -> LLMAdapter:
    """Creates an adapter instance using a normalized provider key."""
    key = provider.strip().lower()
    if key == "openai":
        return OpenAIAdapter(model=model, client=kwargs.get("client"))
    if key == "openrouter":
        return OpenRouterAdapter(
            model=model,
            client=kwargs.get("client"),
            base_url=kwargs.get("base_url", "https://openrouter.ai/api/v1"),
            api_key=kwargs.get("api_key"),
            site_url=kwargs.get("site_url"),
            app_name=kwargs.get("app_name", "distill-abm"),
            timeout_seconds=kwargs.get("timeout_seconds", 120.0),
        )
    if key == "anthropic":
        return AnthropicAdapter(model=model, client=kwargs.get("client"))
    if key == "ollama":
        return OllamaAdapter(model=model, client=kwargs.get("client"), host=kwargs.get("host"))
    if key == "mistral":
        return MistralAdapter(
            model=model,
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url", "https://api.mistral.ai/v1"),
            timeout_seconds=kwargs.get("timeout_seconds", 120.0),
            transport=kwargs.get("transport"),
        )
    if key == "echo":
        return EchoAdapter(model=model)
    raise ValueError(f"unknown provider: {provider}")
