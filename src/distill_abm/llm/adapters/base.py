"""Common adapter contract for all LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from distill_abm.configs.runtime_defaults import get_runtime_defaults

Role = Literal["system", "user", "assistant"]


class LLMMessage(BaseModel):
    """Represents one chat message in a provider-neutral format."""

    role: Role
    content: str


class LLMRequest(BaseModel):
    """Encapsulates generation inputs for both text-only and vision calls."""

    model: str
    messages: list[LLMMessage]
    temperature: float | None = Field(default_factory=lambda: get_runtime_defaults().llm_request.temperature)
    max_tokens: int | None = Field(default_factory=lambda: get_runtime_defaults().llm_request.max_tokens)
    image_b64: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def user_prompt(self) -> str:
        """Returns the latest user prompt for adapters needing a single string."""
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        return ""


class LLMResponse(BaseModel):
    """Normalizes provider outputs for downstream pipeline stages."""

    provider: str
    model: str
    text: str
    raw: dict[str, Any] = Field(default_factory=dict)


class LLMProviderError(RuntimeError):
    """Wraps provider-specific exceptions with actionable context."""


class LLMAdapter(ABC):
    """Adapter interface used by pipeline code to stay provider-agnostic."""

    provider: ClassVar[str]

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Runs one completion call and returns normalized output."""


def to_chat_messages(messages: list[LLMMessage]) -> list[dict[str, str]]:
    """Converts typed messages into the common role/content dictionary shape."""
    return [{"role": message.role, "content": message.content} for message in messages]
