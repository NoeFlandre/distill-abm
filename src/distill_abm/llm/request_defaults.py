"""Provider-aware request defaults used at LLM call sites."""

from __future__ import annotations

from distill_abm.configs.runtime_defaults import get_runtime_defaults

MISTRAL_DEFAULT_TEMPERATURE = 0.2


def resolve_request_temperature(provider: str, explicit_temperature: float | None = None) -> float | None:
    """Return the effective request temperature for one provider.

    An explicit temperature always wins. Otherwise, all providers use the runtime
    default except Mistral, which uses a lower debugging temperature.
    """

    if explicit_temperature is not None:
        return explicit_temperature
    if provider.strip().lower() == "mistral":
        return MISTRAL_DEFAULT_TEMPERATURE
    return get_runtime_defaults().llm_request.temperature
