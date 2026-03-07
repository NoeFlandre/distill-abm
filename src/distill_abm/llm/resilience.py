"""Transient-error classification and simple in-memory circuit breakers."""

from __future__ import annotations

import time
from dataclasses import dataclass

from distill_abm.llm.adapters.base import LLMProviderError

TRANSIENT_ERROR_SNIPPETS = (
    "timeout",
    "timed out",
    "429",
    "500",
    "503",
    "rate limit",
    "temporarily unavailable",
    "connection reset",
    "connection refused",
    "samebatch may not be specified",
)
CIRCUIT_BREAKER_THRESHOLD = 3
CIRCUIT_BREAKER_OPEN_SECONDS = 60.0


@dataclass
class CircuitBreakerState:
    failure_count: int = 0
    opened_until: float | None = None


_BREAKER_STATES: dict[tuple[str, str], CircuitBreakerState] = {}


def reset_circuit_breakers() -> None:
    """Clear all in-memory breaker state."""
    _BREAKER_STATES.clear()


def ensure_circuit_closed(*, provider: str, model: str, now: float | None = None) -> None:
    """Raise immediately if the provider/model circuit is still open."""
    state = _BREAKER_STATES.get((provider, model))
    if state is None or state.opened_until is None:
        return
    current = time.monotonic() if now is None else now
    if current >= state.opened_until:
        state.failure_count = 0
        state.opened_until = None
        return
    remaining = state.opened_until - current
    raise LLMProviderError(f"circuit open for {provider}:{model}; retry after {remaining:.1f}s")


def record_success(*, provider: str, model: str) -> None:
    """Reset breaker state after one successful provider call."""
    _BREAKER_STATES.pop((provider, model), None)


def record_failure(*, provider: str, model: str, error: str, now: float | None = None) -> None:
    """Record one transient failure and open the circuit if needed."""
    if not is_transient_provider_error(error):
        return
    current = time.monotonic() if now is None else now
    state = _BREAKER_STATES.setdefault((provider, model), CircuitBreakerState())
    state.failure_count += 1
    if state.failure_count >= CIRCUIT_BREAKER_THRESHOLD:
        state.opened_until = current + CIRCUIT_BREAKER_OPEN_SECONDS


def is_transient_provider_error(error: str) -> bool:
    """Return whether an error message looks transient and worth retrying."""
    lowered = error.lower()
    return any(snippet in lowered for snippet in TRANSIENT_ERROR_SNIPPETS)
