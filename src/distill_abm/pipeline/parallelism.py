"""Provider-aware concurrency policies shared by smoke workflows."""

from __future__ import annotations

DEFAULT_MAX_PARALLEL_TRENDS = 6


def resolve_provider_worker_count(provider: str, *, default_workers: int) -> int:
    """Return the provider-adjusted worker count for one smoke workflow."""
    if provider.strip().lower() == "mistral":
        return 1
    return default_workers
