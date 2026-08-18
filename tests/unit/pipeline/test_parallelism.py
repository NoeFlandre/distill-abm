from __future__ import annotations

from distill_abm.pipeline.parallelism import resolve_provider_worker_count


def test_resolve_provider_worker_count_serializes_mistral() -> None:
    assert resolve_provider_worker_count(" MISTRAL ", default_workers=6) == 1


def test_resolve_provider_worker_count_preserves_default_for_other_providers() -> None:
    assert resolve_provider_worker_count("openrouter", default_workers=3) == 3
