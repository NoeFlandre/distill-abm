from __future__ import annotations

from distill_abm.pipeline.full_case_smoke import resolve_parallel_trend_workers
from distill_abm.pipeline.parallelism import DEFAULT_MAX_PARALLEL_TRENDS, resolve_provider_worker_count


def test_resolve_provider_worker_count_serializes_mistral() -> None:
    assert resolve_provider_worker_count(" MISTRAL ", default_workers=6) == 1


def test_resolve_provider_worker_count_preserves_default_for_other_providers() -> None:
    assert resolve_provider_worker_count("openrouter", default_workers=3) == 3


def test_full_case_trend_default_uses_shared_parallelism_policy() -> None:
    assert resolve_parallel_trend_workers("openrouter") == DEFAULT_MAX_PARALLEL_TRENDS
