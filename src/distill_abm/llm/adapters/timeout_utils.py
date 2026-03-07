"""Shared timeout helpers for provider adapters."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from distill_abm.llm.adapters.base import LLMProviderError


def run_with_timeout(
    *,
    timeout_seconds: float,
    label: str,
    fn: Callable[[], Any],
) -> Any:
    """Run one provider call with a hard timeout and normalized error wrapping."""
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fn)
            return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError as exc:
        raise LLMProviderError(f"{label} timed out after {timeout_seconds:.1f}s") from exc
    except LLMProviderError:
        raise
    except Exception as exc:
        raise LLMProviderError(f"{label} failed: {exc}") from exc
