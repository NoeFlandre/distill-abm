"""Mistral chat completion adapter with optional image and structured-output support."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from distill_abm.llm.adapters.base import LLMAdapter, LLMProviderError, LLMRequest, LLMResponse
from distill_abm.llm.adapters.openai_compatible_utils import (
    build_openai_compatible_payload,
    extract_openai_compatible_completion_text,
)
from distill_abm.llm.adapters.timeout_utils import run_with_timeout

DEFAULT_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
MISTRAL_MIN_REQUEST_INTERVAL_SECONDS = 1.0


class MistralAdapter(LLMAdapter):
    """Calls Mistral's chat completions endpoint through direct HTTP."""

    provider = "mistral"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_MISTRAL_BASE_URL,
        timeout_seconds: float = 120.0,
        transport: Any | None = None,
        time_fn: Any | None = None,
        sleep_fn: Any | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._transport = transport
        self._time_fn = time.monotonic if time_fn is None else time_fn
        self._sleep_fn = time.sleep if sleep_fn is None else sleep_fn
        self._request_lock = threading.Lock()
        self._next_request_at = 0.0

    def complete(self, request: LLMRequest) -> LLMResponse:
        api_key = self.api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise LLMProviderError("mistral api key missing: set MISTRAL_API_KEY")

        self._wait_for_request_slot()
        payload = build_openai_compatible_payload(request)
        payload["model"] = self.model or request.model

        try:
            raw_payload = run_with_timeout(
                timeout_seconds=self.timeout_seconds,
                label="mistral completion",
                fn=lambda: self._post_chat_completion(api_key=api_key, payload=payload),
            )
        except (HTTPError, URLError) as exc:
            raise LLMProviderError(f"mistral request failed: {exc}") from exc

        text = extract_openai_compatible_completion_text(raw_payload)
        raw_payload.setdefault("provider", "mistral")
        return LLMResponse(
            provider=self.provider,
            model=str(raw_payload.get("model", request.model)),
            text=text,
            raw=raw_payload,
        )

    def _wait_for_request_slot(self) -> None:
        with self._request_lock:
            now = float(self._time_fn())
            wait_seconds = self._next_request_at - now
            if wait_seconds > 0:
                self._sleep_fn(wait_seconds)
                now = float(self._time_fn())
            self._next_request_at = max(now, self._next_request_at) + MISTRAL_MIN_REQUEST_INTERVAL_SECONDS

    def _post_chat_completion(self, *, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            result = self._transport(api_key=api_key, base_url=self.base_url, payload=payload)
            return cast(dict[str, Any], dict(result))
        request = Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urlopen(request) as response:
            body = response.read().decode("utf-8")
        return cast(dict[str, Any], json.loads(body))
