"""OpenRouter adapter using the OpenAI-compatible chat completions API."""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from distill_abm.llm.adapters.base import LLMAdapter, LLMProviderError, LLMRequest, LLMResponse
from distill_abm.llm.adapters.openai_compatible_utils import (
    build_openai_compatible_payload,
    extract_openai_compatible_completion_text,
    normalize_openai_compatible_completion,
)
from distill_abm.llm.adapters.timeout_utils import run_with_timeout

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_ENDPOINT_QUANTIZATION_CACHE: dict[tuple[str, str, str], dict[str, str]] = {}
_ENDPOINT_QUANTIZATION_FAILURE_CACHE: dict[tuple[str, str, str], float] = {}
_ENDPOINT_QUANTIZATION_FAILURE_TTL_SECONDS = 60.0


class OpenRouterAdapter(LLMAdapter):
    """Calls OpenRouter through the OpenAI Python SDK interface."""

    provider = "openrouter"

    def __init__(
        self,
        model: str,
        client: Any | None = None,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        api_key: str | None = None,
        site_url: str | None = None,
        app_name: str = "distill-abm",
        timeout_seconds: float = 120.0,
        provider_quantization_resolver: Any | None = None,
    ) -> None:
        self.model = model
        self._client = client
        self.base_url = base_url
        self.api_key = api_key
        self.site_url = site_url
        self.app_name = app_name
        self.timeout_seconds = timeout_seconds
        self._provider_quantization_resolver = (
            provider_quantization_resolver or _resolve_openrouter_provider_quantization
        )

    def complete(self, request: LLMRequest) -> LLMResponse:
        payload = build_openai_compatible_payload(request)
        payload["model"] = self.model or request.model
        if "response_format" in payload:
            response_format = payload.get("response_format")
            if isinstance(response_format, dict):
                json_schema = response_format.get("json_schema")
                if isinstance(json_schema, dict):
                    json_schema.setdefault("strict", True)
            extra_body = payload.get("extra_body")
            if not isinstance(extra_body, dict):
                extra_body = {}
            extra_body["provider"] = {"require_parameters": True}
            extra_body["plugins"] = [{"id": "response-healing"}]
            payload["extra_body"] = extra_body
        completion = run_with_timeout(
            timeout_seconds=self.timeout_seconds,
            label="openrouter completion",
            fn=lambda: self._client_for_request().chat.completions.create(**payload),
        )
        normalized = normalize_openai_compatible_completion(completion)
        self._annotate_runtime_precision(normalized, model=str(normalized.get("model", request.model)))
        text = extract_openai_compatible_completion_text(normalized)
        model = str(normalized.get("model", request.model))
        normalized.setdefault("provider", "openrouter")
        return LLMResponse(provider=self.provider, model=model, text=text, raw=normalized)

    def _annotate_runtime_precision(self, raw: dict[str, Any], *, model: str) -> None:
        provider_name = _extract_runtime_provider_name(raw)
        if not provider_name or _raw_has_precision(raw):
            return
        try:
            precision = self._provider_quantization_resolver(
                model,
                provider_name,
                api_key=self.api_key or os.getenv("OPENROUTER_API_KEY"),
                base_url=self.base_url,
            )
        except Exception:
            return
        if not precision:
            return
        provider_block = raw.get("provider")
        if isinstance(provider_block, dict):
            provider_block.setdefault("name", provider_name)
            provider_block.setdefault("precision", precision)
            return
        provider_metadata = raw.get("provider_metadata")
        if not isinstance(provider_metadata, dict):
            provider_metadata = {}
        provider_metadata.setdefault("provider", provider_name)
        provider_metadata.setdefault("quantization", precision)
        raw["provider_metadata"] = provider_metadata

    def _client_for_request(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMProviderError("openrouter api key missing: set OPENROUTER_API_KEY")
        try:
            from openai import OpenAI
        except Exception as exc:
            raise LLMProviderError(f"openrouter OpenAI SDK unavailable: {exc}") from exc
        default_headers: dict[str, str] = {}
        if self.site_url:
            default_headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            default_headers["X-Title"] = self.app_name
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            default_headers=default_headers or None,
        )
        return self._client


def _extract_runtime_provider_name(raw: dict[str, Any]) -> str | None:
    provider_block = raw.get("provider")
    if isinstance(provider_block, str):
        normalized = provider_block.strip()
        if normalized:
            return normalized
    if isinstance(provider_block, dict):
        for key in ("name", "provider", "slug", "id"):
            value = provider_block.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    provider_metadata = raw.get("provider_metadata")
    if isinstance(provider_metadata, dict):
        for key in ("provider", "provider_name", "name", "slug", "id"):
            value = provider_metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _raw_has_precision(raw: dict[str, Any]) -> bool:
    provider_block = raw.get("provider")
    if isinstance(provider_block, dict):
        for key in ("precision", "quantization"):
            value = provider_block.get(key)
            if isinstance(value, str) and value.strip():
                return True
    provider_metadata = raw.get("provider_metadata")
    if isinstance(provider_metadata, dict):
        for key in ("precision", "quantization"):
            value = provider_metadata.get(key)
            if isinstance(value, str) and value.strip():
                return True
    for key in ("precision", "quantization", "provider_precision", "provider_quantization"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _resolve_openrouter_provider_quantization(
    model: str,
    provider_name: str,
    *,
    api_key: str | None,
    base_url: str,
) -> str | None:
    provider_map = _fetch_openrouter_endpoint_quantizations(model=model, api_key=api_key, base_url=base_url)
    return provider_map.get(provider_name.strip().lower())


def _fetch_openrouter_endpoint_quantizations(
    *,
    model: str,
    api_key: str | None,
    base_url: str,
) -> dict[str, str]:
    cache_key = (base_url.rstrip("/"), model, _quantization_cache_auth_fingerprint(api_key))
    if cache_key in _ENDPOINT_QUANTIZATION_CACHE:
        cached = _ENDPOINT_QUANTIZATION_CACHE[cache_key]
        return dict(cached)
    failure_timestamp = _ENDPOINT_QUANTIZATION_FAILURE_CACHE.get(cache_key)
    now = time.monotonic()
    if failure_timestamp is not None and now - failure_timestamp <= _ENDPOINT_QUANTIZATION_FAILURE_TTL_SECONDS:
        return {}
    quantizations = _request_openrouter_endpoint_quantizations(model=model, api_key=api_key, base_url=base_url)
    if quantizations:
        _ENDPOINT_QUANTIZATION_CACHE[cache_key] = dict(quantizations)
        _ENDPOINT_QUANTIZATION_FAILURE_CACHE.pop(cache_key, None)
        return dict(quantizations)
    _ENDPOINT_QUANTIZATION_FAILURE_CACHE[cache_key] = now
    return dict(quantizations)


def _quantization_cache_auth_fingerprint(api_key: str | None) -> str:
    if not api_key:
        return "missing"
    return f"set:{len(api_key)}:{api_key[-4:]}"


def _request_openrouter_endpoint_quantizations(
    *,
    model: str,
    api_key: str | None,
    base_url: str,
) -> dict[str, str]:
    if "/" not in model:
        return {}
    author, slug = model.split("/", 1)
    request = Request(
        f"{base_url.rstrip('/')}/models/{quote(author, safe='')}/{quote(slug, safe='')}/endpoints",
        headers={
            "Accept": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
    )
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError):
        return {}
    data = payload.get("data")
    endpoints = data.get("endpoints") if isinstance(data, dict) else None
    if not isinstance(endpoints, list):
        return {}
    quantizations: dict[str, str] = {}
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        provider = endpoint.get("provider_name")
        quantization = endpoint.get("quantization")
        if isinstance(provider, str) and provider.strip() and isinstance(quantization, str) and quantization.strip():
            quantizations[provider.strip().lower()] = quantization.strip()
    return quantizations
