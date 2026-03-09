import time
from types import SimpleNamespace

import pytest

from distill_abm.llm.adapters.base import (
    LLMAdapter,
    LLMMessage,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
)
from distill_abm.llm.adapters.mistral_adapter import MistralAdapter
from distill_abm.llm.adapters.openrouter_adapter import OpenRouterAdapter
from distill_abm.llm.factory import create_adapter


class DummyAdapter(LLMAdapter):
    provider = "dummy"

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(provider=self.provider, model=request.model, text="ok", raw={})


def make_request() -> LLMRequest:
    return LLMRequest(
        model="test-model",
        messages=[LLMMessage(role="user", content="hello")],
        temperature=0.1,
        max_tokens=100,
    )


def test_llm_request_default_temperature_is_one() -> None:
    req = LLMRequest(model="test-model", messages=[LLMMessage(role="user", content="hello")])
    assert req.temperature == 1.0


def test_adapter_contract_returns_response() -> None:
    response = DummyAdapter().complete(make_request())
    assert response.provider == "dummy"
    assert response.text == "ok"


def test_factory_creates_known_adapter() -> None:
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: None)))
    adapter = create_adapter("openrouter", model="google/gemini-3.1-pro-preview", client=client)
    assert isinstance(adapter, OpenRouterAdapter)


def test_factory_creates_openrouter_adapter() -> None:
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                    model="google/gemini-3.1-pro-preview",
                )
            )
        )
    )
    adapter = create_adapter("openrouter", model="google/gemini-3.1-pro-preview", client=client)
    assert isinstance(adapter, OpenRouterAdapter)


def test_openrouter_adapter_success() -> None:
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello from openrouter"), finish_reason="stop")],
        model="google/gemini-3.1-pro-preview",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: completion)))
    response = OpenRouterAdapter(model="google/gemini-3.1-pro-preview", client=client).complete(make_request())
    assert response.provider == "openrouter"
    assert response.text == "hello from openrouter"
    assert response.raw["model"] == "google/gemini-3.1-pro-preview"
    assert response.raw["choices"][0]["finish_reason"] == "stop"
    assert response.raw["usage"]["total_tokens"] == 15


def test_mistral_adapter_success_with_transport() -> None:
    request = make_request().model_copy(
        update={
            "metadata": {
                "structured_output_name": "structured_smoke_text",
                "structured_output_schema": {
                    "type": "object",
                    "properties": {"response_text": {"type": "string"}},
                    "required": ["response_text"],
                },
            }
        }
    )

    def transport(*, api_key: str, base_url: str, payload: dict[str, object]) -> dict[str, object]:
        assert api_key == "secret"
        assert base_url == "https://api.mistral.ai/v1"
        assert payload["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_smoke_text",
                "schema": request.metadata["structured_output_schema"],
            },
        }
        return {
            "model": "mistral-medium-latest",
            "choices": [{"message": {"content": '{"response_text":"ok"}'}}],
        }

    response = MistralAdapter(
        model="mistral-medium-latest",
        api_key="secret",
        transport=transport,
    ).complete(request)

    assert response.provider == "mistral"
    assert response.text == '{"response_text":"ok"}'


def test_mistral_adapter_raises_on_missing_api_key() -> None:
    with pytest.raises(LLMProviderError, match="mistral api key missing"):
        MistralAdapter(model="mistral-medium-latest", api_key=None).complete(make_request())


def test_openrouter_adapter_forwards_structured_output_metadata() -> None:
    seen: dict[str, object] = {}

    def _create(**payload):  # type: ignore[no-untyped-def]
        seen.update(payload)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"response_text":"ok"}'), finish_reason="stop")],
            model="nvidia/nemotron-nano-12b-v2-vl:free",
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))
    request = LLMRequest(
        model="nvidia/nemotron-nano-12b-v2-vl:free",
        messages=[LLMMessage(role="user", content="hello")],
        metadata={
            "structured_output_name": "structured_smoke_text",
            "structured_output_schema": {
                "type": "object",
                "properties": {"response_text": {"type": "string"}},
                "required": ["response_text"],
            },
        },
    )

    OpenRouterAdapter(model="nvidia/nemotron-nano-12b-v2-vl:free", client=client).complete(request)

    assert seen["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_smoke_text",
            "schema": request.metadata["structured_output_schema"],
        },
    }
def test_openrouter_adapter_timeout_is_wrapped() -> None:
    def _slow_create(**_: object) -> object:
        time.sleep(0.05)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="late"))],
            model="google/gemini-3.1-pro-preview",
        )

    with pytest.raises(LLMProviderError, match="timed out"):
        OpenRouterAdapter(
            model="google/gemini-3.1-pro-preview",
            client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_slow_create))),
            timeout_seconds=0.001,
        ).complete(make_request())


@pytest.mark.parametrize(
    ("adapter_cls", "client"),
    [
        (
            OpenRouterAdapter,
            SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(create=lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))
                )
            ),
        ),
        (
            MistralAdapter,
            SimpleNamespace(),
        ),
    ],
)
def test_external_errors_are_wrapped(adapter_cls: type[LLMAdapter], client: object) -> None:
    with pytest.raises(LLMProviderError):
        if adapter_cls is MistralAdapter:
            adapter = adapter_cls(
                model="mistral-medium-latest",
                api_key="secret",
                transport=lambda **_: (_ for _ in ()).throw(RuntimeError("boom")),
            )
            adapter.complete(make_request())  # type: ignore[call-arg]
        else:
            adapter_cls(model="x", client=client).complete(make_request())  # type: ignore[call-arg]


def test_openrouter_adapter_raises_on_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that OpenRouterAdapter raises LLMProviderError when API key is missing."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    
    adapter = OpenRouterAdapter(model="test-model")
    request = LLMRequest(
        model="test-model",
        messages=[LLMMessage(role="user", content="hello")],
        temperature=0.1,
        max_tokens=100,
    )
    
    with pytest.raises(LLMProviderError) as exc_info:
        adapter.complete(request)

    assert "api key missing" in str(exc_info.value).lower()


def test_openrouter_adapter_raises_on_completion_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that OpenRouterAdapter raises LLMProviderError when completion fails."""
    class FakeClient:
        class Completions:
            def create(self, **kwargs: object) -> None:
                raise RuntimeError("network error")
        
        chat = Completions()
    
    adapter = OpenRouterAdapter(model="test-model", client=FakeClient())
    request = LLMRequest(
        model="test-model",
        messages=[LLMMessage(role="user", content="hello")],
        temperature=0.1,
        max_tokens=100,
    )
    
    with pytest.raises(LLMProviderError) as exc_info:
        adapter.complete(request)
    
    assert "completion failed" in str(exc_info.value).lower()
