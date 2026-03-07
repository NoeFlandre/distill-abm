import time
from types import SimpleNamespace

import pytest

from distill_abm.llm.adapters.anthropic_adapter import AnthropicAdapter
from distill_abm.llm.adapters.base import (
    LLMAdapter,
    LLMMessage,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
)
from distill_abm.llm.adapters.ollama_adapter import OllamaAdapter
from distill_abm.llm.adapters.openai_adapter import OpenAIAdapter
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
    client = SimpleNamespace(
        chat=lambda **_: {
            "message": {"content": "ok"},
            "model": "deepseek-r1",
        },
    )
    adapter = create_adapter("ollama", model="deepseek-r1", client=client)
    assert isinstance(adapter, OllamaAdapter)


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


def test_openai_adapter_success() -> None:
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
        model="gpt-4o",
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: completion)))
    response = OpenAIAdapter(model="gpt-4o", client=client).complete(make_request())
    assert response.provider == "openai"
    assert response.text == "hello"


def test_openrouter_adapter_success() -> None:
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello from openrouter"))],
        model="google/gemini-3.1-pro-preview",
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: completion)))
    response = OpenRouterAdapter(model="google/gemini-3.1-pro-preview", client=client).complete(make_request())
    assert response.provider == "openrouter"
    assert response.text == "hello from openrouter"


def test_openai_adapter_includes_image_payload_when_present() -> None:
    seen: dict[str, object] = {}

    def _create(**payload):  # type: ignore[no-untyped-def]
        seen.update(payload)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
            model="gpt-4o",
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))
    request = LLMRequest(
        model="gpt-4o",
        messages=[LLMMessage(role="user", content="hello")],
        image_b64="abc",
    )
    OpenAIAdapter(model="gpt-4o", client=client).complete(request)
    messages = seen["messages"]
    assert isinstance(messages, list)
    assert isinstance(messages[0]["content"], list)


def test_anthropic_adapter_success() -> None:
    result = SimpleNamespace(content=[SimpleNamespace(type="text", text="hello")], model="claude-3-sonnet")
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: result))
    response = AnthropicAdapter(model="claude-3-sonnet", client=client).complete(make_request())
    assert response.provider == "anthropic"
    assert response.text == "hello"


def test_anthropic_adapter_includes_image_payload_when_present() -> None:
    seen: dict[str, object] = {}

    def _create(**payload):  # type: ignore[no-untyped-def]
        seen.update(payload)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="hello")], model="claude-3-sonnet")

    client = SimpleNamespace(messages=SimpleNamespace(create=_create))
    request = LLMRequest(
        model="claude-3-sonnet",
        messages=[LLMMessage(role="user", content="hello")],
        image_b64="abc",
    )
    AnthropicAdapter(model="claude-3-sonnet", client=client).complete(request)
    messages = seen["messages"]
    assert isinstance(messages, list)
    assert isinstance(messages[0]["content"], list)


def test_openai_adapter_timeout_is_wrapped() -> None:
    def _slow_create(**_: object) -> object:
        time.sleep(0.05)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="late"))],
            model="gpt-4o",
        )

    with pytest.raises(LLMProviderError, match="timed out"):
        OpenAIAdapter(
            model="gpt-4o",
            client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_slow_create))),
            timeout_seconds=0.001,
        ).complete(make_request())


def test_anthropic_adapter_timeout_is_wrapped() -> None:
    def _slow_create(**_: object) -> object:
        time.sleep(0.05)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="late")], model="claude-3-sonnet")

    with pytest.raises(LLMProviderError, match="timed out"):
        AnthropicAdapter(
            model="claude-3-sonnet",
            client=SimpleNamespace(messages=SimpleNamespace(create=_slow_create)),
            timeout_seconds=0.001,
        ).complete(make_request())


def test_ollama_adapter_success() -> None:
    client = SimpleNamespace(chat=lambda **_: {"message": {"content": "hi"}, "model": "deepseek-r1"})
    response = OllamaAdapter(model="deepseek-r1", client=client).complete(make_request())
    assert response.provider == "ollama"
    assert response.text == "hi"


def test_ollama_adapter_includes_image_payload_when_present() -> None:
    seen: dict[str, object] = {}

    def _chat(**payload):  # type: ignore[no-untyped-def]
        seen.update(payload)
        return {"message": {"content": "hi"}, "model": "deepseek-r1"}

    request = LLMRequest(
        model="deepseek-r1",
        messages=[LLMMessage(role="user", content="hello")],
        image_b64="abc",
    )
    OllamaAdapter(model="deepseek-r1", client=SimpleNamespace(chat=_chat)).complete(request)
    messages = seen["messages"]
    assert isinstance(messages, list)
    assert isinstance(messages[0]["images"], list)


def test_ollama_adapter_forwards_max_tokens_as_num_predict() -> None:
    seen: dict[str, object] = {}

    def _chat(**payload):  # type: ignore[no-untyped-def]
        seen.update(payload)
        return {"message": {"content": "hi"}, "model": "qwen3.5:0.8b"}

    request = LLMRequest(
        model="qwen3.5:0.8b",
        messages=[LLMMessage(role="user", content="hello")],
        max_tokens=321,
        temperature=0.5,
    )
    OllamaAdapter(model="qwen3.5:0.8b", client=SimpleNamespace(chat=_chat)).complete(request)
    options = seen["options"]
    assert isinstance(options, dict)
    assert options["temperature"] == 0.5
    assert options["num_predict"] == 321


def test_ollama_adapter_normalizes_chatresponse_objects() -> None:
    class ChatResponse:
        def __init__(self) -> None:
            self.model = "qwen3.5:0.8b"
            self.message = SimpleNamespace(content="hi-from-object")

        def model_dump(self) -> dict[str, object]:
            return {"model": self.model, "message": {"content": self.message.content}}

    response = OllamaAdapter(
        model="qwen3.5:0.8b",
        client=SimpleNamespace(chat=lambda **_: ChatResponse()),
    ).complete(make_request())
    assert response.provider == "ollama"
    assert response.text == "hi-from-object"
    assert response.raw["message"]["content"] == "hi-from-object"


def test_ollama_adapter_timeout_is_wrapped() -> None:
    def _slow_chat(**_: object) -> dict[str, object]:
        time.sleep(0.05)
        return {"message": {"content": "late"}, "model": "qwen3.5:0.8b"}

    with pytest.raises(LLMProviderError, match="timed out"):
        OllamaAdapter(
            model="qwen3.5:0.8b",
            client=SimpleNamespace(chat=_slow_chat),
            timeout_seconds=0.001,
        ).complete(make_request())


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
            OpenAIAdapter,
            SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(create=lambda **_: (_ for _ in ()).throw(RuntimeError("boom"))),
                ),
            ),
        ),
        (
            AnthropicAdapter,
            SimpleNamespace(messages=SimpleNamespace(create=lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))),
        ),
        (
            OllamaAdapter,
            SimpleNamespace(chat=lambda **_: (_ for _ in ()).throw(RuntimeError("boom"))),
        ),
        (
            OpenRouterAdapter,
            SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(create=lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))
                )
            ),
        ),
    ],
)
def test_external_errors_are_wrapped(adapter_cls: type[LLMAdapter], client: object) -> None:
    with pytest.raises(LLMProviderError):
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


def test_ollama_adapter_raises_on_connection_error() -> None:
    """Test that OllamaAdapter raises LLMProviderError on connection errors."""
    import urllib.error

    class FakeClient:
        def chat(self, **_: object) -> None:
            raise urllib.error.URLError("connection refused")

    with pytest.raises(LLMProviderError):
        OllamaAdapter(model="qwen3.5:0.8b", client=FakeClient()).complete(make_request())
