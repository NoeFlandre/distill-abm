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
from distill_abm.llm.adapters.janus_adapter import JanusAdapter
from distill_abm.llm.adapters.ollama_adapter import OllamaAdapter
from distill_abm.llm.adapters.openai_adapter import OpenAIAdapter
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


def test_openai_adapter_success() -> None:
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
        model="gpt-4o",
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: completion)))
    response = OpenAIAdapter(model="gpt-4o", client=client).complete(make_request())
    assert response.provider == "openai"
    assert response.text == "hello"


def test_anthropic_adapter_success() -> None:
    result = SimpleNamespace(content=[SimpleNamespace(type="text", text="hello")], model="claude-3-sonnet")
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: result))
    response = AnthropicAdapter(model="claude-3-sonnet", client=client).complete(make_request())
    assert response.provider == "anthropic"
    assert response.text == "hello"


def test_ollama_adapter_success() -> None:
    client = SimpleNamespace(chat=lambda **_: {"message": {"content": "hi"}, "model": "deepseek-r1"})
    response = OllamaAdapter(model="deepseek-r1", client=client).complete(make_request())
    assert response.provider == "ollama"
    assert response.text == "hi"


def test_janus_adapter_success() -> None:
    class FakeJanusClient:
        def generate(
            self,
            prompt: str,
            image_b64: str | None,
            model: str,
            max_tokens: int | None,
            temperature: float | None,
        ) -> str:
            assert prompt == "hello"
            assert image_b64 is None
            return "vision-output"

    response = JanusAdapter(model="janus-pro", client=FakeJanusClient()).complete(make_request())
    assert response.provider == "janus"
    assert response.text == "vision-output"


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
    ],
)
def test_external_errors_are_wrapped(adapter_cls: type[LLMAdapter], client: object) -> None:
    with pytest.raises(LLMProviderError):
        adapter_cls(model="x", client=client).complete(make_request())  # type: ignore[call-arg]
