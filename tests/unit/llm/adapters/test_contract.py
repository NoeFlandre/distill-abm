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
from distill_abm.llm.adapters.janus_adapter import JanusAdapter
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


def test_llm_request_default_temperature_is_point_five() -> None:
    req = LLMRequest(model="test-model", messages=[LLMMessage(role="user", content="hello")])
    assert req.temperature == 0.5


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
                    model="qwen/qwen3-vl-235b-a22b-thinking",
                )
            )
        )
    )
    adapter = create_adapter("openrouter", model="qwen/qwen3-vl-235b-a22b-thinking", client=client)
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
        model="qwen/qwen3-vl-235b-a22b-thinking",
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: completion)))
    response = OpenRouterAdapter(model="qwen/qwen3-vl-235b-a22b-thinking", client=client).complete(make_request())
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


def test_janus_adapter_passes_image_and_sampling_fields() -> None:
    class FakeJanusClient:
        def __init__(self) -> None:
            self.payload: dict[str, object] = {}

        def generate(
            self,
            prompt: str,
            image_b64: str | None,
            model: str,
            max_tokens: int | None,
            temperature: float | None,
        ) -> str:
            self.payload = {
                "prompt": prompt,
                "image_b64": image_b64,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            return "vision-output"

    client = FakeJanusClient()
    request = LLMRequest(
        model="janus-pro",
        messages=[LLMMessage(role="user", content="hello-image")],
        image_b64="abc",
        max_tokens=512,
        temperature=0.5,
    )
    JanusAdapter(model="janus-pro", client=client).complete(request)
    assert client.payload["prompt"] == "hello-image"
    assert client.payload["image_b64"] == "abc"
    assert client.payload["max_tokens"] == 512
    assert client.payload["temperature"] == 0.5


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
