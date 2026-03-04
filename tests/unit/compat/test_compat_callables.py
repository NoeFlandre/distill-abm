from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, cast

import pytest

from distill_abm.compat import compat_callables
from distill_abm.compat.compat_callables import _call_notebook_first as call_notebook_first

_compat = cast(Any, compat_callables)
_call_notebook_first = call_notebook_first


def test_call_notebook_first_uses_notebook_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    def fallback(value: str) -> str:
        return f"fallback:{value}"

    def notebook_handler(value: str) -> str:
        return f"notebook:{value}"

    monkeypatch.setattr(_compat.notebook_loader, "should_dispatch_notebook", lambda _name: True)
    monkeypatch.setattr(_compat.notebook_loader, "get_notebook_function", lambda _name: notebook_handler)

    assert _call_notebook_first("x", fallback, "value") == "notebook:value"


def test_call_notebook_first_falls_back_when_not_dispatched(monkeypatch: pytest.MonkeyPatch) -> None:
    def fallback(value: str) -> str:
        return f"fallback:{value}"

    monkeypatch.setattr(_compat.notebook_loader, "should_dispatch_notebook", lambda _name: False)
    assert _call_notebook_first("x", fallback, "value") == "fallback:value"


def test_call_notebook_first_falls_back_on_missing_notebook(monkeypatch: pytest.MonkeyPatch) -> None:
    def fallback(value: str) -> str:
        return f"fallback:{value}"

    monkeypatch.setattr(_compat.notebook_loader, "should_dispatch_notebook", lambda _name: True)

    def _raise(_name: str) -> object:
        raise KeyError("missing")

    monkeypatch.setattr(_compat.notebook_loader, "get_notebook_function", _raise)
    assert _call_notebook_first("x", fallback, "value") == "fallback:value"


def test_encode_image_returns_base64_and_none_on_failure(tmp_path: Path) -> None:
    source = tmp_path / "img.bin"
    source.write_bytes(b"img-bytes")
    assert compat_callables.encode_image(source) == base64.b64encode(b"img-bytes").decode()
    assert compat_callables.encode_image(tmp_path / "missing.bin") is None


def test_llm_compat_functions_delegate_to_echo_adapter() -> None:
    prompt = "integration prompt"
    assert compat_callables.get_llm_response(prompt) == prompt
    assert compat_callables.get_llm_response2(prompt) == prompt
    assert compat_callables.get_llm_response_no_image(prompt) == prompt
    assert compat_callables.analyze_image_with_janus(prompt, "img") == prompt
    assert compat_callables.generate_response(prompt) == prompt
    assert compat_callables.setup_janus_model() == "janus-model-initialized"


def test_get_response_helpers_use_first_encoded_image(tmp_path: Path) -> None:
    image = tmp_path / "img.txt"
    image.write_text("x", encoding="utf-8")
    assert compat_callables.get_response_with_images("hello", [image]) == "hello"
    assert compat_callables.get_response_with_images("hello", []) == "hello"


def test_prepare_conversation_returns_copied_payload() -> None:
    history = [{"role": "user", "content": "hi"}]
    prepared = compat_callables.prepare_conversation(history)
    prepared[0]["content"] = "changed"
    assert history[0]["content"] == "hi"
