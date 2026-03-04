from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

import distill_abm.compat as compat
import distill_abm.compat.reference_loader as notebook_loader


def test_compat_prefers_notebook_callable_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_notebook_function(name: str) -> Callable[..., Any]:
        assert name == "clean_context_response"
        return lambda text: f"from-notebook:{text}"

    monkeypatch.setattr(notebook_loader, "should_dispatch_notebook", lambda _name: True)
    monkeypatch.setattr(notebook_loader, "get_notebook_function", fake_get_notebook_function)
    assert compat.clean_context_response("hello") == "from-notebook:hello"


def test_compat_falls_back_to_refactored_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_missing(name: str) -> Callable[..., Any]:
        raise KeyError(name)

    monkeypatch.setattr(notebook_loader, "should_dispatch_notebook", lambda _name: True)
    monkeypatch.setattr(notebook_loader, "get_notebook_function", raise_missing)
    assert compat.clean_context_response("prefix </think> keep") == "keep"


def test_compat_falls_back_when_notebook_callable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_notebook_function(name: str) -> Callable[..., Any]:
        assert name == "clean_context_response"

        def _raise(_text: str) -> str:
            raise RuntimeError("runtime notebook failure")

        return _raise

    monkeypatch.setattr(notebook_loader, "should_dispatch_notebook", lambda _name: True)
    monkeypatch.setattr(notebook_loader, "get_notebook_function", fake_get_notebook_function)
    assert compat.clean_context_response("prefix </think> keep") == "keep"


def test_compat_dispatches_other_helpers_notebook_first(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_notebook_function(name: str) -> Callable[..., Any]:
        assert name == "clean_symbols"

        def _clean(value: str) -> str:
            return f"notebook::{value}"

        return _clean

    monkeypatch.setattr(notebook_loader, "should_dispatch_notebook", lambda _name: True)
    monkeypatch.setattr(notebook_loader, "get_notebook_function", fake_get_notebook_function)
    assert compat.clean_symbols("hello") == "notebook::hello"


def test_compat_skips_notebook_dispatch_when_function_not_required(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_if_called(_name: str) -> Callable[..., Any]:
        raise AssertionError("notebook loader should not be called")

    monkeypatch.setattr(notebook_loader, "get_notebook_function", raise_if_called)
    assert compat.clean_context_response("prefix </think> keep") == "keep"
