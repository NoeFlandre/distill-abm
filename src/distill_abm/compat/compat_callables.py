"""Compatibility helpers for reference-dispatch and LLM call plumbing."""

from __future__ import annotations

import base64
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import distill_abm.compat.reference_loader as notebook_loader
from distill_abm.llm.adapters.base import LLMMessage, LLMRequest
from distill_abm.llm.adapters.echo_adapter import EchoAdapter


def _call_notebook_first(name: str, fallback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    if not notebook_loader.should_dispatch_notebook(name):
        return cast(Any, fallback(*args, **kwargs))
    try:
        notebook_callable = notebook_loader.get_notebook_function(name)
    except KeyError:
        return cast(Any, fallback(*args, **kwargs))
    try:
        return cast(Any, notebook_callable(*args, **kwargs))
    except Exception:
        return cast(Any, fallback(*args, **kwargs))


def encode_image(image_path: str | Path) -> str | None:
    try:
        return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    except Exception:
        return None


def get_llm_response(prompt: str, base64_image: str | None = None) -> str:
    adapter = EchoAdapter(model="compat-echo")
    req = LLMRequest(model="compat-echo", messages=[LLMMessage(role="user", content=prompt)], image_b64=base64_image)
    return adapter.complete(req).text


def get_llm_response2(prompt: str) -> str:
    return get_llm_response(prompt)


def get_llm_response_no_image(prompt: str) -> str:
    return get_llm_response(prompt)


def analyze_image_with_janus(prompt: str, base64_image: str) -> str:
    return get_llm_response(prompt, base64_image)


def setup_janus_model() -> str:
    """Compatibility stub for Janus model initialization."""
    return "janus-model-initialized"


def get_response_with_images(question: str, image_paths: list[str | Path]) -> str:
    encoded = [encode_image(path) for path in image_paths]
    images = [item for item in encoded if item]
    return get_llm_response(question, images[0] if images else None)


def get_response_with_pdf_and_images(question: str, pdf_path: str | Path, image_paths: list[str | Path]) -> str:
    text = extract_text_from_pdf(pdf_path)
    return get_response_with_images(f"{question}\n{text}", image_paths)


def prepare_conversation(history: list[dict[str, str]]) -> list[dict[str, str]]:
    return [dict(item) for item in history]


def generate_response(prompt: str) -> str:
    return get_llm_response(prompt)


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    try:
        import pypdf
    except Exception:
        return ""
    reader = pypdf.PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)
