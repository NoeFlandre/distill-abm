"""Compatibility utilities for parity-sensitive call paths."""

from __future__ import annotations

from pathlib import Path

from distill_abm.compat.compat import *  # noqa: F403
from distill_abm.compat.compat import (
    encode_image,
    extract_text_from_pdf,
    get_llm_response,
    summarize_with_bart,
    summarize_with_bert,
)

_WRAPPER_OVERRIDES = {
    "get_response_with_images",
    "get_response_with_pdf_and_images",
    "summarize_text_with_models",
}


def get_response_with_images(question: str, image_paths: list[str | Path]) -> str:  # type: ignore[no-redef]
    """Call the LLM wrapper after encoding image attachments."""
    images = [encode_image(path) for path in image_paths]
    image = images[0] if images else None
    return get_llm_response(question, image)


def get_response_with_pdf_and_images(question: str, pdf_path: str | Path, image_paths: list[str | Path]) -> str:  # type: ignore[no-redef]
    """Call the LLM wrapper with the PDF text and encoded images."""
    text = extract_text_from_pdf(pdf_path)
    return get_response_with_images(f"{question}\n{text}", image_paths)


def summarize_text_with_models(text: str) -> dict[str, str]:  # type: ignore[no-redef]
    """Generate parity-compatible summary outputs for BART and BERT."""
    return {"bart": summarize_with_bart(text), "bert": summarize_with_bert(text)}


__all__ = [name for name in globals() if not name.startswith("_")]
