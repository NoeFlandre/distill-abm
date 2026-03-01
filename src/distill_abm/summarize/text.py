"""Text cleanup helpers extracted from notebook post-processing cells."""

from __future__ import annotations


def strip_think_prefix(text: str) -> str:
    """Drops chain-of-thought wrappers so outputs remain publication-ready."""
    return text.split("</think>")[-1].strip()


def clean_markdown_symbols(text: str) -> str:
    """Removes markdown artifacts introduced by some LLMs in CSV exports."""
    return text.replace("#", "").replace("*", "").strip()


def chunk_text(text: str, max_chars: int = 1024) -> list[str]:
    """Chunks long strings for summarizers with constrained context windows."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]
