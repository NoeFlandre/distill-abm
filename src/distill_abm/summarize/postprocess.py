"""Regex cleanup pipeline equivalent to notebook postprocessing steps."""

from __future__ import annotations

import csv
import re
from pathlib import Path


def remove_sentences_with_www(text: str) -> str:
    """Drops sentences containing `www.` links that leak into generated outputs."""
    paragraphs = text.split("\n")
    cleaned: list[str] = []
    for paragraph in paragraphs:
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", paragraph)
        filtered = [sentence for sentence in sentences if "www." not in sentence]
        cleaned.append(" ".join(filtered).strip())
    return "\n".join(cleaned).strip()


def remove_hyphens_after_punctuation(text: str) -> str:
    """Fixes artifacts like `. -` introduced by summarizer chunk boundaries."""
    return re.sub(r"([.,!?;:]\s*)-\s*", r"\1", text)


def remove_unnecessary_punctuation(text: str) -> str:
    """Removes punctuation collisions at sentence boundaries."""
    return re.sub(r"([.!?]\s*)[,;:]\s*", r"\1", text)


def remove_unnecessary_spaces_in_parentheses(text: str) -> str:
    """Compacts spacing inside parentheses for publication-ready prose."""
    compact = re.sub(r"\(\s*", "(", text)
    return re.sub(r"\s*\)", ")", compact)


def remove_space_before_dot(text: str) -> str:
    """Deletes spaces before periods left by tokenizer detokenization."""
    return re.sub(r"\s+\.", ".", text)


def capitalize_sentences(text: str) -> str:
    """Capitalizes sentence and paragraph starts after regex cleanups."""
    if not text:
        return text
    updated = text[0].upper() + text[1:]
    updated = re.sub(r"([.!?]\s+)([a-z])", _capitalize_match, updated)
    return re.sub(r"(\n\s*)([a-z])", _capitalize_match, updated)


def _capitalize_match(match: re.Match[str]) -> str:
    return match.group(1) + match.group(2).upper()


def clean_non_unicode(input_file: Path, output_file: Path) -> None:
    """Normalizes Unicode to ASCII exactly like notebook unidecode cleanup."""
    with input_file.open(mode="r", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        rows = [row for row in reader]
    with output_file.open(mode="w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        for row in rows:
            writer.writerow([_remove_non_unicode(cell) for cell in row])


def _remove_non_unicode(text: str) -> str:
    try:
        from unidecode import unidecode
    except Exception:
        return text.encode("ascii", errors="ignore").decode("ascii")
    return str(unidecode(text))


def postprocess_summary(text: str) -> str:
    """Applies notebook cleanup stages in the same order used in practice."""
    cleaned = remove_sentences_with_www(text)
    cleaned = remove_hyphens_after_punctuation(cleaned)
    cleaned = remove_unnecessary_punctuation(cleaned)
    cleaned = remove_unnecessary_spaces_in_parentheses(cleaned)
    cleaned = remove_space_before_dot(cleaned)
    return capitalize_sentences(cleaned)
