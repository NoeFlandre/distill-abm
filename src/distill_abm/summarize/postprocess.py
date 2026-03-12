"""Regex cleanup pipeline for output normalization."""

from __future__ import annotations

import csv
import re
import tempfile
import unicodedata
from pathlib import Path

import pandas as pd


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


def remove_repeated_sentences(text: str) -> str:
    """Collapse adjacent duplicate sentences while preserving non-adjacent repeats."""
    sentences = _split_sentences(text)
    if not sentences:
        return text.strip()

    deduped: list[str] = []
    previous_normalized: str | None = None
    for sentence in sentences:
        normalized = _normalize_sentence_for_comparison(sentence)
        if normalized and normalized == previous_normalized:
            continue
        deduped.append(sentence.strip())
        previous_normalized = normalized
    return " ".join(part for part in deduped if part).strip()


def remove_repeated_phrases(text: str) -> str:
    """Collapse obvious contiguous repeated tail loops inside sentences."""
    sentences = _split_sentences(text)
    if not sentences:
        return text.strip()
    cleaned = [_collapse_tail_loop(sentence) for sentence in sentences]
    return " ".join(part for part in cleaned if part).strip()


def capitalize_sentences(text: str) -> str:
    """Capitalizes sentence and paragraph starts after regex cleanups."""
    if not text:
        return text
    updated = text
    if not _starts_with_label_prefix(text):
        updated = text[0].upper() + text[1:]
    updated = re.sub(r"([.!?]\s+)([a-z])", _capitalize_match, updated)
    return re.sub(r"(\n\s*)([a-z])", _capitalize_match, updated)


def _capitalize_match(match: re.Match[str]) -> str:
    return match.group(1) + match.group(2).upper()


def clean_non_unicode(input_file: Path, output_file: Path) -> None:
    """Normalize Unicode to ASCII."""
    with input_file.open(mode="r", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        rows = [row for row in reader]
    with output_file.open(mode="w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        for row in rows:
            writer.writerow([_remove_non_unicode(cell) for cell in row])


def _remove_non_unicode(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", errors="ignore").decode("ascii")


def postprocess_summary(text: str) -> str:
    """Apply cleanup stages in deterministic order."""
    cleaned = remove_sentences_with_www(text)
    cleaned = remove_repeated_sentences(cleaned)
    cleaned = remove_repeated_phrases(cleaned)
    cleaned = remove_hyphens_after_punctuation(cleaned)
    cleaned = remove_unnecessary_punctuation(cleaned)
    cleaned = remove_unnecessary_spaces_in_parentheses(cleaned)
    cleaned = remove_space_before_dot(cleaned)
    return capitalize_sentences(cleaned)


def postprocess_csv_batch(
    input_csv: Path,
    output_csv: Path,
    bart_column: str = "Summary (BART) Reduced",
    bert_column: str = "Summary (BERT) Reduced",
) -> pd.DataFrame:
    """Run staged CSV postprocessing over BART/BERT summary columns."""
    frame = pd.read_csv(input_csv)

    # First pass applies `www` filtering to both summary columns.
    for column in [bart_column, bert_column]:
        if column in frame.columns:
            frame[column] = frame[column].apply(lambda value: remove_sentences_with_www(str(value)))

    # Then apply these passes to BERT only.
    if bert_column in frame.columns:
        frame[bert_column] = frame[bert_column].apply(
            lambda value: remove_hyphens_after_punctuation(value) if pd.notnull(value) else value
        )
        frame[bert_column] = frame[bert_column].apply(
            lambda value: remove_unnecessary_punctuation(value) if pd.notnull(value) else value
        )
        frame[bert_column] = frame[bert_column].apply(
            lambda value: remove_unnecessary_spaces_in_parentheses(value) if pd.notnull(value) else value
        )
        frame[bert_column] = frame[bert_column].apply(
            lambda value: capitalize_sentences(value) if pd.notnull(value) else value
        )

    # Apply space-before-dot cleanup to BART only.
    if bart_column in frame.columns:
        frame[bart_column] = frame[bart_column].apply(
            lambda value: remove_space_before_dot(value) if pd.notnull(value) else value
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        interim = Path(temp_dir) / "interim.csv"
        frame.to_csv(interim, index=False)
        clean_non_unicode(interim, output_csv)

    return pd.read_csv(output_csv)


def _split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]


def _normalize_sentence_for_comparison(sentence: str) -> str:
    return " ".join(sentence.split()).strip().casefold()


def _starts_with_label_prefix(text: str) -> bool:
    return re.match(r"^[A-Za-z0-9_-]+:{1,2}\S", text) is not None


def _collapse_tail_loop(sentence: str) -> str:
    trailing_punctuation = ""
    if sentence and sentence[-1] in ".!?":
        trailing_punctuation = sentence[-1]
        sentence = sentence[:-1]

    tokens = re.findall(r"[A-Za-z0-9']+", sentence)
    if len(tokens) < 9:
        return (sentence + trailing_punctuation).strip()

    lowered = [token.casefold() for token in tokens]
    best_candidate: list[str] | None = None
    best_start = len(tokens) + 1

    for start in range(len(tokens)):
        tail = lowered[start:]
        if len(tail) < 9:
            continue
        for unit_length in range(3, min(12, len(tail)) + 1):
            unit = tail[:unit_length]
            if not unit:
                continue
            if not _tail_matches_repeating_unit(tail, unit):
                continue
            full_repeats, remainder = divmod(len(tail), unit_length)
            if full_repeats < 3:
                continue
            keep_length = remainder if remainder else unit_length
            candidate = tokens[:start] + tokens[start : start + keep_length]
            if start < best_start or (start == best_start and best_candidate is not None and len(candidate) < len(best_candidate)):
                best_candidate = candidate
                best_start = start

    if best_candidate is None:
        return (sentence + trailing_punctuation).strip()
    collapsed = " ".join(best_candidate).strip()
    return f"{collapsed}{trailing_punctuation}".strip()


def _tail_matches_repeating_unit(tail: list[str], unit: list[str]) -> bool:
    if len(tail) < len(unit) * 3:
        return False
    for index, token in enumerate(tail):
        if token != unit[index % len(unit)]:
            return False
    return True
