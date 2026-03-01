"""Lightweight summary scoring used by integration and regression tests."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel

from distill_abm.eval.legacy_scores import compute_scores


class SummaryScores(BaseModel):
    """Captures both lexical overlap and notebook legacy evaluation metrics."""

    token_f1: float
    precision: float
    recall: float
    bleu: float
    meteor: float
    rouge1: float
    rouge2: float
    rouge_l: float
    flesch_reading_ease: float
    reference_length: int
    candidate_length: int


def score_summary(reference: str, candidate: str) -> SummaryScores:
    """Computes notebook-compatible lexical metrics plus stable token overlap."""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    overlap = _overlap_count(ref_tokens, cand_tokens)
    precision = _safe_divide(overlap, len(cand_tokens))
    recall = _safe_divide(overlap, len(ref_tokens))
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    legacy = compute_scores(reference, candidate)
    return SummaryScores(
        token_f1=f1,
        precision=precision,
        recall=recall,
        bleu=legacy.bleu,
        meteor=legacy.meteor,
        rouge1=legacy.rouge1,
        rouge2=legacy.rouge2,
        rouge_l=legacy.rouge_l,
        flesch_reading_ease=legacy.flesch_reading_ease,
        reference_length=len(ref_tokens),
        candidate_length=len(cand_tokens),
    )


def _overlap_count(reference: list[str], candidate: list[str]) -> int:
    ref_counter = Counter(reference)
    cand_counter = Counter(candidate)
    return sum(min(ref_counter[token], cand_counter[token]) for token in ref_counter)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
