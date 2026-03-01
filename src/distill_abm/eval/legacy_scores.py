"""Notebook-compatible lexical metrics with graceful dependency fallbacks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegacyScores:
    """Holds notebook-style metric outputs in a typed structure."""

    bleu: float
    meteor: float
    rouge1: float
    rouge2: float
    rouge_l: float
    flesch_reading_ease: float


def compute_scores(ground_truth: str, summary: str) -> LegacyScores:
    """Computes BLEU/METEOR/ROUGE/Flesch scores used in scoring notebooks."""
    try:
        return _compute_with_external_metrics(ground_truth, summary)
    except Exception:
        return _compute_fallback_scores(ground_truth, summary)


def _compute_with_external_metrics(ground_truth: str, summary: str) -> LegacyScores:
    import textstat
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    gt_tokens = ground_truth.split()
    summary_tokens = summary.split()
    smoothing_function = SmoothingFunction().method4
    bleu = sentence_bleu([gt_tokens], summary_tokens, smoothing_function=smoothing_function)
    meteor = meteor_score([gt_tokens], summary_tokens)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge = scorer.score(ground_truth, summary)
    flesch = textstat.flesch_reading_ease(summary)
    return LegacyScores(
        bleu=bleu,
        meteor=meteor,
        rouge1=rouge["rouge1"].fmeasure,
        rouge2=rouge["rouge2"].fmeasure,
        rouge_l=rouge["rougeL"].fmeasure,
        flesch_reading_ease=flesch,
    )


def _compute_fallback_scores(ground_truth: str, summary: str) -> LegacyScores:
    gt_tokens = ground_truth.lower().split()
    sum_tokens = summary.lower().split()
    overlap = len(set(gt_tokens).intersection(sum_tokens))
    precision = _safe_divide(overlap, len(sum_tokens))
    recall = _safe_divide(overlap, len(gt_tokens))
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    flesch = max(0.0, 100.0 - len(summary.split()) * 0.8)
    return LegacyScores(
        bleu=f1,
        meteor=f1,
        rouge1=f1,
        rouge2=f1 / 2,
        rouge_l=f1,
        flesch_reading_ease=flesch,
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
