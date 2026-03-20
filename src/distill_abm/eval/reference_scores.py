"""Lexical metric helpers used by quantitative scoring.

The metric path intentionally matches the legacy scoring notebooks:
- tokenization via ``str.split()``
- BLEU with NLTK smoothing method4
- METEOR with tokenized inputs
- ROUGE-1/2/L with stemming
- Flesch Reading Ease on the candidate summary text

If any dependency or required NLTK resource is unavailable, scoring must fail
explicitly rather than silently substituting fallback values.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ReferenceScores:
    """Typed container for lexical metric outputs."""

    bleu: float
    meteor: float
    rouge1: float
    rouge2: float
    rouge_l: float
    flesch_reading_ease: float


def compute_scores(ground_truth: str, summary: str) -> ReferenceScores:
    """Compute BLEU/METEOR/ROUGE/Flesch scores."""
    try:
        return _compute_with_external_metrics(ground_truth, summary)
    except (ImportError, LookupError) as exc:
        raise RuntimeError(
            "lexical metrics unavailable: install required metric dependencies "
            "and NLTK resources instead of using fallback scores"
        ) from exc


def _compute_with_external_metrics(ground_truth: str, summary: str) -> ReferenceScores:
    import textstat
    from nltk.translate import bleu_score as nltk_bleu_score
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    gt_tokens = ground_truth.split()
    summary_tokens = summary.split()
    smoothing_function = nltk_bleu_score.SmoothingFunction().method4
    bleu = _compute_bleu_score(
        gt_tokens=gt_tokens,
        summary_tokens=summary_tokens,
        sentence_bleu=nltk_bleu_score.sentence_bleu,
        smoothing_function=smoothing_function,
        bleu_module=nltk_bleu_score,
    )
    meteor = meteor_score([gt_tokens], summary_tokens)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge = scorer.score(ground_truth, summary)
    flesch = textstat.flesch_reading_ease(summary)
    return ReferenceScores(
        bleu=bleu,
        meteor=meteor,
        rouge1=rouge["rouge1"].fmeasure,
        rouge2=rouge["rouge2"].fmeasure,
        rouge_l=rouge["rougeL"].fmeasure,
        flesch_reading_ease=flesch,
    )


def _compute_bleu_score(
    gt_tokens: list[str],
    summary_tokens: list[str],
    sentence_bleu: Callable[..., float],
    smoothing_function: Callable[..., object],
    bleu_module: Any,
) -> float:
    try:
        return sentence_bleu([gt_tokens], summary_tokens, smoothing_function=smoothing_function)
    except TypeError as exc:
        # NLTK BLEU can fail on newer Python fractions implementations because it
        # still passes a private `_normalize` argument. Patch only that known case.
        if "_normalize" not in str(exc):
            raise

    original_fraction = getattr(bleu_module, "Fraction", None)

    def _compat_fraction(
        numerator: int = 0,
        denominator: int | None = None,
        _normalize: bool = True,
    ) -> Fraction:
        _ = _normalize
        if denominator is None:
            return Fraction(numerator)
        return Fraction(numerator, denominator)

    bleu_module.Fraction = _compat_fraction
    try:
        return sentence_bleu([gt_tokens], summary_tokens, smoothing_function=smoothing_function)
    finally:
        if original_fraction is not None:
            bleu_module.Fraction = original_fraction


def _compute_fallback_scores(ground_truth: str, summary: str) -> ReferenceScores:
    gt_tokens = ground_truth.lower().split()
    sum_tokens = summary.lower().split()
    overlap = len(set(gt_tokens).intersection(sum_tokens))
    precision = _safe_divide(overlap, len(sum_tokens))
    recall = _safe_divide(overlap, len(gt_tokens))
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    flesch = max(0.0, 100.0 - len(summary.split()) * 0.8)
    return ReferenceScores(
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


def score_summaries_csv_batch(
    input_csv: Path,
    output_csv: Path,
    ground_truth_text: str,
    bart_column: str = "Summary (BART) Reduced",
    bert_column: str = "Summary (BERT) Reduced",
    score_fn: Callable[[str, str], ReferenceScores] | None = None,
) -> pd.DataFrame:
    """Run lexical scoring for BART and BERT summary columns."""
    return score_summary_columns_csv_batch(
        input_csv=input_csv,
        output_csv=output_csv,
        ground_truth_text=ground_truth_text,
        summary_columns={"BART": bart_column, "BERT": bert_column},
        score_fn=score_fn,
    )


def score_summary_columns_csv_batch(
    input_csv: Path,
    output_csv: Path,
    ground_truth_text: str,
    summary_columns: Mapping[str, str],
    score_fn: Callable[[str, str], ReferenceScores] | None = None,
) -> pd.DataFrame:
    """Run notebook-equivalent lexical scoring for an arbitrary set of summary columns."""
    frame = pd.read_csv(input_csv)
    missing = [column for column in summary_columns.values() if column not in frame.columns]
    if missing:
        raise ValueError(f"missing summary columns: {', '.join(missing)}")

    scorer = score_fn or compute_scores
    for label, column in summary_columns.items():
        (
            frame[f"BLEU ({label})"],
            frame[f"METEOR ({label})"],
            frame[f"ROUGE-1 ({label})"],
            frame[f"ROUGE-2 ({label})"],
            frame[f"ROUGE-L ({label})"],
            frame[f"Flesch Reading Ease ({label})"],
        ) = _score_column(frame, column, ground_truth_text, scorer)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def _score_column(
    frame: pd.DataFrame,
    column: str,
    ground_truth_text: str,
    score_fn: Callable[[str, str], ReferenceScores],
) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
    bleu_scores: list[float] = []
    meteor_scores: list[float] = []
    rouge1_scores: list[float] = []
    rouge2_scores: list[float] = []
    rouge_l_scores: list[float] = []
    flesch_scores: list[float] = []
    for summary in frame[column]:
        scores = score_fn(ground_truth_text, str(summary))
        bleu_scores.append(scores.bleu)
        meteor_scores.append(scores.meteor)
        rouge1_scores.append(scores.rouge1)
        rouge2_scores.append(scores.rouge2)
        rouge_l_scores.append(scores.rouge_l)
        flesch_scores.append(scores.flesch_reading_ease)
    return (
        bleu_scores,
        meteor_scores,
        rouge1_scores,
        rouge2_scores,
        rouge_l_scores,
        flesch_scores,
    )
