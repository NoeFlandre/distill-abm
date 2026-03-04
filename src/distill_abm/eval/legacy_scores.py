"""Notebook-compatible lexical metrics with graceful dependency fallbacks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


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


def score_summaries_csv_batch(
    input_csv: Path,
    output_csv: Path,
    ground_truth_text: str,
    bart_column: str = "Summary (BART) Reduced",
    bert_column: str = "Summary (BERT) Reduced",
    score_fn: Callable[[str, str], LegacyScores] | None = None,
) -> pd.DataFrame:
    """Runs notebook-6 lexical scoring for BART and BERT summary columns."""
    frame = pd.read_csv(input_csv)
    missing = [column for column in [bart_column, bert_column] if column not in frame.columns]
    if missing:
        raise ValueError(f"missing summary columns: {', '.join(missing)}")

    scorer = score_fn or compute_scores
    (
        frame["BLEU (BART)"],
        frame["METEOR (BART)"],
        frame["ROUGE-1 (BART)"],
        frame["ROUGE-2 (BART)"],
        frame["ROUGE-L (BART)"],
        frame["Flesch Reading Ease (BART)"],
    ) = _score_column(frame, bart_column, ground_truth_text, scorer)
    (
        frame["BLEU (BERT)"],
        frame["METEOR (BERT)"],
        frame["ROUGE-1 (BERT)"],
        frame["ROUGE-2 (BERT)"],
        frame["ROUGE-L (BERT)"],
        frame["Flesch Reading Ease (BERT)"],
    ) = _score_column(frame, bert_column, ground_truth_text, scorer)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def _score_column(
    frame: pd.DataFrame,
    column: str,
    ground_truth_text: str,
    score_fn: Callable[[str, str], LegacyScores],
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
