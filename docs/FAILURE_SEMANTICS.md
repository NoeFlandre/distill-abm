# Failure Semantics

This document describes the pipeline's behavior when summarization fails and how it impacts reproducibility and run integrity.

## Summary Generation Modes

The pipeline (`distill-abm run`) supports two primary text source modes via the `--text-source-mode` flag:

- **`summary_only` (Default):** The pipeline attempts to summarize the generated trend narratives using the configured summarizers (e.g., BART, T5).
- **`full_text_only`:** Summarization is skipped, and the raw trend narratives are used for scoring and evaluation.

## Fallback Policy

When `--text-source-mode summary_only` is active, the pipeline's behavior depends on the `--allow-summary-fallback` (or `-f`) flag.

| Mode | `--allow-summary-fallback` | Outcome if all summarizers fail/return empty |
| :--- | :--- | :--- |
| `summary_only` | **`False` (Default)** | **CRITICAL FAILURE:** The run exits with code 1. No scores are computed. |
| `summary_only` | **`True`** | **DEGRADATION:** The run continues using the raw "full text" trend narrative. |
| `full_text_only` | *Ignored* | **SUCCESS:** Raw text is used intentionally. |

### Why use Strict Mode (Default)?
In benchmark settings, using the "full text" when a summary was requested can lead to "optimistic" lexical scores (since the full text is usually much longer and contains more overlap with the reference). Strict mode prevents this "silent degradation" from contaminating your experiment results.

## Traceability & Reproducibility

The fallback state is explicitly recorded in `pipeline_run_metadata.json` to ensure results can be audited:

```json
{
  "inputs": {
    "text_source_mode": "summary_only",
    "allow_summary_fallback": true
  },
  "reproducibility": {
    "trend_summary_present": false,
    "score_source": "full_text_fallback"
  }
}
```

- **`reproducibility.trend_summary_present`:** `true` if at least one summarizer succeeded.
- **`reproducibility.score_source`:** Indicates whether scores were derived from `summary` or `full_text_fallback`.

## Error Handling

If a summarizer dependency (like `transformers`) is missing, the pipeline raises a `SummarizationError`. 
In strict mode, this error propagates and terminates the run. In fallback mode, the error is logged as a warning, and the pipeline proceeds with the raw text.
