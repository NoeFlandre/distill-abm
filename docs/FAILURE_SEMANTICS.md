# Failure Semantics

## Summary generation policy

The pipeline has two source modes:

- `summary_only` (default): summarization is attempted using configured summarizers.
- `full_text_only`: summarization is intentionally skipped and raw trend text is used.

## Fallback behavior

- Default behavior is strict: `allow_summary_fallback = false`.
- In strict mode, if `summary_only` is selected and no configured summarizer returns a non-empty result, the run fails.
- In fallback mode (`allow_summary_fallback = true`), the run keeps going by using the raw full trend text.

## Traceability

`allow_summary_fallback` is persisted in run metadata under `inputs.allow_summary_fallback`.

## Reproducibility impact

- `inputs.allow_summary_fallback` is recorded for every run.
- If `allow_summary_fallback` is `false`, a failed summarization path is considered a full run failure, not just a degradation in score quality.
