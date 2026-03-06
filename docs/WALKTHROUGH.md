# Walkthrough

## 1. Install

```bash
uv sync --frozen --extra dev
```

## 2. Establish a Baseline

```bash
uv run pytest
uv run ruff check src tests
uv run mypy src tests
```

## 3. Run Pre-LLM Verification First

```bash
uv run distill-abm smoke-ingest-netlogo --models-root data --output-root results/ingest_smoke_latest
uv run distill-abm smoke-viz --models-root data --netlogo-home /path/to/NetLogo --output-root results/viz_smoke_latest
uv run distill-abm smoke-doe --ingest-root results/ingest_smoke_latest --viz-root results/viz_smoke_latest --output-root results/doe_smoke_latest
```

These commands do not execute any benchmark LLM request. They exist to validate the ingest, visualization, and exact pre-LLM DOE payloads before a full run.

## 4. Run a Benchmark-Policy Pipeline

```bash
uv run distill-abm run \
  --csv-path data/samples/sim.csv \
  --parameters-path data/samples/params.txt \
  --documentation-path data/samples/docs.txt \
  --model-id kimi_k2_5 \
  --evidence-mode plot+table \
  --text-source-mode summary_only \
  --allow-summary-fallback \
  --summarizer bart --summarizer bert --summarizer t5 --summarizer longformer_ext
```

`--allow-summary-fallback` is optional. By default, summary-only runs are strict: if every selected summarizer returns no output, the command fails.

See [Failure Semantics](FAILURE_SEMANTICS.md) for the full policy.

## 5. Run Full-Text Only Ablation

```bash
uv run distill-abm run \
  --csv-path data/samples/sim.csv \
  --parameters-path data/samples/params.txt \
  --documentation-path data/samples/docs.txt \
  --model-id gemini_3_1_pro_preview \
  --evidence-mode table \
  --text-source-mode full_text_only
```

## 6. Debug Smoke Matrix

```bash
uv run distill-abm smoke-qwen \
  --csv-path data/samples/sim.csv \
  --parameters-path data/samples/params.txt \
  --documentation-path data/samples/docs.txt \
  --allow-debug-model
```

## 7. DOE / ANOVA

```bash
uv run distill-abm analyze-doe --input-csv results/sweep/combinations_report.csv
```

## 8. Reproducibility Checklist
1. Preserve generated `pipeline_run_metadata.json` files.
2. Preserve command-line invocation and config snapshots.
3. Keep model registry and runtime defaults version-controlled.
4. Re-run with identical inputs to validate run-signature reuse behavior.
