# Walkthrough

## 1. Install

```bash
uv sync --frozen --extra dev
```

## 2. Run a Benchmark-Policy Pipeline

```bash
uv run distill-abm run \
  --csv-path data/samples/sim.csv \
  --parameters-path data/samples/params.txt \
  --documentation-path data/samples/docs.txt \
  --model-id kimi_k2_5 \
  --evidence-mode plot+table \
  --text-source-mode summary_only \
  --summarizer bart --summarizer bert --summarizer t5 --summarizer longformer_ext
```

## 3. Run Full-Text Only Ablation

```bash
uv run distill-abm run \
  --csv-path data/samples/sim.csv \
  --parameters-path data/samples/params.txt \
  --documentation-path data/samples/docs.txt \
  --model-id gemini_3_1_pro_preview \
  --evidence-mode table \
  --text-source-mode full_text_only
```

## 4. Debug Smoke Matrix

```bash
uv run distill-abm smoke-qwen \
  --csv-path data/samples/sim.csv \
  --parameters-path data/samples/params.txt \
  --documentation-path data/samples/docs.txt \
  --allow-debug-model
```

## 5. DOE / ANOVA

```bash
uv run distill-abm analyze-doe --input-csv results/sweep/combinations_report.csv
```

## 6. Reproducibility Checklist
1. Preserve generated `pipeline_run_metadata.json` files.
2. Preserve command-line invocation and config snapshots.
3. Keep model registry and runtime defaults version-controlled.
4. Re-run with identical inputs to validate run-signature reuse behavior.
