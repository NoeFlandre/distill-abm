# Distill ABM Results Bucket

This folder is the artifacts-only results mirror for this project.

If you landed here for code, configuration, or run instructions, switch to the source repository:

https://github.com/NoeFlandre/distill-abm

## What is in this bucket

The mirror exposes final outputs, smoke-chain artifacts, and reproducible review tables produced by the
`distill-abm` pipeline. The bucket is synchronized from a local `results/` tree and is not where
runtime code is developed.

The maintained top-level structure is:

- `quantitative_master_overview/`: rolled-up latest quantitative tables (ANOVA, factorial, evidence, best-scores, prompt compression).
- `side_studies/`: focused methodological side experiments and ad-hoc investigations.
- `archive/`: frozen historical runs kept for reproducibility and audit.

**Screening Stage** (fast, low-cost exploration):

- `screening/kimi-k2.5_all_abms_chain/`: full single-LLM chain artifacts for Kimi runs.
- `screening/qwen3.5-27b_openrouter_all_abms_chain/`: full single-LLM chain artifacts for Qwen runs.
- `screening/eval_qwen_kimi/`: cross-model quantitative comparison for Qwen vs Kimi.

**Optimization Stage** (high-quality final runs):

- `optimisation/gemini-3.1-pro-preview_optimization_all_abms_chain/`: full single-LLM chain artifacts for Gemini runs.
- `optimisation/claude-opus-4.6_optimization_all_abms_chain/`: full single-LLM chain artifacts for Claude runs.

**Debug/Development** (not for benchmark):

- `debug/mistral-medium-latest_all_abms_chain/`: full single-LLM chain artifacts for Mistral runs.
- `debug/mistral-large-2512_optimization_all_abms_chain/`: optimization-chain artifacts for Mistral runs.
- `debug/eval_qwen_mistral/`: cross-model quantitative comparison for Qwen vs Mistral.
- `debug/eval_mistral_kimi/`: cross-model quantitative comparison for Mistral vs Kimi.
- `debug/eval_qwen_mistral_kimi/`: three-way quantitative comparison table and rows.

Each chain folder typically contains numbered smoke phases:

- `01_ingest_smoke_latest`
- `02_viz_smoke_latest`
- `03_doe_smoke_latest`
- `04_generation_smoke_latest`
- `05_summarizer_smoke_latest`
- `06_quantitative_smoke_latest`

Every phase keeps run metadata in its own `runs/` subfolders and a `latest_run.txt` pointer so the newest
run can be discovered quickly.

## Remote locations

- Bucket URI: `hf://buckets/NoeFlandre/distill-abms-results`
- Bucket web UI: `https://huggingface.co/buckets/NoeFlandre/distill-abms-results`

## Sync contract

Preferred mirror command:

```bash
uv run distill-abm sync-results-bucket
```

This command mirrors `./results` to the HF bucket and removes remote files that are no longer present locally by default.
It also ignores local macOS/cache noise (`.DS_Store`, `**/.DS_Store`, `.cache/**`, `**/.cache/**`).
For safety, apply-mode sync refuses to run when the local tree has no syncable result files after exclusions and `--delete` is still enabled.

Dry run + reviewable plan:

```bash
uv run distill-abm sync-results-bucket --dry-run --plan-path /tmp/distill_abm_results_sync_plan.jsonl
```

Download back to a local mirror:

```bash
hf sync hf://buckets/NoeFlandre/distill-abms-results ./results
```

## Maintenance Checklist

When updating this bucket later:

1. Keep code changes in Git and result changes in `results/`.
2. If you are unsure that the local mirror is complete, refresh it from the bucket first.
3. Run the dry run and inspect `/tmp/distill_abm_results_sync_plan.jsonl`.
4. Apply `uv run distill-abm sync-results-bucket` only after the dry run looks correct.
5. If the local tree is intentionally incomplete, use `--no-delete` or `--allow-empty-source` explicitly instead of relying on the default destructive mode.

To clean remote `.DS_Store` files and `.cache/**` clutter already in the bucket, build a targeted delete plan with:

```bash
hf sync /tmp/hf_bucket_cleanup_empty hf://buckets/NoeFlandre/distill-abms-results \
  --delete \
  --include '.DS_Store' \
  --include '**/.DS_Store' \
  --include '.cache/**' \
  --include '**/.cache/**' \
  --plan /tmp/distill_abm_bucket_cleanup_plan.jsonl
```

Then apply it with:

```bash
hf sync --apply /tmp/distill_abm_bucket_cleanup_plan.jsonl
```
