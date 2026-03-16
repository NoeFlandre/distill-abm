# Results Bucket

If you are looking for run outputs, quantitative tables, or smoke artifacts, start here. If you are looking for code, pipeline logic, tests, or documentation, go to the GitHub repository.

- Bucket URI: `hf://buckets/NoeFlandre/distill-abms-results`
- Bucket Web UI: `https://huggingface.co/buckets/NoeFlandre/distill-abms-results`
- Repository: `https://github.com/NoeFlandre/distill-abm`

## Purpose

The Git repository is publication-facing source code. The Hugging Face bucket is the publication-facing results store.

Practical split:

- Use this bucket for results.
- Use GitHub for code, docs, and reproducibility logic.

## Mirrored Layout

The bucket mirrors the local `results/` tree. The currently maintained top-level folders are:

- `archive/`
- `quantitative_master_overview/`
- `kimi-k2.5_all_abms_chain/`
- `mistral-medium-latest_all_abms_chain/`
- `qwen3.5-27b_openrouter_all_abms_chain/`
- `eval_qwen_mistral/`
- `eval_qwen_kimi/`
- `eval_mistral_kimi/`
- `eval_qwen_mistral_kimi/`

## Sync Commands

Install a recent official CLI:

```bash
python -m pip install --upgrade "huggingface_hub[hf_xet]"
```

Authenticate:

```bash
hf auth login
```

Preferred one-command repo workflow:

```bash
uv run distill-abm sync-results-bucket
```

The command excludes local macOS and cache clutter by default, currently:

- `.DS_Store`
- `**/.DS_Store`
- `.cache/**`
- `**/.cache/**`

Dry-run and save a reviewable sync plan:

```bash
uv run distill-abm sync-results-bucket --dry-run --plan-path /tmp/distill_abm_results_sync_plan.jsonl
```

The command mirrors `./results` to `hf://buckets/NoeFlandre/distill-abms-results`, deletes remote files missing locally by default, excludes hidden cache clutter, and reuses an existing HF login unless `HF_TOKEN` is set.

The main paper-facing entrypoint inside the bucket is `quantitative_master_overview/`, which collects the latest overview tables without requiring you to inspect each run directory manually.

Equivalent raw CLI form:

```bash
hf sync ./results hf://buckets/NoeFlandre/distill-abms-results --delete
```

Download the mirrored tree back locally:

```bash
hf sync hf://buckets/NoeFlandre/distill-abms-results ./results
```
