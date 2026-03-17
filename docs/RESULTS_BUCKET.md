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
- `side_studies/`

**Screening Stage** (fast, low-cost exploration):

- `kimi-k2.5_all_abms_chain/`
- `qwen3.5-27b_openrouter_all_abms_chain/`

**Optimization Stage** (high-quality final runs):

- `gemini-3.1-pro-preview_optimization_all_abms_chain/`
- `claude-opus-4.6_optimization_all_abms_chain/`

**Debug/Development** (not for benchmark):

- `mistral-medium-latest_all_abms_chain/`
- `mistral-large-2512_optimization_all_abms_chain/`

**Cross-Model Evaluations**:

- `eval_qwen_kimi/`
- `eval_mistral_kimi/`
- `eval_qwen_mistral_kimi/`
- `eval_qwen_mistral/` (debug)

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

Apply-mode sync also refuses to run with `--delete` when the local tree has no syncable result files after exclusions. This prevents an almost-empty checkout from wiping the remote bucket by mistake.

Dry-run and save a reviewable sync plan:

```bash
uv run distill-abm sync-results-bucket --dry-run --plan-path /tmp/distill_abm_results_sync_plan.jsonl
```

The command mirrors `./results` to `hf://buckets/NoeFlandre/distill-abms-results`, deletes remote files missing locally by default, excludes hidden cache clutter, and reuses an existing HF login unless `HF_TOKEN` is set.

The main paper-facing entrypoint inside the bucket is `quantitative_master_overview/`, which collects the latest overview tables without requiring you to inspect each run directory manually.

## Maintenance Workflow

Use this sequence when you want to keep the bucket current without risking accidental deletion:

1. Make sure the code is committed in Git separately from the results.
2. Refresh the local results mirror if you are not certain your checkout is complete:

```bash
hf sync hf://buckets/NoeFlandre/distill-abms-results ./results
```

3. Run a dry run and save the plan:

```bash
uv run distill-abm sync-results-bucket --dry-run --plan-path /tmp/distill_abm_results_sync_plan.jsonl
```

4. Inspect the plan file before applying the sync.
5. Apply the sync only after the dry run looks correct:

```bash
uv run distill-abm sync-results-bucket
```

6. If the local tree is intentionally partial, prefer `--no-delete` or explicitly acknowledge the state with `--allow-empty-source`.

## Remote Cleanup

To remove remote macOS/cache clutter already present in the bucket, use a targeted empty-source sync plan instead of a broad manual delete.

Create an empty local directory:

```bash
mkdir -p /tmp/hf_bucket_cleanup_empty
```

Dry run the cleanup and save the delete plan:

```bash
hf sync /tmp/hf_bucket_cleanup_empty hf://buckets/NoeFlandre/distill-abms-results \
  --delete \
  --include '.DS_Store' \
  --include '**/.DS_Store' \
  --include '.cache/**' \
  --include '**/.cache/**' \
  --plan /tmp/distill_abm_bucket_cleanup_plan.jsonl
```

Apply the cleanup plan after inspection:

```bash
hf sync --apply /tmp/distill_abm_bucket_cleanup_plan.jsonl
```

Operational notes:

- The repository keeps `results/README.md` as a local pointer file; the sync guard does not count that file as result data.
- Apply-mode sync refuses to run with `--delete` when the local tree has no syncable result files after exclusions.
- Use the bucket as the durable store for outputs and Git as the durable store for code and documentation.

Equivalent raw CLI form:

```bash
hf sync ./results hf://buckets/NoeFlandre/distill-abms-results --delete
```

Download the mirrored tree back locally:

```bash
hf sync hf://buckets/NoeFlandre/distill-abms-results ./results
```
