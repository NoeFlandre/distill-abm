# Results and synchronization

The Git repository is publication-facing source code. The Hugging Face bucket is the durable store for generated results and review artifacts.

| Resource | Location |
| --- | --- |
| Bucket URI | `hf://buckets/NoeFlandre/distill-abms-results` |
| Bucket web UI | [huggingface.co/buckets/NoeFlandre/distill-abms-results](https://huggingface.co/buckets/NoeFlandre/distill-abms-results) |
| Repository | [github.com/NoeFlandre/distill-abm](https://github.com/NoeFlandre/distill-abm) |

## Layout

The maintained top-level result families are:

- `quantitative_master_overview/` — current overview tables and paper-facing summaries;
- `screening/` — fast, lower-cost exploration runs;
- `optimisation/` — higher-quality optimization runs;
- `side_studies/` — focused methodological investigations;
- `archive/` — frozen historical and validation artifacts;
- `debug/` — development-only runs and comparisons.

Chain directories commonly contain numbered phases for ingestion, visualization, DOE preparation, generation, summarization, and quantitative analysis. Case-based workflows keep run-separated folders, manifests, reports, and `latest_run.txt` pointers.

## Download results

Install and authenticate the official Hugging Face CLI, then mirror the bucket into the ignored local `results/` directory:

```bash
hf auth login
hf sync hf://buckets/NoeFlandre/distill-abms-results ./results
```

Start by inspecting `results/quantitative_master_overview/` rather than opening every run directory.

## Upload results safely

The project CLI mirrors the local `results/` tree and deletes remote files missing locally by default. Treat an apply sync as a destructive remote operation.

First create and inspect a dry-run plan:

```bash
uv run distill-abm sync-results-bucket --dry-run
```

To save the plan for review:

```bash
uv run distill-abm sync-results-bucket \
  --dry-run \
  --plan-path /tmp/distill_abm_results_sync_plan.jsonl
```

Then apply the reviewed state:

```bash
uv run distill-abm sync-results-bucket
```

The command excludes `.DS_Store`, cache directories, and similar local clutter. It refuses an apply-mode delete sync when no syncable result files remain after exclusions unless `--allow-empty-source` is explicit. Use `--no-delete` when the local tree is intentionally partial.

## Targeted remote cleanup

For remote macOS/cache clutter, make the deletion plan explicit and inspect it before applying:

```bash
mkdir -p /tmp/hf_bucket_cleanup_empty
hf sync /tmp/hf_bucket_cleanup_empty hf://buckets/NoeFlandre/distill-abms-results \
  --delete \
  --include '.DS_Store' \
  --include '**/.DS_Store' \
  --include '.cache/**' \
  --include '**/.cache/**' \
  --plan /tmp/distill_abm_bucket_cleanup_plan.jsonl
```

After review:

```bash
hf sync --apply /tmp/distill_abm_bucket_cleanup_plan.jsonl
```

Keep code changes in Git and result changes in the bucket. Do not use a partial local checkout with default deletion enabled unless you have intentionally reviewed the complete plan.
