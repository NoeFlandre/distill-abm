# Results Bucket

If you are looking for run outputs, quantitative tables, or smoke artifacts, start here. If you are looking for code, pipeline logic, tests, or documentation, go to the GitHub repository.

- Bucket URI: `hf://buckets/NoeFlandre/distill-abms-results`
- Bucket Web UI: `https://huggingface.co/buckets/NoeFlandre/distill-abms-results`
- Repository: `https://github.com/NoeFlandre/distill-abm`

## Purpose

The repository keeps a tracked `results/` tree for the most important validated artifacts. The Hugging Face bucket mirrors that tree so the results can also be copied, downloaded, and shared as a single storage surface without inventing a second layout.

Practical split:

- Use this bucket for results.
- Use GitHub for code, docs, and reproducibility logic.

## Mirrored Layout

The bucket mirrors the local `results/` tree. The currently maintained top-level folders are:

- `archive/`
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

Dry-run and save a reviewable sync plan:

```bash
uv run distill-abm sync-results-bucket --dry-run --plan-path /tmp/distill_abm_results_sync_plan.jsonl
```

The command mirrors `./results` to `hf://buckets/NoeFlandre/distill-abms-results`, deletes remote files missing locally by default, and reuses an existing HF login unless `HF_TOKEN` is set.

Equivalent raw CLI form:

```bash
hf sync ./results hf://buckets/NoeFlandre/distill-abms-results --delete
```

Download the mirrored tree back locally:

```bash
hf sync hf://buckets/NoeFlandre/distill-abms-results ./results
```
