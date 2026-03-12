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

The bucket mirrors the four top-level folders found under local `results/`:

- `archive/`
- `mistral-medium-latest_all_abms_chain/`
- `qwen3.5-27b_openrouter_all_abms_chain/`
- `eval_qwen_mistral/`

Local-to-remote mapping:

- `results/archive/` -> `hf://buckets/NoeFlandre/distill-abms-results/archive/`
- `results/mistral-medium-latest_all_abms_chain/` -> `hf://buckets/NoeFlandre/distill-abms-results/mistral-medium-latest_all_abms_chain/`
- `results/qwen3.5-27b_openrouter_all_abms_chain/` -> `hf://buckets/NoeFlandre/distill-abms-results/qwen3.5-27b_openrouter_all_abms_chain/`
- `results/eval_qwen_mistral/` -> `hf://buckets/NoeFlandre/distill-abms-results/eval_qwen_mistral/`

## Sync Commands

Install the official CLI:

```bash
uv tool install hf
```

Authenticate:

```bash
hf auth login
```

Upload the current local results tree:

```bash
hf sync ./results hf://buckets/NoeFlandre/distill-abms-results
```

Equivalent bucket-native form:

```bash
hf buckets sync ./results hf://buckets/NoeFlandre/distill-abms-results
```

Download the mirrored tree back locally:

```bash
hf sync hf://buckets/NoeFlandre/distill-abms-results ./results
```
