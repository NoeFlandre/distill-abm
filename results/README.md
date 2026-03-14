# Distill ABM Results

This tree mirrors the maintained results bucket contents.

Top-level folders:

- `quantitative_master_overview/`: aggregated latest quantitative overview tables.
- `kimi-k2.5_all_abms_chain/`: single-LLM run chain and latest quantitative outputs for Kimi.
- `mistral-medium-latest_all_abms_chain/`: single-LLM run chain and latest quantitative outputs for Mistral.
- `qwen3.5-27b_openrouter_all_abms_chain/`: single-LLM run chain and latest quantitative outputs for Qwen.
- `eval_qwen_mistral/`, `eval_qwen_kimi/`, `eval_mistral_kimi/`, `eval_qwen_mistral_kimi/`: multi-LLM quantitative comparisons.
- `archive/`: older archived artifacts kept for reproducibility.

The preferred sync command is:

```bash
uv run distill-abm sync-results-bucket
```

That command mirrors this tree to `hf://buckets/NoeFlandre/distill-abms-results`, deletes remote files missing locally by default, and excludes hidden cache clutter such as `.DS_Store` and `.cache/**`.
