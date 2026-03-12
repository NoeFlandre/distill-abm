# Run Observability Summary

- observed_row_count: `792`
- request_count: `570`
- reused_request_count: `222`
- request_counts_by_kind: `{"trend": 570}`
- request_counts_by_abm: `{"grazing": 570}`
- providers: `["openrouter"]`
- models: `["qwen/qwen3.5-27b"]`
- runtime_providers: `["Alibaba", "AtlasCloud", "Novita"]`
- runtime_precisions: `["bf16", "fp8", "unknown"]`
- temperatures: `["1.0"]`
- max_tokens: `["32768"]`
- retry_settings: `{"max_retries": ["2"], "retry_backoff_seconds": ["2.0"]}`
- compression: `{"compression_tiers_used": ["0"], "request_count_with_compression": 0, "request_counts_by_tier": {"0": 570}, "table_downsample_strides_used": ["1"]}`
- usage_totals: `{"completion_tokens": 982222, "prompt_tokens": 1526620, "total_tokens": 2508842}`
- observability_coverage: `{"requests_with_runtime_precision": 570, "requests_with_runtime_provider": 570}`
- resumed_request_count: `570`
- run_observability_csv: `results/chains/qwen3.5-27b_openrouter_all_abms_chain/04_generation_smoke_latest/abms/grazing/runs/run_20260312_105437_842508/run_observability.csv`
