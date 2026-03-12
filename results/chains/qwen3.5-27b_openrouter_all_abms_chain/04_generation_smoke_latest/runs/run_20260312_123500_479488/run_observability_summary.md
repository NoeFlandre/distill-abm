# Run Observability Summary

- observed_row_count: `2808`
- request_count: `2431`
- reused_request_count: `377`
- request_counts_by_kind: `{"context": 3, "trend": 2428}`
- request_counts_by_abm: `{"fauna": 995, "grazing": 570, "milk_consumption": 866}`
- providers: `["openrouter"]`
- models: `["qwen/qwen3.5-27b"]`
- runtime_providers: `["Alibaba", "AtlasCloud", "Novita"]`
- runtime_precisions: `["bf16", "fp8", "unknown"]`
- temperatures: `["1.0"]`
- max_tokens: `["32768"]`
- retry_settings: `{"max_retries": ["2"], "retry_backoff_seconds": ["2.0"]}`
- compression: `{"compression_tiers_used": ["0"], "request_count_with_compression": 0, "request_counts_by_tier": {"0": 2428}, "table_downsample_strides_used": ["1"]}`
- usage_totals: `{"completion_tokens": 4261300, "prompt_tokens": 6507248, "total_tokens": 10768548}`
- observability_coverage: `{"requests_with_runtime_precision": 1436, "requests_with_runtime_provider": 2431}`
- resumed_request_count: `570`
- run_observability_csv: `results/chains/qwen3.5-27b_openrouter_all_abms_chain/04_generation_smoke_latest/runs/run_20260312_123500_479488/run_observability.csv`
