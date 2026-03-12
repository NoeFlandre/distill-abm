# Run Observability Summary

- observed_row_count: `936`
- request_count: `866`
- reused_request_count: `70`
- request_counts_by_kind: `{"context": 2, "trend": 864}`
- request_counts_by_abm: `{"milk_consumption": 866}`
- providers: `["openrouter"]`
- models: `["qwen/qwen3.5-27b"]`
- runtime_providers: `["Alibaba", "AtlasCloud", "Novita"]`
- runtime_precisions: `["bf16", "fp8", "unknown"]`
- temperatures: `["1.0"]`
- max_tokens: `["32768"]`
- retry_settings: `{"max_retries": ["2"], "retry_backoff_seconds": ["2.0"]}`
- compression: `{"compression_tiers_used": ["0"], "request_count_with_compression": 0, "request_counts_by_tier": {"0": 864}, "table_downsample_strides_used": ["1"]}`
- usage_totals: `{"completion_tokens": 1410677, "prompt_tokens": 2000085, "total_tokens": 3410762}`
- observability_coverage: `{"requests_with_runtime_precision": 866, "requests_with_runtime_provider": 866}`
- resumed_request_count: `0`
- run_observability_csv: `results/qwen3.5-27b_openrouter_all_abms_chain/04_generation_smoke_latest/abms/milk_consumption/runs/run_20260312_111830_338495/run_observability.csv`
