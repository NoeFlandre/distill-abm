# Run Observability Summary

- observed_row_count: `2808`
- request_count: `160`
- reused_request_count: `2648`
- request_counts_by_kind: `{"trend": 160}`
- request_counts_by_abm: `{"fauna": 64, "grazing": 1, "milk_consumption": 95}`
- providers: `["openrouter"]`
- models: `["moonshotai/kimi-k2.5"]`
- runtime_providers: `["AtlasCloud", "BaseTen", "Inceptron", "NextBit", "Novita", "Parasail", "Phala", "Together"]`
- runtime_precisions: `["fp4", "int4", "unknown"]`
- temperatures: `["1.0"]`
- max_tokens: `["32768"]`
- retry_settings: `{"max_retries": ["2"], "retry_backoff_seconds": ["2.0"]}`
- compression: `{"compression_tiers_used": ["0"], "request_count_with_compression": 0, "request_counts_by_tier": {"0": 160}, "table_downsample_strides_used": ["1"]}`
- usage_totals: `{"completion_tokens": 469346, "prompt_tokens": 453520, "total_tokens": 922866}`
- observability_coverage: `{"requests_with_runtime_precision": 160, "requests_with_runtime_provider": 160}`
- resumed_request_count: `160`
- run_observability_csv: `results/kimi-k2.5_all_abms_chain/04_generation_smoke_latest/runs/run_20260313_130231_449387/run_observability.csv`
