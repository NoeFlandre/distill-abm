# Run Observability Summary

- observed_row_count: `936`
- request_count: `95`
- reused_request_count: `841`
- request_counts_by_kind: `{"trend": 95}`
- request_counts_by_abm: `{"milk_consumption": 95}`
- providers: `["openrouter"]`
- models: `["moonshotai/kimi-k2.5"]`
- runtime_providers: `["AtlasCloud", "BaseTen", "Inceptron", "NextBit", "Novita", "Parasail", "Phala", "Together"]`
- runtime_precisions: `["fp4", "int4", "unknown"]`
- temperatures: `["1.0"]`
- max_tokens: `["32768"]`
- retry_settings: `{"max_retries": ["2"], "retry_backoff_seconds": ["2.0"]}`
- compression: `{"compression_tiers_used": ["0"], "request_count_with_compression": 0, "request_counts_by_tier": {"0": 95}, "table_downsample_strides_used": ["1"]}`
- usage_totals: `{"completion_tokens": 276869, "prompt_tokens": 236427, "total_tokens": 513296}`
- observability_coverage: `{"requests_with_runtime_precision": 95, "requests_with_runtime_provider": 95}`
- resumed_request_count: `95`
- run_observability_csv: `results/kimi-k2.5_all_abms_chain/04_generation_smoke_latest/abms/milk_consumption/runs/run_20260313_130242_029157/run_observability.csv`
