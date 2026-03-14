# Run Observability Summary

- observed_row_count: `1080`
- request_count: `64`
- reused_request_count: `1016`
- request_counts_by_kind: `{"trend": 64}`
- request_counts_by_abm: `{"fauna": 64}`
- providers: `["openrouter"]`
- models: `["moonshotai/kimi-k2.5"]`
- runtime_providers: `["AtlasCloud", "BaseTen", "Inceptron", "NextBit", "Novita", "Parasail", "Phala", "Together"]`
- runtime_precisions: `["fp4", "int4", "unknown"]`
- temperatures: `["1.0"]`
- max_tokens: `["32768"]`
- retry_settings: `{"max_retries": ["2"], "retry_backoff_seconds": ["2.0"]}`
- compression: `{"compression_tiers_used": ["0"], "request_count_with_compression": 0, "request_counts_by_tier": {"0": 64}, "table_downsample_strides_used": ["1"]}`
- usage_totals: `{"completion_tokens": 190904, "prompt_tokens": 213494, "total_tokens": 404398}`
- observability_coverage: `{"requests_with_runtime_precision": 64, "requests_with_runtime_provider": 64}`
- resumed_request_count: `64`
- run_observability_csv: `results/kimi-k2.5_all_abms_chain/04_generation_smoke_latest/abms/fauna/runs/run_20260313_071158_572112/run_observability.csv`
