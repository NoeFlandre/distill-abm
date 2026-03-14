# Run Observability Summary

- observed_row_count: `1872`
- request_count: `65`
- reused_request_count: `1807`
- request_counts_by_kind: `{"trend": 65}`
- request_counts_by_abm: `{"fauna": 64, "grazing": 1}`
- providers: `["openrouter"]`
- models: `["moonshotai/kimi-k2.5"]`
- runtime_providers: `["AtlasCloud", "BaseTen", "Inceptron", "NextBit", "Novita", "Parasail", "Phala", "Together"]`
- runtime_precisions: `["fp4", "int4", "unknown"]`
- temperatures: `["1.0"]`
- max_tokens: `["32768"]`
- retry_settings: `{"max_retries": ["2"], "retry_backoff_seconds": ["2.0"]}`
- compression: `{"compression_tiers_used": ["0"], "request_count_with_compression": 0, "request_counts_by_tier": {"0": 65}, "table_downsample_strides_used": ["1"]}`
- usage_totals: `{"completion_tokens": 192477, "prompt_tokens": 217093, "total_tokens": 409570}`
- observability_coverage: `{"requests_with_runtime_precision": 65, "requests_with_runtime_provider": 65}`
- resumed_request_count: `65`
- run_observability_csv: `results/kimi-k2.5_all_abms_chain/04_generation_smoke_latest/runs/run_20260313_115013_235788/run_observability.csv`
