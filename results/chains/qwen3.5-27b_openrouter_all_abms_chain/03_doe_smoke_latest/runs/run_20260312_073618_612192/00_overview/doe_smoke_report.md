# DOE Smoke Report

## Overview

This report materializes the exact pre-LLM design matrix for the current DOE setup.
It groups shared ABM artifacts separately from case-specific request plans.

- total_cases: `1080`
- total_planned_requests: `14040`
- total_context_requests: `1080`
- total_trend_requests: `12960`
- success: `true`
- design_matrix_csv_path: `results/chains/qwen3.5-27b_openrouter_all_abms_chain/03_doe_smoke_latest/runs/run_20260312_073618_612192/00_overview/design_matrix.csv`
- request_matrix_csv_path: `results/chains/qwen3.5-27b_openrouter_all_abms_chain/03_doe_smoke_latest/runs/run_20260312_073618_612192/00_overview/request_matrix.csv`
- request_review_csv_path: `results/chains/qwen3.5-27b_openrouter_all_abms_chain/03_doe_smoke_latest/runs/run_20260312_073618_612192/00_overview/request_review.csv`
- layout_guide_path: `results/chains/qwen3.5-27b_openrouter_all_abms_chain/03_doe_smoke_latest/00_overview/README.md`
- shared_root: `results/chains/qwen3.5-27b_openrouter_all_abms_chain/03_doe_smoke_latest/10_shared`
- case_index_root: `results/chains/qwen3.5-27b_openrouter_all_abms_chain/03_doe_smoke_latest/20_case_index`
- case_index_jsonl_path: `results/chains/qwen3.5-27b_openrouter_all_abms_chain/03_doe_smoke_latest/runs/run_20260312_073618_612192/20_case_index/cases.jsonl`
- request_index_jsonl_path: `results/chains/qwen3.5-27b_openrouter_all_abms_chain/03_doe_smoke_latest/runs/run_20260312_073618_612192/20_case_index/requests.jsonl`

## DOE Dimensions

- abm_count: `3`
- model_count: `1`
- evidence_mode_count: `3`
- summarization_count: `5`
- prompt_variant_count: `8`
- repetition_count: `3`

## Case Distribution

| group | id | case_count | request_count |
| --- | --- | --- | --- |
| abm | fauna | 360 | 5400 |
| abm | grazing | 360 | 3960 |
| abm | milk_consumption | 360 | 4680 |
| model | qwen3_5_27b | 1080 | - |

## Shared ABM Bundles

| abm | plot_count | source | shared_dir | stage_errors |
| --- | --- | --- | --- | --- |
| fauna | 14 | fallback | `results/chains/qwen3.5-27b_openrouter_all_abms_chain/03_doe_smoke_latest/runs/run_20260312_073618_612192/10_shared/fauna` |  |
| grazing | 10 | fallback | `results/chains/qwen3.5-27b_openrouter_all_abms_chain/03_doe_smoke_latest/runs/run_20260312_073618_612192/10_shared/grazing` |  |
| milk_consumption | 12 | fallback | `results/chains/qwen3.5-27b_openrouter_all_abms_chain/03_doe_smoke_latest/runs/run_20260312_073618_612192/10_shared/milk_consumption` |  |

## Failure Summary

No failed DOE smoke cases.

## Failed Cases

No failed DOE smoke cases.
