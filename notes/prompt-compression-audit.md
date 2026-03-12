# Prompt Compression Audit

Date: 2026-03-12

## Baseline

Command:

```bash
uv run pytest
```

Observed result:

- Baseline on `main` was not clean before this change.
- Existing failures:
  - `tests/unit/pipeline/test_smoke.py::test_run_qwen_smoke_suite_writes_matrix_and_reports`
  - `tests/unit/pipeline/test_smoke_optional_steps.py::test_run_doe_if_requested_writes_output_csv`

## Focused Verification

Command:

```bash
uv run ruff check src/distill_abm/pipeline/prompt_compression_artifacts.py src/distill_abm/pipeline/local_qwen_sample_smoke.py src/distill_abm/pipeline/full_case_smoke.py src/distill_abm/pipeline/full_case_matrix_smoke.py src/distill_abm/run_viewer.py src/distill_abm/run_viewer_payloads.py tests/unit/pipeline/test_local_qwen_sample_smoke.py tests/unit/pipeline/test_full_case_smoke.py tests/unit/test_run_viewer.py
```

Observed result:

- Passed.

Command:

```bash
uv run pytest tests/unit/pipeline/test_local_qwen_sample_smoke.py tests/unit/pipeline/test_full_case_smoke.py tests/unit/test_run_viewer.py tests/unit/pipeline/test_local_qwen_sample_artifacts.py tests/unit/pipeline/test_full_case_suite_smoke.py
```

Observed result:

- `45 passed`

## Environment Gap

Command:

```bash
uv run pytest tests/e2e/test_cli_local_qwen_sample_smoke.py tests/e2e/test_cli_full_case_matrix_smoke.py
```

Observed result:

- Collection failed in this worktree because `typer` was missing from the worktree virtualenv.
