# Testing Report

## Required Quality Gates

1. `uv run pytest`
2. `uv run ruff check .`
3. `uv run mypy src tests`
4. `uv build`

## Test Scope

- `tests/e2e`: CLI command and policy behavior.
- `tests/integration`: full pipeline execution behavior.
- `tests/unit`: focused module behavior.

## Key Guarantees Covered

1. Benchmark/debug model policy enforcement.
2. Evidence mode and text-source mode behavior.
3. Summarizer routing across all four summarizers.
4. Reproducibility metadata and resumable runs.
5. DOE/metric utilities and smoke orchestration.

## Execution Status (Current)
- Full test suite passes.
- Lint and type checks pass.
- Build succeeds.
