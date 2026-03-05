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
- Full test suite passes (~199 tests).
- Lint and type checks pass.
- Build succeeds.

## Recent Coverage Expansions

### Error Path Coverage
- CLI: continue-on-missing behavior, unknown case-id rejection, model registry errors, missing/invalid experiment parameters
- Smoke helpers: missing metadata file handling
- Run-state resumability: signature mismatch, missing artifacts, malformed metadata
- DOE: unreadable CSV, invalid content, empty data, missing factors/metrics, zero variance
- NetLogo parser: missing doc/code sections, empty GUI/experiments
- Adapters: API key errors, completion failures, HTTP errors, connection errors
