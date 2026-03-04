# Testing Report

This is the single authoritative testing reference for the project.
Use this document as the publication-facing source for what is tested, how it is tested, and what constitutes a passing release.

## 1. Purpose

Testing enforces four guarantees:

1. Runtime correctness for production workflows (ingest, prompting, evidence ablations, summarization, scoring, report export).
2. Stability of compatibility behavior for reference-aligned helper surfaces.
3. Reproducibility through deterministic metadata and strict quality gates.
4. Regression protection for future refactors.

## 2. Release Gate (Required)

A change is release-eligible only if all checks pass:

```bash
uv run pre-commit run --all-files
uv run ruff check .
uv run black --check .
uv run mypy src tests
uv run pytest --cov=distill_abm --cov-report=term-missing --cov-fail-under=85
```

## 3. Current Baseline

Latest validated baseline:

- Total tests: `227`
- Passing tests: `227`
- Coverage threshold: `>=85%`
- Observed coverage: `85.62%`
- Failing gates: `0`


### Repository hygiene checks

The repository policy tests now enforce:

- root artifact folder normalization: `results/` is required in clean state,
- explicit ignore coverage in `.gitignore` and `.dockerignore` for both `results` and `Results`,
- deterministic behavior under rerun-resume contracts.

## 4. Test Taxonomy

### Unit tests

Validate deterministic behavior of isolated modules:

- `tests/unit/ingest/`: CSV and NetLogo ingestion utilities
- `tests/unit/viz/`: plot rendering and stats table generation
- `tests/unit/summarize/`: summarizer runners, batch summarization, text cleanup
- `tests/unit/eval/`: lexical metrics, qualitative parsing, DoE helpers
- `tests/unit/pipeline/`: prompt composition, mode resolution, CSV writers, sweep helpers
- `tests/unit/compat/`: compatibility wrappers, reference loader behavior, fallback dispatch
- `tests/unit/configs/`: strict typed config loading/validation
- `tests/unit/repo/`: repo-level policy checks (layout/devops assets)

### Integration tests

Validate multi-module runtime behavior:

- `tests/integration/test_pipeline_run.py`:
  - evidence modes (`plot`, `table-csv`, `plot+table`)
  - summarization modes (`full`, `summary`, `both`)
  - scoring modes (`full`, `summary`, `both`)
  - metadata output integrity
  - adapter request behavior (image/text paths)
  - prompt text capture in metadata (`prompts.context_prompt`, `prompts.trend_prompt`)
- `tests/integration/test_pipeline_uses_abm_and_full_metrics.py`: ABM config defaults with full metric output.

### End-to-end tests

Validate CLI entrypoints and option forwarding:

- `tests/e2e/test_cli.py`
  - `run`
  - `analyze-doe`
  - `evaluate-qualitative`
  - `smoke-qwen`
  - mode/option wiring and invalid-option rejection

### Regression tests

Protect compatibility/replay guarantees:

- `tests/regression/test_reference_equivalence.py`
- `tests/regression/test_reference_function_coverage.py`
- `tests/regression/test_runtime_reference_dependencies.py`
- `tests/regression/test_prompt_reference_equivalence.py`
- `tests/regression/test_archive_manifest.py`
- `tests/regression/test_archive_retired.py`
- `tests/regression/test_model_assets_migration.py`
- `tests/regression/test_evaluation_assets_migration.py`

## 5. What Is Explicitly Verified

### Pipeline behavior

- Context prompt composition from model parameters + documentation + style features.
- Trend prompt composition with controlled evidence ablations.
- Evidence-path contract:
  - `plot`: image only
  - `table-csv`: CSV text table only, no vision attachment
  - `plot+table`: image plus CSV table text
- Summarization contract:
  - full text only
  - summary only
  - dual path (both)
- Optional add-on summarizers:
  - `t5`
  - `longformer_ext`
  - robust fallback when one backend is unavailable.
- Ollama request wiring:
  - `temperature` is mapped to `options.temperature`
  - `max_tokens` is mapped to `options.num_predict`

### Scoring behavior

- Metric computation for selected text path(s).
- Extended report columns when dual scoring is enabled.
- Qualitative score extraction contract (`coverage`, `faithfulness`).

### Artifact behavior

- Output files are created in expected paths.
- `pipeline_run_metadata.json` includes requested/resolved settings and model hyperparameters.
- CSV report headers match selected mode behavior.
- Smoke-suite outputs (`smoke_report.md/json`, per-case folders, DoE output, sweep output) are validated.

## 6. Reproducibility and Audit

Every production run persists:

- `report.csv`
- plot artifact(s)
- `pipeline_run_metadata.json`

`pipeline_run_metadata.json` is the reproducibility key and captures:

- input paths
- requested + resolved runtime modes
- prompt signatures/lengths
- provider/model request metadata
- summarizer configuration
- selected score outputs

## 7. CI Alignment

CI runs the same gate sequence as local release validation. There is no separate “hidden” CI-only requirement.

Workflow file:

- `.github/workflows/ci.yml`

## 8. Failure Triage Rules

When a gate fails:

1. Reproduce locally with the exact failing command.
2. Fix root cause, not test expectations.
3. Re-run full gate sequence (not just the failing test file).
4. Do not lower coverage threshold without explicit approval.

## 9. Acceptance Criteria for New Changes

A new feature/refactor is accepted only when:

1. Targeted tests are added first or updated with clear intent.
2. Existing tests stay green unless behavior change is deliberate and documented.
3. All release gates pass.
4. User-facing docs are updated if behavior/CLI changes.

## 10. Canonical Command Set

Quick full validation:

```bash
uv run pre-commit run --all-files \
  && uv run ruff check . \
  && uv run black --check . \
  && uv run mypy src tests \
  && uv run pytest --cov=distill_abm --cov-report=term-missing --cov-fail-under=85
```
