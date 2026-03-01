# distill-abm

Production Python package for ABM-to-LLM distillation, extracted from legacy notebooks with regression-backed parity.

## Repository layout

- `src/distill_abm/` - main production codebase
- `tests/` - unit, integration, e2e, and notebook-regression tests
- `configs/` - YAML configs for models, prompts, ABMs, evaluation, logging
- `archive/legacy_repo/` - full legacy repository snapshot (notebooks + assets)
- `docs/` - architecture and parity documentation

## Quickstart

```bash
uv sync --extra dev
uv run ruff check .
uv run black --check .
uv run mypy src tests
uv run pytest -q
```

## CLI

Run distillation pipeline:

```bash
uv run distill-abm run \
  --csv-path path/to/reduced.csv \
  --parameters-path path/to/params.txt \
  --documentation-path path/to/docs.txt \
  --abm milk_consumption \
  --provider openai \
  --model gpt-4o
```

Run DoE ANOVA analysis:

```bash
uv run distill-abm analyze-doe \
  --input-csv path/to/FinalResultsYesNo.csv \
  --output-csv results/doe/anova_factorial_contributions.csv
```

## Parity policy

- Legacy notebooks are preserved under `archive/legacy_repo/`.
- `distill_abm.legacy.notebook_loader` builds a callable registry from notebooks and prefers sources in this order:
  - non-`archives` notebooks
  - non-checkpoint notebooks
  - non-`copy` notebook filenames
- `distill_abm.legacy.compat` dispatches notebook-first for selected deterministic helpers and falls back to refactored package implementations when notebook loading/calls fail.
- `tests/regression/test_notebook_equivalence.py` validates behavior parity for migrated core utilities.
- `tests/regression/test_notebook_function_coverage.py` ensures every notebook function name is accounted for in the new codebase (`distill_abm.legacy.compat`) or explicitly exempted.

## Known limits

- Notebook execution is intentionally restricted to safe AST node types in the loader; some notebook side-effects are not replayed.
- Fallback behavior for complex legacy DoE CSV writers (`return_csv`, `return_csv_2`) aims for robustness but may differ in formatting if notebook execution is unavailable.

See `docs/PARITY.md` for details.
