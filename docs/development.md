# Development and verification

## Repository layout

```text
src/distill_abm/     package source and CLI
configs/             validated runtime, prompt, model, and ABM settings
data/                benchmark models, fallback inputs, and references
docs/                public MkDocs source
tests/               unit, integration, end-to-end, and repository tests
results/             ignored generated artifacts and local validation reports
```

The repository is a `uv` project with a strict mypy configuration, Ruff checks, pytest suites, and a package build. Keep generated outputs out of source-control paths.

## Verification commands

Install the development environment:

```bash
uv sync --frozen --extra dev
```

Run the normal quality gates:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src tests
uv build
```

Build the public documentation locally:

```bash
uv run mkdocs build --strict --site-dir /tmp/distill-abm-site
```

Use the CLI wrapper when you want a structured validation report:

```bash
uv run distill-abm quality-gate --scope static --json
uv run distill-abm validate-workspace --profile default --json
```

The full validation suite can run pre-LLM ingestion checks and writes report artifacts under its configured output root. It does not make a provider-backed run safe by itself; inspect inputs and staged artifacts before external inference.

## Tests by layer

- `tests/unit/` covers configuration, ingestion, adapters, summarization, evaluation, visualization, pipeline helpers, run-state contracts, and repository hygiene.
- `tests/integration/` exercises real pipeline composition with controlled adapters and filesystem artifacts.
- `tests/e2e/` treats the Typer CLI as a public interface.
- `tests/unit/repo/` checks required files, documentation surfaces, benchmark assets, and retired legacy paths.

Prefer a narrow regression test for a discovered defect. Run the focused test first, then the relevant package or full suite.

## Documentation workflow

The public site is built from `docs/` with `mkdocs.yml`. Internal design and implementation notes are excluded from the rendered site.

Before changing public docs:

1. Verify commands against `distill-abm --help` and the current configuration.
2. Keep links repository-relative or use stable public URLs.
3. State whether a command is local-only, NetLogo-backed, provider-backed, or results-mutating.
4. Build with `--strict` and inspect the generated site.

## Contribution expectations

Keep changes focused and reviewable. Do not rewrite immutable benchmark inputs, delete generated results, upload artifacts, or modify remote state without an explicit review step.

Use Conventional Commit subjects, for example:

```text
docs: publish MkDocs site
test: add documentation contract checks
ci: build documentation on main
```

Run the relevant checks before opening a pull request and describe commands, observed results, and intentionally skipped checks. The pull-request template includes the current documentation and reproducibility checklist.
