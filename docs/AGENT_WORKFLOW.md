# Agent Workflow

This repository is maintained with an agentic engineering workflow. The intent is not to produce the largest volume of code. The intent is to produce small, verified, reviewable changes with durable evidence.

## Default Loop

1. Read `AGENTS.md`, then inspect the relevant implementation, tests, and docs.
2. Establish a baseline before editing:
   - `uv run pytest`
   - `uv run ruff check src tests`
   - `uv run mypy src tests`
3. Prefer red/green TDD when behavior changes:
   - add or update the narrowest useful failing test
   - implement the smallest change
   - rerun the narrow check
   - rerun the broader suite
4. Manually exercise the affected workflow whenever it is practical.
5. Leave behind evidence:
   - updated tests
   - updated docs
   - smoke artifacts under `results/`
   - a walkthrough or validation note when the change spans multiple steps

## Repo Verification Order

Use the narrowest command that proves the point, then widen.

### Static Checks

- `uv run ruff check src tests`
- `uv run mypy src tests`

### Automated Tests

- `uv run pytest`
- or a targeted subset such as:
  - `uv run pytest tests/unit/pipeline/test_doe_smoke.py`
  - `uv run pytest tests/e2e/test_cli.py -k smoke_doe`

### Manual Pre-LLM Validation

- `uv run distill-abm smoke-ingest-netlogo --models-root data --output-root results/ingest_smoke_latest`
- `uv run distill-abm smoke-viz --models-root data --output-root results/viz_smoke_latest --netlogo-home /path/to/NetLogo`
- `uv run distill-abm smoke-doe --ingest-root results/ingest_smoke_latest --viz-root results/viz_smoke_latest --output-root results/doe_smoke_latest`

### Canonical Agent Verification

- `uv run distill-abm validate-workspace --json`
- `uv run distill-abm quality-gate --scope pre-llm --json`

This is the default agent-facing verification entrypoint when a change touches multiple subsystems.

Use `quality-gate` when the change scope is obvious and you want the CLI to choose the matching validation profile or check roster for you. Use `validate-workspace` when you need direct control over the exact checks.

## Evidence Expectations

The repository already preserves several evidence-producing surfaces:

- `results/ingest_smoke_latest/`
- `results/viz_smoke_latest/`
- `results/doe_smoke_latest/`
- `results/agent_validation/latest/`

When a task changes one of these workflows, the agent should leave the refreshed artifacts behind and reference them in the final report instead of only claiming that the workflow was run.

When a durable narrative artifact would help review, prefer `uvx showboat` to build the document from executed commands rather than writing the results by hand.

## Reviewability Rules

- Prefer one focused commit per coherent change.
- Keep diffs small enough that a human can actually review them.
- If a change is subtle, update a walkthrough or add a short validation note.
- If a behavior was validated and intentionally frozen, update `docs/DECISION_LOG.md`.

## Manual Testing Notes

`showboat` and `rodney` are available as external CLIs:

- use `uvx showboat --help` before using Showboat
- use `uvx rodney --help` before using Rodney

Current repo guidance:

- Showboat is relevant for validation and walkthrough artifacts.
- Rodney is only relevant when the repository exposes a browser UI or local web app that needs manual/browser verification.
- For the current CLI and pre-LLM workflow surface, preserve command transcripts and result artifacts; do not introduce browser automation where there is no browser surface to inspect.
