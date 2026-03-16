# distill-abm

`distill-abm` is the publication-facing codebase for the ABM-to-LLM distillation pipeline used in the accompanying paper. The repository contains the code, configs, benchmark assets, and reproducibility logic. Published run outputs live in the Hugging Face results bucket, not in Git.

## Repository Scope

The pipeline supports six main stages:

1. ingest ABM parameters and documentation
2. generate simulation plots and statistical evidence
3. generate trend narratives from evidence
4. optionally summarize the generated narratives
5. score outputs against reference texts
6. produce quantitative summary tables

Benchmark model policy is enforced in the CLI. Paper-facing benchmark runs are restricted to:

1. `moonshotai/kimi-k2.5`
2. `google/gemini-3.1-pro-preview`
3. `qwen/qwen3.5-27b`

Supported summarizers are `bart`, `bert`, `t5`, and `longformer_ext`.

## Canonical Setup

The supported setup path is local `uv` on Python 3.11.

```bash
uv sync --frozen --extra dev
```

Environment assumptions:

- tested on macOS and Linux with Python 3.11
- NetLogo-based workflows require a working local NetLogo installation
- API-backed workflows require provider credentials such as `OPENROUTER_API_KEY` and, for debug-only Mistral paths, `MISTRAL_API_KEY`

Runtime notes:

- `uv run pytest` currently completes in about one minute in this workspace
- NetLogo and API-backed smoke or paper runs are substantially slower and depend on local hardware, provider latency, and model choice
- provider defaults are documented in [docs/HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md)

## Canonical Workflows

Validate the local workspace without calling any LLM:

```bash
uv run distill-abm validate-workspace --json
```

Fetch published results from the Hugging Face bucket:

```bash
hf sync hf://buckets/NoeFlandre/distill-abms-results ./results
```

Sync a local `results/` tree back to the bucket:

```bash
uv run distill-abm sync-results-bucket
```

Run the full pipeline on one input bundle:

```bash
uv run distill-abm run \
  --csv-path data/samples/sim.csv \
  --parameters-path data/samples/params.txt \
  --documentation-path data/samples/docs.txt \
  --model-id kimi_k2_5 \
  --evidence-mode plot+table \
  --text-source-mode summary_only \
  --summarizer bart --summarizer bert --summarizer t5 --summarizer longformer_ext
```

Run the standard fixed-factor optimization smoke chain used for the current exploitation preset:

```bash
uv run distill-abm smoke-optimization-gemini-chain \
  --netlogo-home /path/to/NetLogo
```

## Reproducing the Paper Outputs

The publication contract is:

1. set up the repository with `uv`
2. fetch the frozen results bucket into `./results`
3. inspect the paper-facing quantitative rollups in `results/quantitative_master_overview/`
4. rerun selected analysis or smoke commands only if you need to regenerate derived artifacts locally

The fastest entrypoint for the published outputs is:

- bucket URI: `hf://buckets/NoeFlandre/distill-abms-results`
- bucket web UI: `https://huggingface.co/buckets/NoeFlandre/distill-abms-results`

The paper itself is stored in `data/paper/Main.txt` and `data/paper/Main.pdf`.

## Repository Layout

```text
src/distill_abm/        package source
configs/                runtime, model, prompt, and ABM configs
data/abms/              benchmark ABM assets and repo-local fallback artifacts
data/summaries/         reference texts used for scoring
tests/                  unit, integration, and e2e tests
docs/                   focused reader-facing technical documentation
results/README.md       pointer to the external published results store
```

## Architecture and Module Roles

The main runtime boundaries are:

- `distill_abm.cli`: CLI entrypoint and workflow routing
- `distill_abm.ingest`: CSV and NetLogo ingestion
- `distill_abm.viz`: simulation plots and statistical evidence generation
- `distill_abm.pipeline`: orchestration for pipeline runs, smokes, suites, and reports
- `distill_abm.summarize`: summarizer runners and text cleanup
- `distill_abm.eval`: lexical metrics, reference scoring, DOE, and ANOVA utilities
- `distill_abm.llm`: provider adapters and request defaults

For the fuller runtime/data-flow description, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Additional Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/RESULTS_BUCKET.md](docs/RESULTS_BUCKET.md)
- [docs/HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md)
- [docs/RUN_EXECUTION_ORDER.md](docs/RUN_EXECUTION_ORDER.md)
- [docs/EVALUATION_FREEZE.md](docs/EVALUATION_FREEZE.md)
- [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md)
- [docs/DECISION_LOG.md](docs/DECISION_LOG.md)

Scope-oriented convenience wrapper:

```bash
uv run distill-abm quality-gate --scope pre-llm --json
```

This command is the current repository entrypoint for non-LLM local verification. It runs the standard local checks, emits a structured JSON report, and nests the NetLogo ingest-smoke report under `results/archive/agent_validation/latest/` so agents can inspect outcomes without ad hoc artifact hunting. As with the testing report, treat the command output from the present workspace as authoritative rather than relying on a stale claimed status in documentation.

Repo workflow notes:

- See `docs/AGENT_WORKFLOW.md` for the repository-specific agent workflow, verification order, and evidence expectations.
- See `docs/AGENT_BACKLOG.md` for the current prioritized backlog of agent-friendly quality improvements that remain after the present refactors.
- See `docs/MANUAL_VALIDATION.md` for the current evidence-style manual verification record covering the pre-LLM workflow surfaces.
- See `docs/RUN_EXECUTION_ORDER.md` for a user-facing walkthrough of what happens after `distill-abm run` receives model, evidence-mode, and summarizer arguments.

Agent-oriented CLI additions:

- Most verification-oriented commands now support `--json` so agents can consume structured output instead of parsing human text.
- `quality-gate` is a thin convenience wrapper around `validate-workspace` that maps a declared change scope to the appropriate validation profile and default checks:
  - `static` -> `ruff` + `mypy`
  - `pre-llm` -> quick validation profile
  - `full` -> default validation profile
- `validate-workspace` supports profiles:
  - `quick`: fast static + ingest verification
  - `default`: full local verification
  - `full`: currently equivalent to `default`, reserved as the strictest profile
- `smoke-ingest-netlogo` supports `--require-stage` so callers can assert that specific stage checks are present.
- `smoke-viz` provides stage-filtering and `--require-stage` for the generated simulation CSV and each ordered plot image.
- `smoke-doe` provides a structured pre-LLM view of the full DOE matrix and writes grouped shared artifacts plus compact case/request indexes that can be reviewed without opening thousands of files.
- `smoke-ingest-netlogo`, `smoke-viz`, and `smoke-doe` now follow the same audit contract as the later smokes: each invocation writes into `runs/run_<timestamp>/`, updates `latest_run.txt`, and emits a qualitative root `run.log.jsonl`.
- `smoke-local-qwen` is the legacy command name for the sampled real-inference smoke. It now runs through the configured API model and writes one self-contained folder per sampled case plus a review CSV with exact prompt text, evidence paths, hyperparameters, outputs, and a minimalist static `review.html` viewer.
- `smoke-local-qwen` supports `--resume` and reuses only successful case artifacts; failed or incomplete cases are rerun.
- `smoke-full-case-matrix` runs one ABM across evidence modes, prompt variants, and repetitions, with one context prompt plus all ordered trend prompts per case. It uses the same run separation, resume behavior, `run.log.jsonl`, and `review.html` reviewer surface as the sampled smoke.
- `smoke-optimization-gemini-chain` prepares the standard six smoke stages for the exploitation preset under `results/gemini-3.1-pro-preview_optimization_all_abms_chain/`:
  - `01_ingest_smoke_latest`
  - `02_viz_smoke_latest`
  - `03_doe_smoke_latest`
  - `04_full_case_suite_smoke_latest`
  - `05_summarizer_smoke_latest`
  - `06_quantitative_smoke_latest`
- The current optimization preset is fixed to Gemini (`gemini_3_1_pro_preview`) with the existing provider default `temperature=1.0`, `Evidence=plot`, `Role=off`, `Insights=off`, `Example=on`, repetitions `1..3`, and summarizers `bart`, `bert`, and `t5`.
- `smoke-summarizers` now supports repeated `--summarizer` flags so later runs can intentionally restrict the active summarizer set while preserving the implicit `none` bundle used by the long-reference quantitative path.
- `render-run-viewer` builds the same minimalist static HTML viewer for any existing case-based run directory, including both one-trend sampled smokes and full-case multi-trend runs, or for the latest run when you point it at a root containing `latest_run.txt`.
- `monitor-local-qwen` and `monitor-run` render the same compact live dashboard for case-based smoke runs, including current case or trial, configured `num_ctx`, `max_tokens`, prompt lengths, and observed token usage.
- For all-ABM suite runs, `monitor-run` now reads the suite root directly and uses `suite_progress.json` plus the nested ABM run state so you can monitor the whole run from one place in the terminal.
- `health-check` performs a lightweight read-only validation of configured ABMs, model-registry resolution, and expected ingest/viz artifact roots.
- `ingest-netlogo` and `ingest-netlogo-suite` now write stable artifact manifests.
- Read-only inspection commands are available for agent loops:
  - `uv run distill-abm describe-abm --abm grazing --json`
  - `uv run distill-abm describe-ingest-artifacts --root results/ingest/grazing --json`
  - `uv run distill-abm describe-run --output-dir results/pipeline --json`

Run DOE analysis:

```bash
uv run distill-abm analyze-doe --input-csv results/sweep/combinations_report.csv
```

## Reproducibility Guarantees
Each run writes `pipeline_run_metadata.json` with:

1. Prompt signatures and lengths
2. Model/provider/request defaults
3. Summarizer configuration and enablement
4. Input artifact paths and hashes
5. Run signature for resumable execution
6. Score source provenance (`context_response` or human ground truth file)
7. Additional reference-score blocks when an ABM has optional secondary references

Each run also writes `debug_trace/` with:

1. Snapshotted input files used for the run
2. Request/response JSON for context and trend LLM calls
3. Summarization trace showing the selected text source and outputs
4. Artifact manifests with hashes, sizes, and previews
5. Validation warnings for placeholder-like inputs and missing metric columns

## Verification Commands

```bash
uv run pytest
uv run ruff check .
uv run mypy src tests
uv build
```
