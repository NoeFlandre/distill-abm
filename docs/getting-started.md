# Getting started

## Prerequisites

| Requirement | Why it is needed |
| --- | --- |
| Python 3.11 or newer | Supported project runtime. |
| `uv` | Reproducible environment and build management. |
| A local checkout | Commands and default paths are repository-relative. |
| NetLogo | Required only for NetLogo-backed workflows. |
| Provider credentials | Required only for API-backed inference. |

## Install the environment

From the repository root:

```bash
uv sync --frozen --extra dev
```

The development extra includes tests, linting, type checking, packaging tools, and the MkDocs Material documentation toolchain. Install the summarizer runtime only when needed:

```bash
uv sync --frozen --extra dev --extra summarizers
```

## Validate without model calls

Start with the static quality gate. It runs Ruff and mypy and does not call an LLM:

```bash
uv run distill-abm quality-gate --scope static --json
```

For the broader non-LLM validation suite, use:

```bash
uv run distill-abm validate-workspace --json
```

Use `--check` to select checks such as `pytest`, `ruff`, `mypy`, `build`, or `smoke-ingest-netlogo`. The `full` validation profile may create reports below `results/`; those generated files are intentionally not part of the source tree.

Inspect the available command surface at any time:

```bash
uv run distill-abm --help
```

## Prepare one pipeline run

The core `run` command requires three input files:

- `--csv-path`: simulation data in CSV form;
- `--parameters-path`: parameter text or an extracted parameter artifact;
- `--documentation-path`: model documentation or an extracted documentation artifact.

Select a configured model alias with `--model-id` and, when the input corresponds to a known preset, an ABM with `--abm`:

```bash
uv run distill-abm run \
  --csv-path /path/to/simulation.csv \
  --parameters-path /path/to/parameters.txt \
  --documentation-path /path/to/documentation.txt \
  --model-id kimi_k2_5 \
  --abm fauna \
  --evidence-mode plot+table \
  --text-source-mode summary_only
```

The default output root is `results/pipeline`. Use `--output-dir` to isolate an experiment. Use `--json` when a calling script needs structured command output.

## NetLogo ingestion

Ingestion extracts model documentation and experiment parameters into explicit artifacts. Run one model or a suite:

```bash
uv run distill-abm ingest-netlogo \
  --model-path data/abms/fauna/fauna.nlogo \
  --output-dir results/ingest/fauna

uv run distill-abm ingest-netlogo-suite \
  --models-root data \
  --output-root results/ingest
```

Use `smoke-ingest-netlogo` or the corresponding validation check to audit configured ABMs before starting inference. A local NetLogo installation is required for workflows that execute simulations rather than only inspect artifacts.

## Credentials and cost boundaries

OpenRouter and Mistral adapters read `OPENROUTER_API_KEY` and `MISTRAL_API_KEY` from the environment. Do not write credentials to YAML, source files, generated reports, or commits.

API-backed inference can incur cost and provider rate limits. Begin with a narrow smoke or one-case run, inspect the generated artifacts, and expand only after the inputs and evidence are correct. See [Results and synchronization](RESULTS_BUCKET.md) before uploading generated output.

## Where to go next

- Read the [pipeline architecture](ARCHITECTURE.md) to understand stages and artifacts.
- Review [configuration](CONFIG_REFERENCE.md) before changing model, prompt, or reference settings.
- Use the [CLI reference](cli-reference.md) for workflow selection.
- Follow [development and verification](development.md) before opening a change.
