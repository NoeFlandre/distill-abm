# Pipeline and architecture

## End-to-end flow

The main `run` command follows a deterministic sequence around an external model call:

1. Resolve runtime defaults, model policy, ABM configuration, prompts, and scoring references.
2. Load and normalize the simulation CSV, parameter text, and documentation text.
3. Generate the configured plot and optional statistical evidence.
4. Build a context prompt and request the model's context explanation.
5. Build one or more trend prompts from the context and selected evidence.
6. Request trend explanations and optionally summarize them locally.
7. Score selected and full-text outputs against configured references.
8. Write reports, metadata, prompt signatures, traces, and artifact manifests.

The `plot`, `table`, and `plot+table` evidence modes are experimental inputs. A table is derived statistical evidence for the plotted series; it is not a raw dump of the input CSV.

## Module boundaries

| Package | Responsibility |
| --- | --- |
| `distill_abm.cli` | Typer command definitions, option validation, and public workflow routing. |
| `distill_abm.cli_actions` | Command execution, path resolution, model aliases, and result formatting. |
| `distill_abm.configs` | Pydantic-validated YAML loading and runtime-default resolution. |
| `distill_abm.ingest` | CSV normalization and NetLogo documentation/parameter extraction. |
| `distill_abm.viz` | Ordered plot generation and visualization smoke checks. |
| `distill_abm.pipeline` | End-to-end runs, smoke suites, resumability, reports, monitoring, and studies. |
| `distill_abm.summarize` | Optional BART, BERT, T5, and Longformer-based summarization. |
| `distill_abm.eval` | Lexical/reference scoring, qualitative evaluation, DOE, and ANOVA utilities. |
| `distill_abm.llm` | OpenRouter, Mistral, and compatible adapter contracts. |
| `distill_abm.run_viewer*` | Static review-viewer payloads and HTML generation for case-based runs. |

## Evidence and artifact lifecycle

The standard pipeline output directory contains the core artifacts:

| Artifact | Purpose |
| --- | --- |
| `plot_*.png` | Plot evidence selected for prompting and review. |
| `stats_table.csv` | Statistical evidence when table output is enabled. |
| `report.csv` | Structured generated text and scores. |
| `pipeline_run_metadata.json` | Inputs, settings, references, signatures, selected artifacts, and run provenance. |
| `debug_trace/` | Request/response traces and runtime details for inspection. |
| `manifests/` | Artifact records used by resumable execution and review tooling. |

Case-based smoke workflows use run-separated directories under `runs/run_<timestamp>/`. They maintain pointers such as `latest_run.txt`, structured JSONL logs, manifests, review CSVs, and, where applicable, a static `review.html` viewer. A run can be reused only when its signature and required artifacts still match.

## Operational boundaries

- Configuration is validated before it is used to build prompts or run an experiment.
- Benchmark and debug model policy is enforced at the CLI boundary.
- Provider credentials are read from environment variables; they are not stored in YAML or result artifacts by design.
- NetLogo and API-backed execution can be slow or costly. Use ingest, visualization, and DOE smoke stages to validate prerequisites before real inference.
- Results are generated artifacts, not source-controlled code. See [Results and synchronization](RESULTS_BUCKET.md) for the publication boundary.

## Research workflow stages

The paper-facing workflow is organized as a screening stage over prompt, evidence, and summarizer factors followed by an optimization stage using retained settings and stronger deployment-oriented models. The CLI exposes these stages through focused smoke, quantitative, and study commands; it does not make a smoke run equivalent to a published benchmark result.
