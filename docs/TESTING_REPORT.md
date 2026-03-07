# Testing Report

## Purpose

This document serves as a structured testing checklist. Its role is not to claim formal verification, nor to argue that every possible failure mode has been eliminated. Rather, it records the concrete checks used to build confidence that the implementation is behaviorally consistent with the intended workflow, robust to common failure conditions, and reproducible under the documented execution environment.

The testing strategy follows a simple principle: confidence should be accumulated in layers. At the lowest level, unit tests isolate local transformations, parsing routines, metric calculations, and helper logic. At the middle level, integration tests verify that the main pipeline composes these components correctly. At the highest level, end-to-end tests exercise the command-line interface, model-policy enforcement, smoke workflows, and user-facing control paths. Static analysis and packaging checks complement executable tests by reducing the risk of type drift, interface inconsistency, and release-time regressions.

In this sense, the present report should be read as supplementary evidence that the implementation is operationally sound. It is a checklist used to increase trust in the system, to detect regressions early, and to make the project easier to inspect, extend, and reproduce.

## Required Quality Gates

All changes are expected to pass the following repository-level checks:

1. `uv run pytest`
2. `uv run ruff check .`
3. `uv run mypy src tests`
4. `uv build`

These four gates serve distinct purposes. `pytest` validates behavioral expectations. `ruff` enforces consistency and eliminates avoidable implementation defects. `mypy` checks that typed interfaces remain coherent as the codebase evolves. `uv build` confirms that the project can still be packaged successfully as a distributable artifact.

## Test Inventory

The test suite is organized into three complementary layers. Because the repository is under active development, the exact number of collected tests and files changes over time and should be measured from the current workspace rather than copied into this document.

### Unit Tests

Unit tests provide the broadest coverage footprint. They target local logic whose correctness is critical to downstream behavior, including:

- configuration loading and runtime defaults
- lexical metrics and DOE analysis helpers
- CSV ingestion and NetLogo preprocessing utilities
- LLM adapter contracts and adapter factory behavior
- pipeline prompt construction, reporting helpers, and smoke helpers
- summarization runners, text normalization, and error handling
- visualization and statistics-table generation
- repository-level structural guardrails

### Integration Tests

Integration tests focus on the main paper-aligned execution path. They verify that a run can proceed from structured inputs through evidence preparation, prompt assembly, model invocation, optional summarization, scoring, and artifact persistence. They also check resumability behavior and the detailed debug-trace artifacts written for reproducibility and diagnosis.

### End-to-End Tests

End-to-end tests exercise the CLI as the public operational surface of the repository. These tests verify command wiring, option parsing, benchmark-model policy enforcement, smoke-suite execution, ingest-smoke execution, and failure handling for invalid user inputs.

## Confidence-Building Checklist

The following checklist summarizes the principal implementation guarantees that are currently tested. The intent is to make explicit what has been checked, why it matters, and where the evidence resides.

| Assurance Area | What Is Checked | Why It Matters | Primary Verification Surface |
|---|---|---|---|
| Runtime entrypoints | CLI commands accept valid inputs and reject invalid configurations | The CLI is the main operational interface for experiments and reproductions | `tests/e2e/test_cli.py`, `tests/e2e/test_cli_ingest_smoke.py` |
| Benchmark model policy | Only approved benchmark models are accepted unless debug mode is explicitly enabled | Prevents accidental benchmark contamination and inconsistent evaluation settings | `tests/e2e/test_cli.py` |
| Core pipeline execution | Context generation, trend generation, optional summarization, scoring, and artifact writing run as one coherent workflow | The main scientific output depends on correct orchestration of these stages | `tests/integration/test_pipeline_run.py` |
| Evidence ablations | `plot`, `table`, and `plot+table` modes are routed correctly | Evidence-mode comparisons are central experimental factors | `tests/unit/pipeline/test_smoke.py`, `tests/integration/test_pipeline_run.py` |
| Text-source ablations | `summary_only` and `full_text_only` modes behave as intended | The reported ablation study depends on reliable text-source selection | `tests/e2e/test_cli.py`, `tests/integration/test_pipeline_run.py` |
| Summarizer routing | BART, BERT, T5, and LongformerExt selection and fallback behavior are handled correctly | Summarization is optional, but when enabled it must remain controlled and inspectable | `tests/integration/test_pipeline_run.py`, `tests/unit/summarize/*` |
| Reproducibility metadata | Run signatures, prompt signatures, hashes, and persisted metadata are written consistently | Reproducibility depends on preserving the configuration and artifact state of each run | `tests/integration/test_pipeline_run.py` |
| Detailed debug tracing | Input snapshots, request/response traces, summarization traces, artifact manifests, and validation warnings are persisted | Debugging complex failures requires observing what entered the pipeline and what each stage produced | `tests/integration/test_pipeline_run.py` |
| Ingestion pipeline | NetLogo-derived artifacts, cleaned documentation, narratives, and extracted code are produced correctly | Downstream prompting quality depends on trustworthy preprocessing | `tests/unit/ingest/test_netlogo.py`, `tests/unit/ingest/test_netlogo_workflow.py` |
| Granular ingest smoke checks | Individual ingest stages can be smoke-tested in isolation across configured ABMs | Localized failure isolation is essential when debugging missing, malformed, or placeholder artifacts | `tests/unit/ingest/test_ingest_smoke.py`, `tests/e2e/test_cli_ingest_smoke.py` |
| Evaluation metrics | BLEU, METEOR, ROUGE, Flesch, token F1, and batch/reference score logic remain stable | Experimental comparisons rely on metric correctness | `tests/unit/eval/test_metrics*.py`, `tests/unit/eval/test_reference_scores*.py` |
| DOE analysis | Factorial ANOVA utilities behave correctly on supported inputs and fail safely on invalid ones | Statistical summary tables should not silently degrade under malformed data | `tests/unit/eval/test_doe.py`, `tests/unit/eval/test_doe_full.py` |
| Visualization artifacts | Metric plots and statistics tables are generated with the expected structure | Evidence preparation is a prerequisite for downstream narrative generation | `tests/unit/viz/test_plots.py`, `tests/unit/viz/test_viz_stats_table.py` |
| Repository guardrails | Archive layout, developer assets, and legacy-surface constraints remain intact | Repository hygiene reduces operational drift and accidental reintroduction of outdated surfaces | `tests/unit/repo/*` |

## Failure Modes Explicitly Exercised

The suite is designed not only to confirm nominal behavior, but also to exercise a range of likely failure conditions. These include:

- unsupported or disallowed benchmark-model selections
- unknown smoke-case identifiers and invalid CLI option combinations
- missing, malformed, or incomplete metadata during resumable execution
- missing plot artifacts or corrupted metadata files during pipeline resume
- absent or invalid experiment-parameter files
- unreadable or statistically invalid DOE inputs
- summarizer failures and summary fallback behavior
- adapter-level request failures and provider-contract errors
- placeholder-like or otherwise suspicious ingestion artifacts surfaced through debug traces and ingest smoke checks
- dependency-missing fallbacks in evaluation helpers without silently hiding runtime failures inside the metric or ANOVA implementations

This emphasis on error-path testing is important because the project is intended for iterative experimentation. In such settings, robustness is not only about obtaining outputs when everything is well configured; it is also about failing in ways that are legible, localized, and recoverable.

## Testing Strategy by System Boundary

### Configuration and Policy Boundary

Configuration files and model-registry settings are tested to ensure that the runtime environment remains explicit and controlled. These checks reduce the risk of hidden default drift, invalid model aliases, or accidental execution under unsupported benchmarking conditions.

### Data and Ingestion Boundary

Input handling is tested at two levels. First, unit tests verify parsing and artifact extraction from CSV and NetLogo sources. Second, granular ingest smoke checks validate that intermediate ingestion products are present, non-empty, and suitable for downstream use. This boundary is especially important because failures here can propagate silently into prompt construction if not caught early.

### Prompting and Generation Boundary

Prompt composition, evidence routing, and adapter invocation are tested as explicit system boundaries. These checks establish that the information supplied to models is structurally correct and that the resulting outputs are persisted in a form that supports later inspection. The debug-trace artifacts strengthen this boundary further by exposing the exact prompt and response objects used during execution.

### Evaluation and Reporting Boundary

Metric calculations, reporting helpers, and DOE analysis are tested separately from the generation logic. This separation matters because evaluation code can introduce distortions even when upstream generation is functioning correctly. By testing these surfaces independently, the project reduces the risk that errors in scoring or reporting are mistaken for model behavior.

## Operational Interpretation

The testing evidence reported here should be interpreted as implementation assurance rather than as proof of scientific validity. Passing tests indicate that the software behaves consistently with the encoded expectations of the repository. They do not, by themselves, guarantee that prompts are epistemically complete, that model outputs are semantically correct in all cases, or that every possible dataset and ABM will behave identically under future changes.

What the checklist does provide is a disciplined basis for confidence. It demonstrates that:

1. the core workflow executes end to end;
2. the principal ablation dimensions are checked explicitly;
3. intermediate and final artifacts are inspected rather than assumed;
4. common operational failures are surfaced through targeted tests;
5. reproducibility and debugging metadata are treated as first-class outputs of the system.

## Current Status

This document does not serve as a durable claim that the repository is currently green.

The authoritative status is the output of the repository-level quality gates run against the present workspace:

1. `uv run pytest`
2. `uv run ruff check .`
3. `uv run mypy src tests`
4. `uv build`

Before citing a passing state in a PR, release note, or demo artifact, rerun those commands and record the actual results, including any collection failures, skipped surfaces, or environment-dependent checks.
