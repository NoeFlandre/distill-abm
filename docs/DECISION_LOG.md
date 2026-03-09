# Decision Log

## DL-001
- Date: 2026-03-04
- Decision: Restrict the benchmark model roster to `qwen/qwen3.5-27b`, `moonshotai/kimi-k2.5`, and `google/gemini-3.1-pro-preview`, with `nvidia/nemotron-nano-12b-v2-vl:free` and `mistral-medium-latest` kept as debug-only models.
- Rationale: benchmark and debug workflows should share the same API-only runtime shape while keeping production-vs-debug model policy explicit. Legacy local-model, Janus, and Qwen-VL paths create stale configuration and review ambiguity.

## DL-002
- Date: 2026-03-04
- Decision: Treat `bart`, `bert`, `t5`, `longformer_ext` as first-class summarizers.
- Rationale: requested extension to baseline paper setup.

## DL-003
- Date: 2026-03-04
- Decision: Standardize evidence mode names to `plot`, `table`, `plot+table`.
- Rationale: align outputs and simplify ablation labeling.

## DL-004
- Date: 2026-03-04
- Decision: Standardize text-source modes to `summary_only` and `full_text_only`.
- Rationale: explicit ablation axis without implicit mode coupling.

## DL-005
- Date: 2026-03-06
- Decision: Freeze the validated NetLogo ingestion pipeline unless a task explicitly requests changes.
- Rationale: the current benchmark-model ingestion flow, including experiment-parameter extraction, GUI-parameter extraction, updated-parameter generation, narrative generation, documentation extraction and cleanup, final-documentation export, code extraction, and stage-level smoke validation, has been manually validated end-to-end and should not be modified casually or reworked speculatively.

## DL-006
- Date: 2026-03-06
- Decision: Freeze the validated visualization smoke pipeline unless a task explicitly requests changes.
- Rationale: the current pre-LLM visualization workflow, including ABM-specific NetLogo visualization configuration, repo-local model inputs, preserved legacy reference CSV/plot artifacts, fallback-first smoke generation, artifact-source reporting, and ordered plot emission for all three benchmark ABMs, has been validated as the desired debugging path and should not be reworked casually.

## DL-007
- Date: 2026-03-06
- Decision: Freeze `docs/TESTING_REPORT.md` unless a task explicitly requests changes.
- Rationale: the testing supplementary material has been manually edited into the publication-ready checklist style requested for this project and should not be rewritten or reformatted casually.

## DL-008
- Date: 2026-03-06
- Decision: Freeze `docs/GROUND_TRUTHS_GPT5.2.md` unless a task explicitly requests changes.
- Rationale: the GPT-5.2 ground-truth supplementary material has been manually curated into the validated structure and wording expected for the paper supplement, while preserving prompt and model-output content, and should not be modified casually.

## DL-009
- Date: 2026-03-07
- Decision: Freeze the validated pre-LLM DOE smoke workflow unless a task explicitly requests changes.
- Rationale: the current DOE smoke path has been manually validated as the review surface for the full pre-LLM experiment design, including grouped shared artifacts, exact context/trend prompt materialization, context-placeholder handoff semantics, evidence-mode-specific prompt wording, plot-relevant statistical table evidence, compact request/case indexes, and the request-review CSV used to verify prompt-to-evidence pairing. This path should not be modified casually once validated.

## DL-010
- Date: 2026-03-09
- Decision: Preserve the public evidence labels `plot`, `table`, and `plot+table`, but redefine `table` as statistical evidence derived only from the plot-relevant simulation series.
- Rationale: reviewer guidance explicitly asks for statistical summaries rather than raw CSV dumps. This must be implemented at the shared helper/core pipeline layer so DOE, smoke workflows, and real runs all use the same evidence semantics.
