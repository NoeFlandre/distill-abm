# Decision Log

## DL-001
- Date: 2026-03-04
- Decision: Restrict the supported LLM roster to `qwen3.5:0.8b`, `moonshotai/kimi-k2.5`, and `google/gemini-3.1-pro-preview`.
- Rationale: these are the only models used in the benchmark and debugging workflows; legacy Janus and Qwen-VL debug paths should not remain in the codebase because they create stale configuration and review ambiguity.

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
- Rationale: the current DOE smoke path has been manually validated as the review surface for the full pre-LLM experiment design, including grouped shared artifacts, exact context/trend prompt materialization, context-placeholder handoff semantics, evidence-mode-specific prompt wording, raw table evidence extraction from plot-relevant simulation columns, compact request/case indexes, and the request-review CSV used to verify prompt-to-evidence pairing. This path should not be modified casually once validated.
