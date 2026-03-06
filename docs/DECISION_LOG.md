# Decision Log

## DL-001
- Date: 2026-03-04
- Decision: Keep Qwen-VL OpenRouter model as debug-only smoke path.
- Rationale: low-cost troubleshooting model should not contaminate benchmark outputs.

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
