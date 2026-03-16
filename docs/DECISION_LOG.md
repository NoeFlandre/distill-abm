# Decision Log

This file keeps only the decisions that shape the public, paper-facing workflow.

## DL-001

- Date: 2026-03-04
- Decision: Restrict the benchmark model roster to `moonshotai/kimi-k2.5`, `google/gemini-3.1-pro-preview`, and `qwen/qwen3.5-27b`.
- Rationale: the publication benchmark should stay API-only and use one stable, reviewable model roster.

## DL-002

- Date: 2026-03-04
- Decision: Treat `bart`, `bert`, `t5`, and `longformer_ext` as the supported summarizers.
- Rationale: these are the summarizers reflected in the current pipeline defaults and paper-facing evaluation path.

## DL-003

- Date: 2026-03-04 and 2026-03-09
- Decision: Preserve the public evidence labels `plot`, `table`, and `plot+table`, where `table` means statistical evidence derived from the plot-relevant simulation series.
- Rationale: this keeps prompt and output naming stable while avoiding raw CSV dumps in the evidence contract.

## DL-004

- Date: 2026-03-04
- Decision: Preserve the two public text-source modes `summary_only` and `full_text_only`.
- Rationale: these modes define the main ablation axis used by the current pipeline and evaluation workflow.

## DL-005

- Date: 2026-03-06
- Decision: Keep `docs/supplementary_material/` frozen unless a task explicitly requests changes there.
- Rationale: those files are part of the paper supplement and are intentionally outside the main repository documentation sweep.

## DL-006

- Date: 2026-03-11
- Decision: Freeze the current quantitative evaluation contract unless a task explicitly asks for an evaluation change.
- Rationale: the quantitative surface has already been debugged and validated, and publication work should not change metric or reporting semantics casually.
