# Evaluation Freeze

## Status

As of 2026-03-11, the quantitative evaluation path is frozen in its current validated form. Do not modify this surface unless a task explicitly asks for an evaluation change.

## Freeze Scope

The freeze applies to the quantitative scoring and publication path rooted in:

- `src/distill_abm/eval/`
- `src/distill_abm/pipeline/quantitative_smoke.py`
- `src/distill_abm/pipeline/quantitative_rendering.py`
- `src/distill_abm/cli_support.py` reference-resolution behavior used by `smoke-quantitative`
- the quantitative smoke output contract under `results/.../06_quantitative_smoke_latest/`

## Frozen Contract

The frozen behavior is:

1. Lexical metrics must be computed for real. Missing metric dependencies or resources are hard failures, not silent fallbacks.
2. Reference scoring remains additive. Author ground truth is primary, while `gpt5.2_short`, `gpt5.2_long`, and `modeler` references are scored in parallel when configured.
3. Publication artifacts are compatibility-filtered:
   - `author`, `modeler`, and `gpt5.2_short` evaluate summarized outputs only
   - `gpt5.2_long` evaluates the `none` output only
   - summary-reference factorials keep `Summarizer` as a factor
   - `gpt5.2_long` uses a separate factorial without `Summarizer`
4. DOE factor detection includes low-cardinality prompt-factor columns such as `Role`, `Insights`, and `Example` when present.
5. Factorial rendering distinguishes absent terms, tiny nonzero effects, and values that round to zero.
6. The publication `overview/` surface contains exactly four files:
   - `anova_table.md`
   - `evidence_summary_table.md`
   - `factorial_table.md`
   - `best_scores_table.md`
7. Higher-precision machine-readable outputs remain in `combined/`.
8. `results/quantitative_master_overview/` mirrors the latest overview tables under `results/`.

## Validation Evidence

The current frozen behavior was revalidated on 2026-03-11 with:

```bash
uv run pytest tests/unit/eval tests/unit/pipeline/test_quantitative_smoke.py
```

Observed result at validation time: `50 passed`

The validated rerun of the quantitative smoke pipeline was written under:

- `results/mistral-medium-latest_all_abms_chain/06_quantitative_smoke_latest/`
- latest validated run: `runs/run_20260311_131203_114102/`

## Change Policy

Allowed without extra discussion:

- adding tests that preserve the frozen behavior
- documentation updates that clarify the frozen behavior
- operational reruns that reproduce the same contract

Not allowed unless explicitly requested:

- changing metric semantics
- reintroducing lexical-metric fallbacks
- changing reference-family handling or labels
- changing DOE or factorial factor selection
- changing the four-file `overview/` contract
- changing result layout in a way that breaks downstream review
