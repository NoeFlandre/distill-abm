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

The current validated behavior is:

1. Lexical metrics must be computed for real. Missing metric dependencies or resources are hard failures, not silent fallbacks.
2. Reference scoring remains additive at the raw-row layer. Author ground truth is the primary contract, while secondary references (`gpt5.2_short`, `gpt5.2_long`, `modeler`) are computed in parallel when configured.
3. Publication/evaluation artifacts are compatibility-filtered:
   - `author`, `modeler`, and `gpt5.2_short` evaluate summarized outputs only (`bart`, `bert`, `t5`, `longformer_ext`)
   - `gpt5.2_long` evaluates the `none` output only
   - factorial analysis excludes `none` entirely and therefore does not include `gpt5.2_long`
4. DOE factor detection must include low-cardinality prompt-factor columns even when pandas stores them as string dtype. `Role`, `Insights`, and `Example` are part of the fitted factorial model when present.
5. Factorial output must distinguish three states:
   - `—` means the term is not estimable or absent from the fitted result
   - `<0.01` means a tiny but nonzero contribution
   - `0.00` means a computed value that rounds to zero at display precision
6. Missing factorial terms must not be backfilled as numeric zero.
7. Factorial feature naming is canonicalized before publication so alias rows do not duplicate or hide real effects.
8. The human-facing `overview/` publication surface contains exactly four files:
   - `anova_table.md`
   - `evidence_summary_table.md`
   - `factorial_table.md`
   - `best_scores_table.md`
9. Each overview table spans all available reference families rather than collapsing them into one unlabeled aggregate table.
10. Higher-precision machine-readable outputs remain in `combined/`, while `overview/` stays compact and publication-oriented.
11. `review.csv`, `quantitative_rows.csv`, `structured_results.csv`, and per-record bundles remain the full scored cross-product so previously computed metrics can be reused without rescoring.

## Validation Evidence

The current frozen behavior was revalidated on 2026-03-11 with:

```bash
uv run pytest tests/unit/eval tests/unit/pipeline/test_quantitative_smoke.py
```

Observed result at validation time:

- `50 passed`

The validated rerun of the quantitative smoke pipeline was written under:

- `results/mistral-medium-latest_all_abms_chain/06_quantitative_smoke_latest/`
- latest validated run: `runs/run_20260311_131203_114102/`

## Change Policy

Allowed without extra discussion:

- adding tests that preserve the frozen behavior
- documentation updates that describe the frozen behavior more clearly
- operational reruns that reproduce the same contract

Not allowed unless explicitly requested:

- changing metric semantics
- reintroducing fallbacks for missing lexical metrics
- changing how references are combined or labeled
- changing which factors enter the DOE/factorial model
- changing how absent versus tiny factorial effects are rendered
- changing the four-file `overview/` contract
- changing result file naming or layout in a way that breaks downstream review
