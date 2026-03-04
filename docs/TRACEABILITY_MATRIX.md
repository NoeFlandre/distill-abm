# Traceability Matrix

| ID | Paper Requirement | Runtime Location | Verification |
|---|---|---|---|
| TM-001 | Extract ABM context from parameters/docs | `src/distill_abm/pipeline/helpers.py` | `tests/integration/test_pipeline_run.py` |
| TM-002 | Generate trend report with evidence attachment(s) | `src/distill_abm/pipeline/run.py` | `tests/integration/test_pipeline_run.py` |
| TM-003 | Prompt factor combinations (`role/example/insights`) | `src/distill_abm/pipeline/run.py` | `tests/unit/pipeline/test_sweep.py` |
| TM-004 | Summarization stage | `src/distill_abm/summarize/models.py` + `src/distill_abm/pipeline/run.py` | `tests/unit/summarize/*` |
| TM-005 | Lexical evaluation (BLEU/METEOR/ROUGE/Flesch) | `src/distill_abm/eval/metrics.py` | `tests/unit/eval/test_metrics*.py` |
| TM-006 | DOE/ANOVA factorial analysis | `src/distill_abm/eval/doe_full.py` | `tests/unit/eval/test_doe_full.py` |
| TM-007 | Reproducibility metadata + signatures | `src/distill_abm/pipeline/run.py` | `tests/integration/test_pipeline_run.py` |
| TM-008 | Evidence ablations (`plot`, `table`, `plot+table`) | `src/distill_abm/pipeline/helpers.py` + `src/distill_abm/pipeline/smoke.py` | `tests/unit/pipeline/test_smoke.py` |
| TM-009 | Text-source ablations (`summary_only`, `full_text_only`) | `src/distill_abm/pipeline/run.py` + CLI | `tests/e2e/test_cli.py` |
| TM-010 | Benchmark model roster enforcement | `src/distill_abm/cli.py` | `tests/e2e/test_cli.py` |
