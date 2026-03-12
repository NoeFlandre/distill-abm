# DOE smoke layout

Use this directory in three passes:

1. `00_overview/` for the global report and matrix CSVs.
2. `10_shared/` for global DOE factors and ABM-level shared inputs, evidence, and prompts.
3. `20_case_index/` for compact case and request indexes.

## 00_overview

- `doe_smoke_report.md`
- `doe_smoke_report.json`
- `design_matrix.csv`
- `request_matrix.csv`

- `request_review.csv`

## 10_shared/global

- `models.json`
- `summarization_modes.json`
- `prompt_variants.json`
- `evidence_modes.json`

## 10_shared/<abm>

- `01_inputs/`: copied simulation CSV, parameters narrative, final documentation
- `02_evidence/plots/`: copied plot images used by the DOE smoke
- `02_evidence/tables/`: statistical evidence text, JSON payloads, and reduced source series per plot
- `03_prompts/context/`: shared context prompts by prompt variant
- `03_prompts/trend/<evidence_mode>/<prompt_variant>/`: per-plot trend prompts

## 20_case_index

- `cases.jsonl`: one compact JSON object per DOE case
- `requests.jsonl`: one compact JSON object per planned request

Use the matrix CSVs first. Use the JSONL indexes only when you need richer case detail without opening hundreds of files.

Generated at: `results/mistral_all_abms_chain/03_doe_smoke_latest/runs/run_20260309_104629_847610`
