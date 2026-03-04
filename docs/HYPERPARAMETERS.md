# Hyperparameters and Defaults

This is the publication-facing reference for explicit runtime defaults and hyperparameters.
If a value is not listed here, it is either required input data or provider-side behavior outside this repository.

Central source of runtime defaults:

- `configs/runtime_defaults.yaml`

## 1. Core LLM request defaults

Defined in `src/distill_abm/llm/adapters/base.py` (`LLMRequest`):

| Field | Default |
| --- | --- |
| `temperature` | `0.5` |
| `max_tokens` | `1000` |
| `max_retries` | `2` |
| `retry_backoff_seconds` | `2.0` |
| `image_b64` | `null` |
| `metadata` | `{}` |

These defaults apply unless explicitly overridden by callers.

## 2. CLI defaults

Defined in `src/distill_abm/cli.py`.

### `distill-abm run`

| Option | Default |
| --- | --- |
| `provider` | `"echo"` |
| `model` | `"echo-model"` |
| `metric_pattern` | `"mean"` |
| `metric_description` | `"simulation trend"` |
| `evidence_mode` | `"plot"` |
| `skip_summarization` | `false` |
| `summarization_mode` | `"both"` |
| `score_on` | `"both"` |
| `additional_summarizer` | `[]` |

### `distill-abm analyze-doe`

| Option | Default |
| --- | --- |
| `output_csv` | `results/doe/anova_factorial_contributions.csv` |
| `max_interaction_order` | `2` |

### `distill-abm evaluate-qualitative`

| Option | Default |
| --- | --- |
| `prompts_path` | `configs/prompts.yaml` |
| `provider` | `"echo"` |
| `model` | `"echo-model"` |
| `source_image_path` | `null` |

### `distill-abm smoke-qwen`

| Option | Default |
| --- | --- |
| `provider` | `"ollama"` |
| `output_dir` | `results/smoke_qwen` |
| `model` | `qwen3.5:0.8b` |
| `metric_pattern` | `"mean"` |
| `metric_description` | `"simulation trend"` |
| `plot_description` | `null` |
| `run_qualitative` | `true` (`--skip-qualitative` disables) |
| `run_sweep` | `true` (`--skip-sweep` disables) |
| `doe_input_csv` | `null` (optional) |
| `profile` | `"matrix"` (`"three-branches"` available) |
| `case_id` | `[]` (optional filter) |
| `max_cases` | `null` (optional cap) |

## 3. Evidence mode values

Canonical values used by runtime:

- `plot` (plot image only)
- `table-csv` (stats table as CSV text only)
- `plot+table` (plot image + stats CSV text)

Accepted aliases (normalized internally):

- `stats-markdown` -> `table-csv`
- `stats-image` -> `table-csv`
- `plot+stats` -> `plot+table`

## 4. Summarization defaults

Defined in `src/distill_abm/summarize/models.py` and recorded in `pipeline_run_metadata.json`.

### Always-on summarizers

| Backend | Model/runtime | `max_input_length` | `min_summary_length` | `max_summary_length` |
| --- | --- | ---: | ---: | ---: |
| BART | `sshleifer/distilbart-cnn-12-6` | 1024 | 50 | 100 |
| BERT extractive | `bert-base-uncased` tokenizer + `summarizer.bert.Summarizer` | 512 | 100 | 150 |

### Optional additional summarizers

| CLI value | Backend | Model | `max_input_length` | `min_summary_length` | `max_summary_length` |
| --- | --- | --- | ---: | ---: | ---: |
| `t5` | T5 abstractive | `t5-small` | 1024 | 40 | 120 |
| `longformer_ext` | Long-document abstractive | `allenai/led-base-16384` | 2048 | 64 | 180 |

## 5. Scoring defaults and ranges

### Quantitative scores

`src/distill_abm/eval/metrics.py` computes:

- token precision/recall/F1
- BLEU
- METEOR
- ROUGE-1 / ROUGE-2 / ROUGE-L
- Flesch reading ease

### Qualitative scores

Prompt contract in `configs/prompts.yaml`:

- Coverage score range: `1` to `5`
- Faithfulness score range: `1` to `5`

## 6. Model alias defaults

Defined in `configs/models.yaml`.

| Alias | Provider | Model |
| --- | --- | --- |
| `gpt4o` | `openai` | `gpt-4o` |
| `claude_sonnet` | `anthropic` | `claude-3-sonnet-20240229` |
| `qwen_openrouter` | `openrouter` | `qwen/qwen3-vl-235b-a22b-thinking` |
| `deepseek_r1` | `ollama` | `deepseek-r1` |
| `qwen_ollama` | `ollama` | `qwen3.5:0.8b` |
| `janus_pro` | `janus` | `janus-pro` |

For Ollama adapter calls:

- `temperature` is passed via `options.temperature`
- `max_tokens` is passed via `options.num_predict`

## 7. Experiment settings defaults

Defined in `configs/notebook_experiment_settings.yaml` (historical filename retained) and typed in `src/distill_abm/configs/models.py`.

### LLM defaults

| Field | Value |
| --- | --- |
| `openai_model` | `gpt-4o` |
| `anthropic_model` | `claude-3-5-sonnet-20241022` |
| `max_tokens` | `1000` |
| `temperature` | `0.5` |

### DoE defaults

| Field | Value |
| --- | --- |
| `repetitions` | `3` |
| `max_interaction_order` | `2` |

### NetLogo run defaults

Applied by each ABM ingestion section unless overridden:

| Field | Value |
| --- | ---: |
| `num_runs` | `40` |
| `max_ticks` | `73000` |
| `interval` | `50` |

### Summary-generation plot counts

| ABM | `num_plots` |
| --- | ---: |
| Fauna | `14` |
| Grazing | `10` |
| Milk consumption | `12` |

## 8. ABM default plotting descriptors

Defined in `configs/abms/*.yaml`.

| ABM | `metric_pattern` | `metric_description` | `plot_descriptions` count |
| --- | --- | --- | ---: |
| Fauna | `count-species` | `species abundance dynamics across repeated fauna simulations` | 14 |
| Grazing | `household-risk-att-init` | `grazing pressure and vegetation regeneration dynamics` | 10 |
| Milk consumption | `mean-incum` | `average weekly whole milk consumption per agent` | 12 |

## 9. Visualization defaults

Defined in `src/distill_abm/viz/plots.py`.

| Setting | Value |
| --- | --- |
| Figure size | `(10, 6)` |
| Plot DPI | `150` |
| Per-series alpha | `0.25` |
| Per-series line width | `1.0` |
| Mean-line width | `2.0` |
| Mean-line default | `enabled` |
| Markdown table precision | `4` decimals |

## 10. Sweep defaults

Defined in `src/distill_abm/pipeline/run.py`.

| Setting | Default |
| --- | --- |
| `style_feature_keys` | `["role", "example", "insights"]` |
| `csv_column_style` | `"trend"` |
| `resume_existing` | `false` |

## 11. Evaluation config defaults

Defined in `configs/evaluation.yaml`.

| Field | Value |
| --- | --- |
| `use_reference_metrics` | `true` |
| `use_token_f1` | `true` |

## 12. Quality thresholds

| Gate | Threshold |
| --- | --- |
| Coverage | `>= 85%` |
| Formatting | `black --check .` |
| Linting | `ruff check .` |
| Typing | `mypy src tests` |

## 13. Reproducibility metadata

Each pipeline run writes `pipeline_run_metadata.json` including:

- requested and resolved runtime modes
- provider/model and request hyperparameters
- request-level image attachment flags for context and trend phases
- full context/trend prompt texts
- prompt signatures (`sha256`)
- summarizer enablement and numeric settings
- score outputs and artifact paths
- scoring reference details (`scores.reference.source/path/signature/length`)

When running with `--abm`, the scoring reference file is selected from `configs/notebook_experiment_settings.yaml`:

- `fauna` -> `scoring.fauna_ground_truth_path`
- `grazing` -> `scoring.grazing_ground_truth_path`
- `milk_consumption` -> `scoring.milk_ground_truth_path`
