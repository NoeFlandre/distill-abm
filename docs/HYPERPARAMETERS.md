# Hyperparameters

## LLM Request Defaults (`configs/runtime_defaults.yaml`)

| Parameter | Default |
|---|---|
| `temperature` | `1.0` |
| `max_tokens` | `1000` |
| `max_retries` | `2` |
| `retry_backoff_seconds` | `2.0` |

Provider-specific override:
- `mistral` requests use `temperature=0.2`
- all other providers keep the runtime default `temperature=1.0`

## Run Defaults

| Parameter | Default |
|---|---|
| `provider` | `openrouter` |
| `model` | `moonshotai/kimi-k2.5` |
| `evidence_mode` | `plot+table` |
| `text_source_mode` | `summary_only` |
| `summarizers` | `bart, bert, t5, longformer_ext` |

## Canonical Model Registry (`configs/models.yaml`)

| ID | Provider | Model |
|---|---|---|
| `kimi_k2_5` | `openrouter` | `moonshotai/kimi-k2.5` |
| `gemini_3_1_pro_preview` | `openrouter` | `google/gemini-3.1-pro-preview` |
| `qwen3_5_27b` | `openrouter` | `qwen/qwen3.5-27b` |
| `nemotron_nano_12b_v2_vl_free` | `openrouter` | `nvidia/nemotron-nano-12b-v2-vl:free` |
| `mistral_large_2512` | `mistral` | `mistral-large-2512` |
| `mistral_medium_debug` | `mistral` | `mistral-medium-latest` |

## Summarizer Runtimes

| Summarizer | Model | Defaults |
|---|---|---|
| `bart` | `sshleifer/distilbart-cnn-12-6` | `max_input=1024, min=50, max=100` |
| `bert` | `bert-base-uncased` | `max_input=512, min=100, max=150` |
| `t5` | `t5-small` | `max_input=1024, min=40, max=120` |
| `longformer_ext` | `allenai/led-base-16384` | `max_input=2048, min=64, max=180` |

## ABM Ground Truth Mapping (`configs/experiment_settings.yaml`)

| ABM | Ground Truth File |
|---|---|
| `fauna` | `data/summaries/authors/fauna_scoring_ground_truth.txt` |
| `grazing` | `data/summaries/authors/grazing_scoring_ground_truth.txt` |
| `milk_consumption` | `data/summaries/authors/milk_scoring_ground_truth.txt` |

## GPT-5.2 Ground Truth Mapping (`configs/experiment_settings.yaml`)

| ABM | GPT-5.2_short | GPT-5.2_long |
|---|---|---|
| `fauna` | `data/summaries/gpt5.2/fauna_gpt5.2_short_ground_truth.txt` | `data/summaries/gpt5.2/fauna_gpt5.2_long_ground_truth.txt` |
| `grazing` | `data/summaries/gpt5.2/grazing_gpt5.2_short_ground_truth.txt` | `data/summaries/gpt5.2/grazing_gpt5.2_long_ground_truth.txt` |
| `milk_consumption` | `data/summaries/gpt5.2/milk_gpt5.2_short_ground_truth.txt` | `data/summaries/gpt5.2/milk_gpt5.2_long_ground_truth.txt` |

## Optional Modeler Ground Truth Mapping (`configs/experiment_settings.yaml`)

These references are scored in addition to the primary author reference when present.

| ABM | Modeler Ground Truth File |
|---|---|
| `milk_consumption` | `data/summaries/modelers/milk_modeler_ground_truth.txt` |
