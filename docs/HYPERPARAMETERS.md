# Hyperparameters

## LLM Request Defaults (`configs/runtime_defaults.yaml`)

| Parameter | Default |
|---|---|
| `temperature` | `0.5` |
| `max_tokens` | `1000` |
| `max_retries` | `2` |
| `retry_backoff_seconds` | `2.0` |

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
| `qwen3_5_local` | `ollama` | `qwen3.5:0.8b` |
| `qwen_vl_debug` | `openrouter` | `qwen/qwen3-vl-235b-a22b-thinking` |

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
| `fauna` | `configs/ground_truth/fauna_scoring_ground_truth.txt` |
| `grazing` | `configs/ground_truth/grazing_scoring_ground_truth.txt` |
| `milk_consumption` | `configs/ground_truth/milk_scoring_ground_truth.txt` |
