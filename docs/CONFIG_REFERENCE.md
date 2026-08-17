# Configuration reference

Configuration is YAML-backed and validated with Pydantic models before it reaches the pipeline. Paths are resolved relative to the repository unless a command option supplies an explicit alternative.

## Configuration files

| File | Responsibility |
| --- | --- |
| `configs/models.yaml` | Provider adapters, model aliases, and model identifiers. |
| `configs/runtime_defaults.yaml` | LLM request defaults, pipeline defaults, smoke defaults, and DOE output paths. |
| `configs/prompts.yaml` | Context, trend, summarization, qualitative-evaluation, and prompt-feature templates. |
| `configs/experiment_settings.yaml` | Reference-text families and scoring paths for each ABM. |
| `configs/evaluation.yaml` | Reference-metric and token-F1 switches. |
| `configs/logging.yaml` | Structured logging behavior. |
| `configs/abms/*.yaml` | ABM names, metrics, plot descriptions, NetLogo reporters, and fallback assets. |

Use the CLI's path options when experimenting with a copy. Do not edit the tracked defaults merely to provide local credentials or machine-specific paths.

## Runtime defaults

The current defaults in `configs/runtime_defaults.yaml` are:

| Setting | Value |
| --- | --- |
| Provider | `openrouter` |
| Default model | `moonshotai/kimi-k2.5` |
| Evidence mode | `plot+table` |
| Text source mode | `summary_only` |
| Summarizers | `bart`, `bert`, `t5`, `longformer_ext` |
| Temperature | `1.0` |
| Maximum output tokens | `1000` |
| Retries | `2` |
| Retry backoff | `2.0` seconds |

These are runtime defaults, not a guarantee that every paper-facing run used the same factors. Record the effective settings in the generated metadata when reproducing a result.

## Model registry and policy

The registry uses a stable alias for each provider/model pair:

| Alias | Provider | Model | Intended use |
| --- | --- | --- | --- |
| `kimi_k2_5` | OpenRouter | `moonshotai/kimi-k2.5` | Screening |
| `qwen3_5_27b` | OpenRouter | `qwen/qwen3.5-27b` | Screening |
| `gemini_3_1_pro_preview` | OpenRouter | `google/gemini-3.1-pro-preview` | Optimization |
| `claude_opus_4_6` | OpenRouter | `anthropic/claude-opus-4.6` | Optimization |
| `nemotron_nano_12b_v2_vl_free` | OpenRouter | `nvidia/nemotron-nano-12b-v2-vl:free` | Debug/development |
| `mistral_large_2512` | Mistral | `mistral-large-2512` | Debug/development |
| `mistral_medium_debug` | Mistral | `mistral-medium-latest` | Debug/development |
| `mistral_small_2506` | Mistral | `mistral-small-2506` | Debug/development |
| `ministral_14b_2512` | Mistral | `ministral-14b-2512` | Debug/development |
| `magistral_medium_2509` | Mistral | `magistral-medium-2509` | Debug/development |

The CLI accepts a registry alias through `--model-id`. Benchmark-oriented commands reject debug models unless `--allow-debug-model` is explicitly supplied. Treat that flag as a development escape hatch, not as a benchmark configuration.

## ABM presets

Each `configs/abms/<name>.yaml` file defines the metric pattern, metric description, plot descriptions, NetLogo experiment settings, reporters, plot ordering, and fallback CSV/plot assets. The current presets are:

| ABM | Primary signal |
| --- | --- |
| `fauna` | Species-abundance dynamics across repeated fauna simulations. |
| `grazing` | Grazing pressure and vegetation-regeneration dynamics. |
| `milk_consumption` | Milk-adoption dynamics. |

Use `distill-abm describe-abm --help` to inspect the command options for resolving one preset without running inference.

## Summarizers

The optional `summarizers` dependency installs the local transformer runtimes used by these names:

| Name | Runtime model | Typical role |
| --- | --- | --- |
| `bart` | `sshleifer/distilbart-cnn-12-6` | Abstractive summary. |
| `bert` | `bert-base-uncased` | Extractive summary. |
| `t5` | `t5-small` | Abstractive summary. |
| `longformer_ext` | `allenai/led-base-16384` | Long-context extractive-style summary. |

Install the optional runtime only when you need it:

```bash
uv sync --frozen --extra dev --extra summarizers
```

## Reference families

`configs/experiment_settings.yaml` maps each ABM to:

- author-written scoring references;
- independent modeler references;
- GPT-5.2 short references;
- GPT-5.2 long references.

Reference files live under `data/summaries/`. Keep them immutable when reproducing a published comparison; change the configuration explicitly if you are running a new study.

## Provider credentials

The adapters read credentials from environment variables:

| Provider | Environment variable |
| --- | --- |
| OpenRouter | `OPENROUTER_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |

Credentials should be supplied by the shell or a secret manager. They should not be placed in YAML, source files, result artifacts, or issue comments.
