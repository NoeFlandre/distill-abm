# distill-abm

![Overview of the distill-abm pipeline](docs/assets/overview-readme-v2.png)

`distill-abm` is a reproducible research pipeline that turns agent-based model (ABM) artifacts and simulation outputs into evidence-backed language-model reports. It ingests model context, creates plots and statistical evidence, generates trend narratives, optionally summarizes them, and scores the resulting text against reference material.

The project supports three paper-facing ABMs:

- `fauna` — megafaunal hunting pressure;
- `milk_consumption` — milk adoption;
- `grazing` — pastoral-system resilience.

The repository stores source code, configuration, tests, benchmark inputs, and documentation. Generated outputs are kept in the [Hugging Face results bucket](https://huggingface.co/buckets/NoeFlandre/distill-abms-results).

## Quick start

Requirements: Python 3.11+, [`uv`](https://docs.astral.sh/uv/), and a local checkout.

```bash
uv sync --frozen --extra dev
uv run distill-abm quality-gate --scope static --json
```

To run the pipeline, provide one simulation CSV, one parameter file, and one documentation file:

```bash
uv run distill-abm run \
  --csv-path /path/to/simulation.csv \
  --parameters-path /path/to/parameters.txt \
  --documentation-path /path/to/documentation.txt \
  --model-id kimi_k2_5 \
  --abm fauna
```

API-backed workflows require the credential for the selected provider. NetLogo workflows also require a local NetLogo installation. Do not place credentials in configuration files or commits.

## Documentation

Read the [full documentation](https://noeflandre.github.io/distill-abm/) for installation details, architecture, CLI workflows, configuration, result synchronization, development, and citation.

The source documentation is also available in [`docs/`](docs/README.md). The overview diagram source is [`docs/assets/overview.pdf`](docs/assets/overview.pdf).

## Research context

The associated manuscript, [*Distilling the Complexity of Agent-Based Simulations into Textual Explanations via Large Language Models*](https://www.mdpi.com/2504-2289/10/4/121), evaluates evidence modes, prompt factors, summarizers, and model choices across the three ABMs. The model registry distinguishes paper-facing benchmark models from debug and development models; see the [configuration reference](https://noeflandre.github.io/distill-abm/CONFIG_REFERENCE/).

## Citation and license

If you use this repository, cite the software record in [CITATION.cff](CITATION.cff). The project is released under the [MIT License](LICENSE).
