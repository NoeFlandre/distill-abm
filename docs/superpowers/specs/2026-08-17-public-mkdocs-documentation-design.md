# Public MkDocs Documentation Design

**Date:** 2026-08-17
**Status:** Approved

## Objective

Make the repository's documentation a concise, professional public reference for `distill-abm` while keeping the root README useful as a durable project landing page. MkDocs Material will be the authoritative rendered documentation surface.

## Audience

- Researchers who want to understand the ABM-to-LLM workflow and inspect published results.
- Contributors who need a reliable local setup, CLI entrypoints, configuration map, and verification commands.
- Reviewers who need clear artifact, reproducibility, and citation boundaries.

## Information architecture

The MkDocs navigation will answer the following questions in order:

1. **Home:** What is this project, what does it produce, and where should a reader start?
2. **Getting started:** How do I install the project and run a safe local validation?
3. **Pipeline and architecture:** What happens from ABM input to scored report, and where are the module boundaries?
4. **CLI reference:** Which commands are intended for normal runs, ingestion, smoke checks, analysis, inspection, and operations?
5. **Configuration:** Which YAML files control models, prompts, ABM presets, runtime defaults, and references?
6. **Results:** Where are published outputs stored, and how can a complete local mirror be synchronized safely?
7. **Development:** How are tests, linting, typing, packaging, and documentation builds verified?
8. **Supplementary material:** Where are the detailed testing and research-support artifacts?
9. **Citation:** How should the software and accompanying manuscript be cited?

Existing stable filenames (`ARCHITECTURE.md`, `CONFIG_REFERENCE.md`, and `RESULTS_BUCKET.md`) will be retained to avoid unnecessary link churn. New pages will use descriptive lowercase names.

## Public surfaces

### Root README

The root README will contain only the material needed to orient a new visitor:

- project purpose and research scope;
- a small overview image;
- supported ABMs and the main pipeline stages;
- installation and non-LLM validation commands;
- links to the rendered documentation, results bucket, paper, license, and citation record.

Detailed command options, configuration tables, artifact contracts, and maintenance procedures will live in MkDocs.

### MkDocs site

MkDocs Material will provide:

- a stable site URL and repository links;
- search, readable code blocks, tables, admonitions, and heading permalinks;
- explicit navigation with no accidental internal pages;
- strict builds so broken references and navigation drift fail CI.

Internal design and implementation-plan notes under `docs/superpowers/` will be excluded from the public build.

### GitHub Pages workflow

`.github/workflows/docs.yml` will build the site on pushes to `main` that affect documentation or its build inputs, and on manual dispatch. The workflow will use least-privilege Pages permissions, build with `mkdocs build --strict`, upload the generated artifact, and deploy through the official Pages actions. Adding the workflow does not itself publish from this local workspace; publication will occur only after the resulting change is committed and pushed.

## Content rules

- Derive commands and option names from the current Typer CLI and repository configuration.
- Separate safe local validation from API-backed, NetLogo-backed, and results-mutating workflows.
- Explain that the Git repository stores source and documentation while the Hugging Face bucket stores generated results.
- Preserve research claims only when they are supported by the repository's current paper-facing material; avoid presenting debug models or smoke workflows as benchmark results.
- Replace machine-specific absolute file links with repository-relative links.
- Keep examples short and runnable; point readers to `--help` for exhaustive option lists.
- Keep supplementary reports available without placing every PDF in the primary navigation.

## Implementation boundaries

Expected changes are limited to documentation and its supporting build surface:

- `README.md` and existing reader-facing pages;
- new MkDocs pages and `mkdocs.yml`;
- the documentation dependency and lockfile entries;
- the Pages workflow;
- repository documentation-contract tests and stale PR-template references where required.

The implementation will preserve unrelated working-tree changes, including the deleted `results/README.md`, the existing `uv.lock` version update, and user-created supplementary material.

## Verification

The completed change will be checked with:

```bash
uv sync --frozen --extra dev
uv run mkdocs build --strict --site-dir /tmp/distill-abm-site
uv run pytest -q
uv run ruff check .
uv run mypy src tests
uv build
```

The generated site will also be inspected for the expected page tree, asset presence, navigation links, and absence of internal planning notes.
