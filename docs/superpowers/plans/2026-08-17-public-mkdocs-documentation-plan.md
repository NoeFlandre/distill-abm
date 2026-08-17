# Public MkDocs Documentation Implementation Plan

**Design:** `docs/superpowers/specs/2026-08-17-public-mkdocs-documentation-design.md`
**Date:** 2026-08-17

## Goal

Implement the approved public documentation system: a strict MkDocs Material site as the authoritative reference, a concise root README, and a GitHub Pages deployment workflow.

## Constraints

- Preserve the deleted `results/README.md`, the existing `uv.lock` version update, and the user-created supplementary material.
- Do not commit, push, publish, or alter GitHub settings in this task.
- Keep command examples aligned with the current Typer CLI and repository configuration.
- Keep internal planning files under `docs/superpowers/` out of the public site.

## Implementation sequence

### 1. Add documentation contract tests first

Update `tests/unit/repo/test_devops_assets.py` to reflect the current results-pointer contract without restoring `results/README.md`. Add `tests/unit/repo/test_docs_site.py` with focused assertions that:

- `mkdocs.yml` exists and selects the Material theme;
- the navigation includes the approved public sections;
- required public pages and the overview asset exist;
- `pyproject.toml` exposes `mkdocs-material` through the development dependencies;
- `.github/workflows/docs.yml` builds strictly and deploys through Pages actions with least-privilege permissions;
- active public docs contain no machine-specific `/Users/...` links;
- the README points readers to the rendered documentation and preserves the results/citation entrypoints.

Run the new focused tests and confirm they fail for the missing site/build surface before implementation.

### 2. Add the MkDocs build surface

Modify `pyproject.toml` to add a bounded `mkdocs-material` development dependency and regenerate `uv.lock` without disturbing the existing package version update. Add `mkdocs.yml` with:

- site metadata for `NoeFlandre/distill-abm`;
- `site_url`, repository links, and Material theme;
- search, tables, admonitions, code fences, tabbed content where useful, and heading permalinks;
- explicit navigation using `docs/README.md` as Home;
- exclusion of `docs/superpowers/` from the rendered site;
- no undocumented or accidental navigation entries.

Add `.github/workflows/docs.yml` for `main` pushes affecting docs/build inputs and manual dispatch. It will install the frozen development environment, run `uv run mkdocs build --strict`, upload the site artifact, and deploy with Pages actions using repository contents read, Pages write, and OIDC token permissions only.

### 3. Rewrite public content

Rewrite `README.md` as a concise project landing page. Preserve the overview image, research scope, Hugging Face results bucket wording, citation sentence, setup path, and a short set of verified commands. Link to the MkDocs site for detail.

Rewrite the existing stable pages:

- `docs/README.md`: public home page with orientation, quick links, scope, and clear next steps;
- `docs/ARCHITECTURE.md`: end-to-end data flow, module boundaries, artifact lifecycle, and safety boundaries;
- `docs/CONFIG_REFERENCE.md`: configuration map, defaults, model policy, ABM presets, summarizers, and reference families;
- `docs/RESULTS_BUCKET.md`: source/results separation, bucket layout, dry-run-first synchronization, deletion guardrails, and download workflow.

Add concise pages:

- `docs/getting-started.md`: installation, prerequisites, non-LLM validation, local sample/run preparation, and provider/NetLogo boundaries;
- `docs/cli-reference.md`: curated command groups with runnable examples and a pointer to `distill-abm --help` for the full surface;
- `docs/development.md`: repository layout, test/lint/type/build/doc commands, contribution expectations, and conventional commits;
- `docs/supplementary-material.md`: links to the testing report and supplementary PDFs without overwhelming the primary navigation;
- `docs/citation.md`: software citation, manuscript citation, license, and repository links.

Update `docs/supplementary_material/TESTING_REPORT.md` only as needed to remove local absolute paths, correct stale run dates/counts, and point to the new public development/verification page. Preserve its substantive testing evidence and the existing supplementary PDFs.

Update `CITATION.cff` to match the current project release metadata if verification confirms the repository version is `1.0.0`. Remove the obsolete `!results/README.md` exception from `.gitignore` and update its repository test because the results pointer is intentionally absent.

### 4. Refresh adjacent public workflow documentation

Update `.github/PULL_REQUEST_TEMPLATE.md` to remove retired documentation names, reference the MkDocs build, and retain concise validation and reproducibility prompts. Keep the template consistent with the public docs and current repository paths.

### 5. Verify and inspect

Run, using a writable temporary uv cache if the sandbox blocks the default cache:

```bash
UV_CACHE_DIR=/tmp/distill-abm-uv-cache uv sync --frozen --extra dev
UV_CACHE_DIR=/tmp/distill-abm-uv-cache uv run mkdocs build --strict --site-dir /tmp/distill-abm-site
UV_CACHE_DIR=/tmp/distill-abm-uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/distill-abm-uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/distill-abm-uv-cache uv run mypy src tests
UV_CACHE_DIR=/tmp/distill-abm-uv-cache uv build
```

Inspect the generated site tree and HTML for:

- all expected navigation pages;
- the overview image and supplementary links;
- no `docs/superpowers` output;
- no `/Users/noeflandre` links;
- no missing-page or missing-asset warnings under strict build.

Review `git diff --check`, `git diff --stat`, and the complete diff. Report the pre-existing results-pointer test issue separately if it remains after the contract update; do not hide unrelated failures.
