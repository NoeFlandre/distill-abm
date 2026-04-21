# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-21

### Added
- Qualitative evaluation CLI for coverage and faithfulness assessment
- Hugging Face results bucket sync (`hf://buckets/NoeFlandre/distill-abms-results`)
- Paper-facing benchmark model stages: screening (`kimi-k2.5`, `qwen3.5-27b`) and optimization (`gemini-3.1-pro-preview`, `claude-opus-4.6`)
- Exploitation factor and LLM same-settings study commands
- CLI optimizations for parallel and chained smoke runs
- Full-case matrix smoke command for comprehensive coverage runs
- Monitor commands for local Qwen runs and generic run observation
- Test coverage for CLI error paths (missing files, analyze-doe failures)
- Test coverage for LLM adapter error handling (missing API keys, completion failures)
- Direct factory tests for `create_adapter` function

### Changed
- Reorganized results into `screening/`, `optimisation/`, and `debug/` folders
- Updated README with overview figure assets and canonical setup instructions
- Refreshed README overview image path to `docs/assets/overview-readme-v2.png`
- Added grazing modeler reference summary to benchmark data
- Renamed `HYPERPARAMETERS.md` to `CONFIG_REFERENCE.md`
- Updated citation year to 2026 in `CITATION.cff`
- Updated RESULTS_BUCKET.md to reflect mirrored directory layout
- Improved test coverage across multiple modules

### Fixed
- Fixed lexical scoring compatibility with latest `rouge-score`
- Fixed DOE ANOVA compatibility with edge-case sample sizes
- Fixed overview path typing in quantitative smoke module
- Fixed pytest regressions in smoke and repo-check tests
- Fixed all remaining mypy strict-mode violations
- Fixed all remaining Ruff lint violations

### Removed
- Removed `EVALUATION_FREEZE.md` and updated all references
- Removed `DECISION_LOG.md` and moved images to `data/images/`
- Removed `scripts/archive_audit.py`, `refresh_parity_artifacts.py`, and other retired archive parity scripts
- Removed `notes/` directory and `.DS_Store` artifacts from source tree
- Removed `configs/prompt_assets/` unused directory
- Removed `data/paper/` from codebase and documentation

## [Unreleased]

### Added
- Test coverage for CLI error paths (missing files, analyze-doe failures)
- Test coverage for LLM adapter error handling (missing API keys, completion failures)
- Direct factory tests for `create_adapter` function
- CHANGELOG.md for version tracking

### Changed
- Improved test coverage across multiple modules

## [0.1.0] - 2025-03-13

### Added
- Initial release of distill-abm
- ABM context extraction from parameters + documentation
- Trend narrative generation from simulation evidence
- Optional summarization (BART, BERT, T5, LongformerExt)
- Lexical scoring (BLEU, METEOR, ROUGE-1/2/L, Flesch)
- DOE/ANOVA analysis over experiment outputs
- NetLogo model ingestion pipeline
- CSV simulation data ingestion
- CLI with run, smoke, analyze-doe, evaluate-qualitative commands

### Fixed
- Various bug fixes and improvements (see git history)

[0.1.0]: https://github.com/NoeFlandre/distill-abm/releases/tag/v0.1.0
