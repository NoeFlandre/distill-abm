# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[Unreleased]: https://github.com/NoeFlandre/distill-abm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/NoeFlandre/distill-abm/releases/tag/v0.1.0
