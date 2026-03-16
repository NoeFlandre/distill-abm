from __future__ import annotations

from pathlib import Path


def test_dockerfile_exists_and_uses_reproducible_python_env() -> None:
    dockerfile = Path("Dockerfile")
    assert dockerfile.exists()
    content = dockerfile.read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in content
    assert "WORKDIR /app" in content
    assert "uv sync --frozen --extra dev" in content


def test_dockerignore_excludes_local_artifacts() -> None:
    dockerignore = Path(".dockerignore")
    assert dockerignore.exists()
    entries = dockerignore.read_text(encoding="utf-8")
    for required in [
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "results",
        "Results",
    ]:
        assert required in entries


def test_ci_workflow_runs_quality_gates() -> None:
    workflow = Path(".github/workflows/ci.yml")
    assert workflow.exists()
    content = workflow.read_text(encoding="utf-8")
    assert "ruff check ." in content
    assert "mypy src tests" in content
    assert "pytest" in content
    assert "uv build" in content


def test_gitignore_keeps_results_repo_lightweight() -> None:
    gitignore = Path(".gitignore")
    content = gitignore.read_text(encoding="utf-8")
    assert "results/*" in content
    assert "!results/README.md" in content
    for retired_entry in [
        "!results/archive/**",
        "!results/quantitative_master_overview/**",
        "!results/kimi-k2.5_all_abms_chain/**",
        "!results/eval_qwen_kimi/**",
        "!results/eval_mistral_kimi/**",
        "!results/eval_qwen_mistral/**",
        "!results/eval_qwen_mistral_kimi/**",
        "!results/mistral-medium-latest_all_abms_chain/**",
        "!results/qwen3.5-27b_openrouter_all_abms_chain/**",
    ]:
        assert retired_entry not in content


def test_public_docs_surface_matches_publication_contract() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    docs_index = Path("docs/README.md").read_text(encoding="utf-8")
    results_bucket = Path("docs/RESULTS_BUCKET.md").read_text(encoding="utf-8")
    results_readme = Path("results/README.md").read_text(encoding="utf-8")
    citation = Path("CITATION.cff").read_text(encoding="utf-8")

    assert "Published run outputs live in the Hugging Face results bucket, not in Git." in readme
    assert "If you use this repository, cite the software record in [CITATION.cff](CITATION.cff)." in readme
    assert "The Git repository is publication-facing source code." in results_bucket
    assert "uv run distill-abm sync-results-bucket --dry-run" in results_bucket
    assert "hf sync --apply /tmp/distill_abm_bucket_cleanup_plan.jsonl" in results_bucket
    assert "hf sync --apply /tmp/distill_abm_bucket_cleanup_plan.jsonl" in results_readme
    assert "ARCHITECTURE.md" in docs_index
    assert "RESULTS_BUCKET.md" in docs_index
    assert "HYPERPARAMETERS.md" in docs_index
    assert "EVALUATION_FREEZE.md" in docs_index
    assert "DECISION_LOG.md" in docs_index
    assert "RUN_EXECUTION_ORDER.md" not in docs_index
    assert "WALKTHROUGH.md" not in docs_index
    assert "FAILURE_SEMANTICS.md" not in docs_index
    assert "MANUAL_VALIDATION.md" not in docs_index
    assert "TRACEABILITY_MATRIX.md" not in docs_index
    assert 'title: "distill-abm"' in citation
    assert "preferred-citation:" in citation


def test_stale_root_level_supplementary_docs_are_absent() -> None:
    assert not Path("docs/TESTING_REPORT.md").exists()
    assert not Path("docs/GROUND_TRUTHS_GPT5.2.pdf").exists()
    assert Path("docs/supplementary_material/TESTING_REPORT.md").exists()
    assert Path("docs/supplementary_material/GROUND_TRUTHS_GPT5.2.pdf").exists()


def test_retired_low_value_docs_are_absent() -> None:
    for retired_path in [
        "docs/RUN_EXECUTION_ORDER.md",
        "docs/WALKTHROUGH.md",
        "docs/FAILURE_SEMANTICS.md",
        "docs/MANUAL_VALIDATION.md",
        "docs/TRACEABILITY_MATRIX.md",
    ]:
        assert not Path(retired_path).exists()


def test_unused_doc_assets_are_absent() -> None:
    assert not Path("docs/assets/plot_example.png").exists()
    assert not Path("docs/assets/stats_table_example.png").exists()
    assert not Path("docs/images/.DS_Store").exists()
