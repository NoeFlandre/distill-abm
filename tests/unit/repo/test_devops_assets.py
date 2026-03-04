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
