from __future__ import annotations

from pathlib import Path

PUBLIC_DOCS = [
    path
    for path in [Path("README.md"), *Path("docs").rglob("*.md")]
    if "superpowers" not in path.parts
]


def test_mkdocs_site_configuration_has_public_navigation() -> None:
    config = Path("mkdocs.yml")
    assert config.exists()
    content = config.read_text(encoding="utf-8")

    for required in [
        "site_name: distill-abm",
        "site_url: https://noeflandre.github.io/distill-abm/",
        "name: material",
        "README.md",
        "getting-started.md",
        "ARCHITECTURE.md",
        "cli-reference.md",
        "CONFIG_REFERENCE.md",
        "RESULTS_BUCKET.md",
        "development.md",
        "supplementary-material.md",
        "citation.md",
        "superpowers",
    ]:
        assert required in content


def test_public_mkdocs_pages_and_dependency_exist() -> None:
    for path in [
        Path("docs/README.md"),
        Path("docs/getting-started.md"),
        Path("docs/ARCHITECTURE.md"),
        Path("docs/cli-reference.md"),
        Path("docs/CONFIG_REFERENCE.md"),
        Path("docs/RESULTS_BUCKET.md"),
        Path("docs/development.md"),
        Path("docs/supplementary-material.md"),
        Path("docs/citation.md"),
        Path("docs/assets/overview-readme-v2.png"),
    ]:
        assert path.exists(), path

    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "mkdocs-material" in pyproject


def test_pages_workflow_builds_and_deploys_with_least_privilege() -> None:
    workflow = Path(".github/workflows/docs.yml")
    assert workflow.exists()
    content = workflow.read_text(encoding="utf-8")

    for required in [
        "workflow_dispatch:",
        "contents: read",
        "pages: write",
        "id-token: write",
        "mkdocs build --strict",
        "actions/upload-pages-artifact",
        "actions/deploy-pages",
    ]:
        assert required in content


def test_public_docs_have_no_machine_specific_absolute_links() -> None:
    violations = [str(path) for path in PUBLIC_DOCS if "/Users/" in path.read_text(encoding="utf-8")]
    assert not violations, ", ".join(violations)


def test_readme_points_to_authoritative_public_docs() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "https://noeflandre.github.io/distill-abm/" in readme
    assert "Hugging Face results bucket" in readme
    assert "If you use this repository, cite the software record in [CITATION.cff](CITATION.cff)." in readme
