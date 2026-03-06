from __future__ import annotations

from pathlib import Path

from distill_abm.cli_models import ArtifactDescriptor, DescribeArtifactsResult, SmokeCommandResult


def test_cli_models_module_exposes_stable_artifact_descriptor() -> None:
    descriptor = ArtifactDescriptor(path=Path("artifact.txt"), exists=False)

    assert descriptor.path == Path("artifact.txt")
    assert descriptor.exists is False
    assert descriptor.size_bytes == 0
    assert descriptor.sha256 is None


def test_cli_models_module_exposes_read_only_result_shapes() -> None:
    result = DescribeArtifactsResult(root=Path("results"), artifacts={})

    assert result.root == Path("results")
    assert result.artifacts == {}


def test_cli_models_module_exposes_smoke_result_defaults() -> None:
    result = SmokeCommandResult(
        command="smoke-viz",
        success=True,
        report_json_path=Path("report.json"),
        report_markdown_path=Path("report.md"),
    )

    assert result.failed_items == []
    assert result.nested_artifacts == {}
