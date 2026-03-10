"""Small shared writers for stable report artifact pairs."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


def write_model_report_files(
    *,
    result: BaseModel,
    report_json_path: Path,
    report_markdown_path: Path,
    markdown: str,
) -> None:
    """Write the standard JSON and Markdown report pair for a Pydantic result."""

    report_json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    report_markdown_path.write_text(markdown, encoding="utf-8")
