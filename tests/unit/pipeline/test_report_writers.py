from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from distill_abm.pipeline.report_writers import write_model_report_files


class _Report(BaseModel):
    ok: bool
    count: int


def test_write_model_report_files_writes_json_and_markdown(tmp_path: Path) -> None:
    report_json_path = tmp_path / "report.json"
    report_markdown_path = tmp_path / "report.md"

    write_model_report_files(
        result=_Report(ok=True, count=3),
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
        markdown="# Report\n",
    )

    assert json.loads(report_json_path.read_text(encoding="utf-8")) == {"ok": True, "count": 3}
    assert report_markdown_path.read_text(encoding="utf-8") == "# Report\n"
