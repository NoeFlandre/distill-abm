from __future__ import annotations

from pathlib import Path

from distill_abm.pipeline.smoke_io import copy_if_exists, dedupe_rows, read_csv_rows, write_csv_rows


def test_write_and_read_csv_rows_round_trip(tmp_path: Path) -> None:
    path = tmp_path / 'rows.csv'
    rows = [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]

    write_csv_rows(path, rows, ("a", "b"))

    assert read_csv_rows(path) == rows


def test_dedupe_rows_keeps_first_unique_key(tmp_path: Path) -> None:
    rows = [
        {
            "run_output_dir": "r",
            "case_id": "c",
            "response_kind": "context",
            "prompt_signature": "p",
            "response_path": "x",
            "value": "1",
        },
        {
            "run_output_dir": "r",
            "case_id": "c",
            "response_kind": "context",
            "prompt_signature": "p",
            "response_path": "x",
            "value": "2",
        },
        {
            "run_output_dir": "r",
            "case_id": "c",
            "response_kind": "trend",
            "prompt_signature": "p2",
            "response_path": "y",
            "value": "3",
        },
    ]

    deduped = dedupe_rows(rows)

    assert deduped == [rows[0], rows[2]]


def test_copy_if_exists_copies_existing_file(tmp_path: Path) -> None:
    source = tmp_path / 'source.txt'
    destination = tmp_path / 'nested' / 'dest.txt'
    source.write_text('content', encoding='utf-8')

    copy_if_exists(source, destination)

    assert destination.read_text(encoding='utf-8') == 'content'


def test_copy_if_exists_ignores_missing_file(tmp_path: Path) -> None:
    destination = tmp_path / 'nested' / 'dest.txt'

    copy_if_exists(tmp_path / 'missing.txt', destination)

    assert not destination.exists()
