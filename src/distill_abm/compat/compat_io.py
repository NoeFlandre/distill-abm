"""Compatibility helpers for interoperability-oriented file utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from distill_abm.ingest.netlogo import remove_default_elements, remove_urls_from_data
from distill_abm.summarize.postprocess import clean_non_unicode
from distill_abm.summarize.reference_text import process_csv_context as _process_csv_context_refactored


def append_to_csv(
    file_name: str | Path,
    combination_desc: str,
    context_prompt: str,
    context_response: str,
    trend_analysis_responses: list[str],
) -> None:
    _append_row(file_name, [combination_desc, context_prompt, context_response, *trend_analysis_responses])


def append_to_csv2(
    file_name: str | Path,
    combination_desc: str,
    context_prompt: str,
    context_response: str,
    trend_analysis_prompts: list[str],
    trend_analysis_responses: list[str],
) -> None:
    row: list[str] = [combination_desc, context_prompt, context_response]
    for prompt, response in zip(trend_analysis_prompts, trend_analysis_responses, strict=False):
        row.extend([prompt, response])
    _append_row(file_name, row)


def append_analysis_to_csv(file_name: str | Path, values: list[str]) -> None:
    _append_row(file_name, values)


def _append_row(file_name: str | Path, row: list[str]) -> None:
    with Path(file_name).open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(row)


def load_existing_rows(file_name: str | Path) -> list[list[str]]:
    path = Path(file_name)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [row for row in csv.reader(handle)]


def process_documentation(input_json: str | Path, output_json: str | Path) -> None:
    payload = json.loads(Path(input_json).read_text(encoding="utf-8"))
    cleaned = remove_urls_from_data(payload)
    if "documentation" in cleaned:
        cleaned["documentation"] = remove_default_elements(str(cleaned["documentation"]))
    Path(output_json).write_text(json.dumps(cleaned, indent=2), encoding="utf-8")


def process_csv(input_file: str | Path, output_file: str | Path) -> None:
    _process_csv_context_refactored(Path(input_file), Path(output_file))


def remove_non_unicode(text: str) -> str:
    temp = Path("/tmp/.reference_non_unicode_in.csv")
    out = Path("/tmp/.reference_non_unicode_out.csv")
    temp.write_text(text, encoding="utf-8")
    clean_non_unicode(temp, out)
    return out.read_text(encoding="utf-8")


def load_example_files(base_dir: str | Path = "Examples/Text") -> dict[str, str]:
    base = Path(base_dir)
    output: dict[str, str] = {}
    for path in sorted(base.glob("*.txt")):
        output[path.stem] = path.read_text(encoding="utf-8")
    return output


def _normalize_output_csv(path_or_prefix: str | Path) -> Path:
    path = Path(path_or_prefix)
    return path if path.suffix.lower() == ".csv" else path.with_suffix(".csv")


def create_collage(image_paths: list[str | Path], output_path: str | Path, columns: int = 2) -> Path:
    try:
        from PIL import Image
    except Exception:
        Path(output_path).write_text("PIL not available", encoding="utf-8")
        return Path(output_path)
    images = [Image.open(path) for path in image_paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * height), color=(255, 255, 255))
    for index, image in enumerate(images):
        x = (index % columns) * width
        y = (index // columns) * height
        canvas.paste(image, (x, y))
    canvas.save(output_path)
    return Path(output_path)
