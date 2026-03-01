"""NetLogo extraction helpers migrated from model notebooks."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_DOCUMENTATION_ELEMENTS: dict[str, str] = {
    "## WHAT IS IT?": "(a general understanding of what the model is trying to show or explain)",
    "## HOW IT WORKS": "(what rules the agents use to create the overall behavior of the model)",
    "## HOW TO USE IT": "(how to use the model, including a description of each of the items in the Interface tab)",
    "## THINGS TO NOTICE": "(suggested things for the user to notice while running the model)",
    "## THINGS TO TRY": "(suggested things for the user to try to do",
    "## EXTENDING THE MODEL": "(suggested things to add or change",
    "## NETLOGO FEATURES": "(interesting or unusual features of NetLogo",
    "## RELATED MODELS": "(models in the NetLogo Models Library",
    "## CREDITS AND REFERENCES": "(what works, ideas, or models this one extends",
}


def extract_parameters(netlogo_code: str) -> dict[str, list[dict[str, Any]]]:
    """Parses GUI parameters so experiments can be reproduced outside NetLogo."""
    parameters: dict[str, list[dict[str, Any]]] = {
        "sliders": _extract_sliders(netlogo_code),
        "switches": _extract_switches(netlogo_code),
        "monitors": _extract_monitors(netlogo_code),
        "buttons": _extract_buttons(netlogo_code),
    }
    return {name: values for name, values in parameters.items() if values}


def _extract_sliders(netlogo_code: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"SLIDER\s+\d+\s+\d+\s+\d+\s+\d+\s+(\S+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        re.MULTILINE,
    )
    return [
        {"name": m.group(1), "min_value": float(m.group(3)), "max_value": float(m.group(4))}
        for m in pattern.finditer(netlogo_code)
    ]


def _extract_switches(netlogo_code: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"SWITCH\s+\d+\s+\d+\s+\d+\s+\d+\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(-?\d+)",
        re.MULTILINE,
    )
    return [
        {
            "name": m.group(1),
            "true_value": bool(int(m.group(3))),
            "false_value": bool(int(m.group(4))),
            "default_value": bool(int(m.group(5))),
        }
        for m in pattern.finditer(netlogo_code)
    ]


def _extract_monitors(netlogo_code: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"MONITOR\s+\d+\s+\d+\s+\d+\s+\d+\s+(\S+)\s+(.*)\s+(\d+)\s+(\d+)\s+(\d+)",
        re.MULTILINE,
    )
    return [
        {
            "name": m.group(1),
            "reporter": m.group(2),
            "x": int(m.group(3)),
            "y": int(m.group(4)),
            "width": int(m.group(5)),
        }
        for m in pattern.finditer(netlogo_code)
    ]


def _extract_buttons(netlogo_code: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"BUTTON\s+\d+\s+\d+\s+\d+\s+\d+\s+(NIL|\S+)\s+(NIL|\S+)\s+(T|F)\s+(T|F)\s+OBSERVER\s+(NIL|\S+)\s+(NIL|\S+)\s+(NIL|\S+)\s+(NIL|\S+)\s+(\d+)",
        re.MULTILINE,
    )
    return [
        {
            "display_name": m.group(1),
            "procedure": m.group(2),
            "forever": m.group(3) == "T",
            "turtle_context": m.group(4) == "T",
            "obs1": m.group(5),
            "obs2": m.group(6),
            "obs3": m.group(7),
            "obs4": m.group(8),
            "state": int(m.group(9)),
        }
        for m in pattern.finditer(netlogo_code)
    ]


def update_parameters(gui_section: list[dict[str, Any]], experiment_parameters: dict[str, Any]) -> None:
    """Injects experiment values into extracted GUI controls for prompt narration."""
    for item in gui_section:
        name = str(item.get("name", ""))
        if name in experiment_parameters:
            item["value"] = experiment_parameters[name]
            del experiment_parameters[name]


def format_json_oneline(data: dict[str, list[dict[str, Any]]]) -> str:
    """Produces notebook-compatible single-line JSON items per section."""
    lines = ["{"]
    for section, items in data.items():
        lines.append(f'  "{section}": [')
        encoded = [json.dumps(item, separators=(",", ": ")) for item in items]
        lines.append("    " + ",\n    ".join(encoded) if encoded else "")
        lines.append("  ],")
    if len(lines) > 1 and lines[-1].endswith(","):
        lines[-1] = lines[-1][:-1]
    lines.append("}")
    return "\n".join(lines)


def extract_documentation(file_path: Path) -> str:
    """Slices model documentation block between NetLogo section delimiters."""
    content = file_path.read_text(encoding="utf-8")
    start_pattern = re.compile(r"@#\$#@#\$#@\s*## WHAT IS IT\?")
    end_pattern = re.compile(r"@#\$#@#\$#@")
    start_match = start_pattern.search(content)
    if not start_match:
        raise ValueError("start of documentation section not found")
    end_match = end_pattern.search(content, start_match.end())
    if not end_match:
        raise ValueError("end of documentation section not found")
    return content[start_match.start() : end_match.start()].strip()


def extract_code(file_path: Path) -> str:
    """Extracts source code section used to infer interface parameters."""
    content = file_path.read_text(encoding="utf-8")
    start_match = re.compile(r"globals\s*\[").search(content)
    end_match = re.compile(r"@#\$#@#\$#@").search(content, start_match.end() if start_match else 0)
    if not start_match or not end_match:
        raise ValueError("code section not found")
    return content[start_match.start() : end_match.start()].strip()


def remove_urls(text: str) -> str:
    """Removes links that inflate prompt tokens without adding model semantics."""
    return re.compile(r"http[s]?://\S+|www\.\S+").sub("", text)


def remove_urls_from_data(data: Any) -> Any:
    """Recursively strips URLs from nested dict/list structures."""
    if isinstance(data, dict):
        return {key: remove_urls_from_data(value) for key, value in data.items()}
    if isinstance(data, list):
        return [remove_urls_from_data(item) for item in data]
    if isinstance(data, str):
        return remove_urls(data)
    return data


def process_documentation_remove_urls(input_json: Path, output_json: Path) -> None:
    """Notebook-equivalent JSON cleaning stage that removes URLs recursively."""
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    cleaned = remove_urls_from_data(payload)
    output_json.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")


def remove_default_elements(text: str, default_elements: Mapping[str, str] | None = None) -> str:
    """Drops untouched NetLogo template sections to keep documentation concise."""
    defaults = default_elements or DEFAULT_DOCUMENTATION_ELEMENTS
    cleaned_sections: list[str] = []
    for section in text.split("\n## "):
        if not section.strip():
            continue
        header, content = _split_section(section)
        default_text = defaults.get(header)
        if default_text and default_text in content.strip():
            continue
        cleaned_sections.append(f"{header}\n\n{content.strip()}")
    return "\n".join(cleaned_sections).strip()


def _split_section(section: str) -> tuple[str, str]:
    if "\n\n" in section:
        raw_header, content = section.split("\n\n", 1)
        return f"## {raw_header.strip()}", content
    return f"## {section.strip()}", ""


def process_documentation_remove_defaults(
    input_json: Path,
    output_json: Path,
    default_elements: Mapping[str, str] | None = None,
) -> None:
    """Removes default template sections and stores cleaned documentation JSON."""
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    if "documentation" in payload:
        payload["documentation"] = remove_default_elements(payload["documentation"], default_elements)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clean_json_content(input_json: Path, output_txt: Path) -> None:
    """Exports plain documentation text exactly like notebook cleanup cells."""
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    documentation = str(payload.get("documentation", ""))
    cleaned = re.sub(r"## @#\$#@#\$#@\n\n\n", "", documentation)
    cleaned = re.sub(r"## .*?\n\n", "", cleaned)
    output_txt.write_text(cleaned, encoding="utf-8")
