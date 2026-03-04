"""Artifact-focused helpers for NetLogo preprocessing workflows."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from distill_abm.ingest.netlogo import (
    clean_json_content,
    extract_code,
    extract_documentation,
    extract_parameters,
    format_json_oneline,
    process_documentation_remove_defaults,
    process_documentation_remove_urls,
    update_parameters,
)

ParameterScalar = bool | int | float | str


def save_experiment_parameters(
    experiment_parameters: Mapping[str, ParameterScalar],
    output_json_path: Path,
) -> None:
    """Write experiment parameter JSON."""
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(experiment_parameters, indent=4), encoding="utf-8")


def extract_gui_parameters_to_json(model_path: Path, output_json_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Extract GUI parameters from NetLogo code and write JSON output."""
    parameters = extract_parameters(extract_code(model_path))
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(parameters, indent=4), encoding="utf-8")
    return parameters


def update_gui_with_experiment_parameters(
    *,
    gui_parameters_path: Path,
    experiment_parameters_path: Path,
    updated_gui_parameters_path: Path,
    updated_experiment_parameters_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Apply experiment values to GUI controls and write companion artifacts."""
    gui_parameters = json.loads(gui_parameters_path.read_text(encoding="utf-8"))
    experiment_parameters = json.loads(experiment_parameters_path.read_text(encoding="utf-8"))
    for section in ["sliders", "switches", "monitors", "buttons"]:
        if section in gui_parameters:
            update_parameters(gui_parameters[section], experiment_parameters)
    updated_gui_parameters_path.parent.mkdir(parents=True, exist_ok=True)
    updated_gui_parameters_path.write_text(format_json_oneline(gui_parameters), encoding="utf-8")
    updated_experiment_parameters_path.parent.mkdir(parents=True, exist_ok=True)
    updated_experiment_parameters_path.write_text(json.dumps(experiment_parameters, indent=4), encoding="utf-8")
    return gui_parameters, experiment_parameters


def build_parameter_narrative(
    *,
    gui_parameters_path: Path,
    experiment_parameters_path: Path,
    output_text_path: Path,
) -> str:
    """Build simulation-setup narrative from updated GUI and experiment parameters."""
    data_gui = json.loads(gui_parameters_path.read_text(encoding="utf-8"))
    data_experiment = json.loads(experiment_parameters_path.read_text(encoding="utf-8"))
    narrative: list[str] = []
    total_parameters = 0
    for section, items in data_gui.items():
        if section == "monitors":
            continue
        total_parameters += len(items)
        for item in items:
            name = str(item["name"])
            if section == "sliders":
                min_value = item["min_value"]
                max_value = item["max_value"]
                value = item["value"]
                narrative.append(f"  - {name}, from {min_value} to {max_value}. We set it to {value}.")
            elif section == "switches":
                value = item["value"]
                narrative.append(f"  - {name}. We set it to {value}.")
    total_parameters += len(data_experiment)
    for key, value in data_experiment.items():
        narrative.append(f"  - {key}. We set it to {value}.")
    narrative.insert(0, f"We have {total_parameters} parameters:")
    narrative_text = "\n".join(narrative)
    output_text_path.parent.mkdir(parents=True, exist_ok=True)
    output_text_path.write_text(narrative_text, encoding="utf-8")
    return narrative_text


def extract_documentation_to_json(model_path: Path, output_json_path: Path) -> str:
    """Extract NetLogo documentation and write JSON output."""
    documentation = extract_documentation(model_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps({"documentation": documentation}, indent=4), encoding="utf-8")
    return documentation


def extract_code_to_text(model_path: Path, output_text_path: Path) -> str:
    """Extract NetLogo code section and write TXT output."""
    code_section = extract_code(model_path)
    output_text_path.parent.mkdir(parents=True, exist_ok=True)
    output_text_path.write_text(code_section, encoding="utf-8")
    return code_section


def clean_documentation_artifacts(documentation_json: Path, final_text_path: Path) -> None:
    """Run documentation cleanup pipeline."""
    clean_json_content(documentation_json, final_text_path)


def remove_documentation_defaults(raw_json_path: Path, output_json_path: Path) -> None:
    """Removes default-valued strings from cleaned documentation JSON."""
    process_documentation_remove_defaults(raw_json_path, output_json_path)


def remove_documentation_urls(documentation_json_path: Path, output_json_path: Path) -> None:
    """Removes links and URLs from documentation JSON."""
    process_documentation_remove_urls(documentation_json_path, output_json_path)
