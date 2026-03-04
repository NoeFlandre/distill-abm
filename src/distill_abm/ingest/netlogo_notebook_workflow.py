"""Notebook-equivalent NetLogo ingestion workflow helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd

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


class NetLogoLinkProtocol(Protocol):
    def load_model(self, model_path: str) -> None: ...

    def command(self, command: str) -> None: ...

    def report(self, reporter: str) -> Any: ...


def _default_link_factory(*, netlogo_home: str) -> NetLogoLinkProtocol:
    try:
        import pynetlogo
    except Exception as exc:  # pragma: no cover - exercised via injected factory in tests
        raise RuntimeError("pynetlogo is required to run NetLogo experiments") from exc
    return cast(NetLogoLinkProtocol, pynetlogo.NetLogoLink(netlogo_home=netlogo_home))


def run_netlogo_experiment(
    *,
    netlogo_home: str,
    model_path: str | Path,
    experiment_parameters: Mapping[str, ParameterScalar],
    reporters: Sequence[str],
    num_runs: int,
    max_ticks: int,
    interval: int,
    output_csv_path: Path,
    netlogo_link_factory: Callable[..., NetLogoLinkProtocol] | None = None,
) -> pd.DataFrame:
    """Runs notebook-style NetLogo repetition loops and writes a combined CSV."""
    factory = netlogo_link_factory or _default_link_factory
    netlogo = factory(netlogo_home=netlogo_home)
    netlogo.load_model(str(model_path))
    all_results = pd.DataFrame()

    for _run in range(num_runs):
        for param, value in experiment_parameters.items():
            output_value: ParameterScalar = value
            if isinstance(value, bool):
                output_value = "true" if value else "false"
            netlogo.command(f"set {param} {output_value}")
        netlogo.command("setup")

        run_results: list[dict[str, Any]] = []
        for tick in range(0, max_ticks, interval):
            netlogo.command("go")
            tick_data: dict[str, Any] = {reporter: netlogo.report(reporter) for reporter in reporters}
            tick_data["tick"] = tick
            run_results.append(tick_data)
        run_df = pd.DataFrame(run_results)
        run_df.columns = [f"{col}" for col in run_df.columns]
        if all_results.empty:
            all_results = run_df
        else:
            all_results = pd.concat([all_results, run_df], axis=1)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(output_csv_path, index=False)
    return all_results


def save_experiment_parameters(experiment_parameters: Mapping[str, ParameterScalar], output_json_path: Path) -> None:
    """Writes notebook-style experiment parameter JSON."""
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(dict(experiment_parameters), indent=4), encoding="utf-8")


def extract_gui_parameters_to_json(model_path: Path, output_json_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Extracts GUI parameters from NetLogo code and writes JSON output."""
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
    """Applies experiment values to GUI controls and writes both notebook outputs."""
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
    """Builds the notebook narrative text from updated GUI and experiment parameters."""
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
    """Extracts NetLogo documentation and writes notebook-style JSON output."""
    documentation = extract_documentation(model_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps({"documentation": documentation}, indent=4), encoding="utf-8")
    return documentation


def extract_code_to_text(model_path: Path, output_text_path: Path) -> str:
    """Extracts NetLogo code section and writes notebook-style TXT output."""
    code_section = extract_code(model_path)
    output_text_path.parent.mkdir(parents=True, exist_ok=True)
    output_text_path.write_text(code_section, encoding="utf-8")
    return code_section


def run_notebook_ingest_workflow(
    *,
    model_path: Path,
    experiment_parameters: Mapping[str, ParameterScalar],
    output_dir: Path,
    suffix: str = "100",
) -> dict[str, Path]:
    """Runs notebook-equivalent preprocessing workflow and returns artifact paths."""
    json_dir = output_dir / "JSON"
    txt_dir = output_dir / "TXT"
    experiment_parameters_json = json_dir / f"experiment_parameters{suffix}.json"
    gui_parameters_json = json_dir / f"gui_parameters{suffix}.json"
    updated_gui_parameters_json = json_dir / f"updated_gui_parameters{suffix}.json"
    updated_experiment_parameters_json = json_dir / f"updated_experiment_parameters{suffix}.json"
    narrative_txt = txt_dir / f"narrativeCombined{suffix}.txt"
    documentation_json = json_dir / f"documentation{suffix}.json"
    cleaned_documentation_json = json_dir / f"cleaneddocumentation{suffix}.json"
    documentation_without_default_json = json_dir / f"documentationWithoutDefault{suffix}.json"
    final_documentation_txt = txt_dir / f"finalDocumentation{suffix}.txt"
    extracted_code_txt = txt_dir / f"extracted_code{suffix}.txt"

    save_experiment_parameters(experiment_parameters, experiment_parameters_json)
    extract_gui_parameters_to_json(model_path, gui_parameters_json)
    update_gui_with_experiment_parameters(
        gui_parameters_path=gui_parameters_json,
        experiment_parameters_path=experiment_parameters_json,
        updated_gui_parameters_path=updated_gui_parameters_json,
        updated_experiment_parameters_path=updated_experiment_parameters_json,
    )
    build_parameter_narrative(
        gui_parameters_path=updated_gui_parameters_json,
        experiment_parameters_path=updated_experiment_parameters_json,
        output_text_path=narrative_txt,
    )
    extract_documentation_to_json(model_path, documentation_json)
    process_documentation_remove_urls(documentation_json, cleaned_documentation_json)
    process_documentation_remove_defaults(cleaned_documentation_json, documentation_without_default_json)
    clean_json_content(documentation_without_default_json, final_documentation_txt)
    extract_code_to_text(model_path, extracted_code_txt)

    return {
        "experiment_parameters_json": experiment_parameters_json,
        "gui_parameters_json": gui_parameters_json,
        "updated_gui_parameters_json": updated_gui_parameters_json,
        "updated_experiment_parameters_json": updated_experiment_parameters_json,
        "narrative_txt": narrative_txt,
        "documentation_json": documentation_json,
        "cleaned_documentation_json": cleaned_documentation_json,
        "documentation_without_default_json": documentation_without_default_json,
        "final_documentation_txt": final_documentation_txt,
        "extracted_code_txt": extracted_code_txt,
    }
