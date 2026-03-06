"""NetLogo ingestion workflow helpers."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from distill_abm.ingest.netlogo import (
    extract_experiment_parameters as _extract_experiment_parameters,
)
from distill_abm.ingest.netlogo import (
    find_reference_narrative_path,
)
from distill_abm.ingest.netlogo_artifacts import (
    build_parameter_narrative as _build_parameter_narrative,
)
from distill_abm.ingest.netlogo_artifacts import (
    clean_documentation_artifacts as _clean_documentation_artifacts,
)
from distill_abm.ingest.netlogo_artifacts import (
    extract_code_to_text as _extract_code_to_text,
)
from distill_abm.ingest.netlogo_artifacts import (
    extract_documentation_to_json as _extract_documentation_to_json,
)
from distill_abm.ingest.netlogo_artifacts import (
    extract_gui_parameters_to_json as _extract_gui_parameters_to_json,
)
from distill_abm.ingest.netlogo_artifacts import (
    remove_documentation_defaults as _remove_documentation_defaults,
)
from distill_abm.ingest.netlogo_artifacts import (
    remove_documentation_urls as _remove_documentation_urls,
)
from distill_abm.ingest.netlogo_artifacts import (
    save_experiment_parameters as _save_experiment_parameters,
)
from distill_abm.ingest.netlogo_artifacts import (
    update_gui_with_experiment_parameters as _update_gui_with_experiment_parameters,
)
from distill_abm.ingest.netlogo_steps import (
    NetLogoLinkProtocol,
    coerce_parameter_value_for_netlogo,
)
from distill_abm.ingest.netlogo_steps import (
    run_single_repetition as _run_single_repetition,
)

ABM_PREFERRED_EXPERIMENTS: dict[str, str] = {
    "fauna": "2023-20-6-0.33-0.8",
    "grazing": "experiment",
    "milk_consumption": "Milk Consumption Trends",
}

ParameterScalar = bool | int | float | str


def _artifact_name_map(suffix: str) -> dict[str, str]:
    """Return canonical artifact filenames for a given suffix."""
    if not suffix:
        return {
            "experiment_parameters_json": "experiment_parameters.json",
            "gui_parameters_json": "gui_parameters.json",
            "updated_gui_parameters_json": "updated_gui_parameters.json",
            "updated_experiment_parameters_json": "updated_experiment_parameters.json",
            "narrative_txt": "narrative_combined.txt",
            "documentation_json": "documentation.json",
            "cleaned_documentation_json": "cleaned_documentation.json",
            "documentation_without_default_json": "documentation_without_default.json",
            "final_documentation_txt": "final_documentation.txt",
            "extracted_code_txt": "extracted_code.txt",
        }
    return {
        "experiment_parameters_json": f"experiment_parameters{suffix}.json",
        "gui_parameters_json": f"gui_parameters{suffix}.json",
        "updated_gui_parameters_json": f"updated_gui_parameters{suffix}.json",
        "updated_experiment_parameters_json": f"updated_experiment_parameters{suffix}.json",
        "narrative_txt": f"narrative_combined{suffix}.txt",
        "documentation_json": f"documentation{suffix}.json",
        "cleaned_documentation_json": f"cleaned_documentation{suffix}.json",
        "documentation_without_default_json": f"documentation_without_default{suffix}.json",
        "final_documentation_txt": f"final_documentation{suffix}.txt",
        "extracted_code_txt": f"extracted_code{suffix}.txt",
    }


def _infer_abm_id(model_path: Path) -> str | None:
    """Infer ABM identifier from model path for narrative/experiment selection."""
    candidates = {
        _normalise_name(component)
        for component in [model_path.stem, model_path.parent.name]
        if component
    }
    for candidate in candidates:
        if "fauna" in candidate:
            return "fauna"
        if "grazing" in candidate:
            return "grazing"
        if "milk" in candidate and "consumption" in candidate:
            return "milk_consumption"
    return None


def _normalise_name(value: str) -> str:
    """Normalize filesystem names for stable ABM inference."""
    return value.lower().replace("-", "_").replace(" ", "_")


def _resolve_preferred_experiment(model_path: Path) -> str | None:
    abm_id = _infer_abm_id(model_path)
    if abm_id is None:
        return None
    return ABM_PREFERRED_EXPERIMENTS.get(abm_id)


def _resolve_experiment_parameters(
    model_path: Path,
    experiment_parameters: Mapping[str, ParameterScalar],
    *,
    preferred_experiment: str | None = None,
) -> dict[str, ParameterScalar]:
    """Merge explicit CLI inputs with deterministic BehaviorSpace defaults."""
    selected_experiment = preferred_experiment or _resolve_preferred_experiment(model_path)
    defaults: dict[str, Any] = _extract_experiment_parameters(
        model_path.read_text(encoding="utf-8"),
        preferred_experiment=selected_experiment,
    )
    merged: dict[str, ParameterScalar] = {
        key: value for key, value in defaults.items() if isinstance(value, (bool, int, float, str))
    }
    for key, value in experiment_parameters.items():
        merged[key] = value
    return merged


def resolve_experiment_parameters(
    *,
    model_path: Path,
    experiment_parameters: Mapping[str, ParameterScalar],
    preferred_experiment: str | None = None,
) -> dict[str, ParameterScalar]:
    """Public wrapper for deterministic BehaviorSpace parameter resolution."""
    return _resolve_experiment_parameters(
        model_path=model_path,
        experiment_parameters=experiment_parameters,
        preferred_experiment=preferred_experiment,
    )


def _write_reference_narrative_or_build(
    *,
    model_path: Path,
    gui_parameters_path: Path,
    experiment_parameters_path: Path,
    narrative_txt: Path,
) -> str:
    reference_path = find_reference_narrative_path(model_path.parent)
    if reference_path is not None:
        narrative = reference_path.read_text(encoding="utf-8")
    else:
        narrative = _build_parameter_narrative(
            gui_parameters_path=gui_parameters_path,
            experiment_parameters_path=experiment_parameters_path,
            output_text_path=narrative_txt,
        )
        return narrative

    narrative_txt.parent.mkdir(parents=True, exist_ok=True)
    narrative_txt.write_text(narrative, encoding="utf-8")
    return narrative


def _default_link_factory(*, netlogo_home: str) -> NetLogoLinkProtocol:
    try:
        import pynetlogo
    except Exception as exc:  # pragma: no cover - exercised via injected factory in tests
        raise RuntimeError("pynetlogo is required to run NetLogo experiments") from exc
    jvm_path = _resolve_jvm_path()
    return cast(
        NetLogoLinkProtocol,
        pynetlogo.NetLogoLink(netlogo_home=netlogo_home, jvm_path=jvm_path),
    )


def _resolve_jvm_path() -> str | None:
    """Prefer a modern macOS JVM for pynetlogo when one is installed."""
    if sys.platform != "darwin":
        return None

    return _find_modern_macos_jvm()


def _find_modern_macos_jvm() -> str | None:
    """Return a Java 17+ libjvm path when available on macOS."""
    for version in ("21", "17", "11"):
        try:
            result = subprocess.run(
                ["/usr/libexec/java_home", "-v", version],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        java_home = result.stdout.strip()
        if not java_home:
            continue
        libjvm = Path(java_home) / "lib" / "server" / "libjvm.dylib"
        if libjvm.exists():
            return str(libjvm)
    return None


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
    """Run NetLogo repetition loops and write a combined CSV artifact."""
    factory = netlogo_link_factory or _default_link_factory
    netlogo = factory(netlogo_home=netlogo_home)
    netlogo.load_model(str(model_path))
    all_results = pd.DataFrame()

    for _run in range(num_runs):
        run_df = run_single_repetition(
            netlogo=netlogo,
            reporters=reporters,
            max_ticks=max_ticks,
            interval=interval,
            experiment_parameters=dict(experiment_parameters),
        )
        if all_results.empty:
            all_results = run_df
        else:
            all_results = pd.concat([all_results, run_df], axis=1)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(output_csv_path, index=False, sep=";")
    return all_results


def run_single_repetition(
    *,
    netlogo: NetLogoLinkProtocol,
    reporters: Sequence[str],
    max_ticks: int,
    interval: int,
    experiment_parameters: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run a single NetLogo repetition and return a dataframe."""
    return _run_single_repetition(
        netlogo=netlogo,
        reporters=reporters,
        max_ticks=max_ticks,
        interval=interval,
        experiment_parameters=experiment_parameters,
    )


def _coerce_parameter_value_for_netlogo(value: ParameterScalar) -> ParameterScalar:
    """Expose NetLogo scalar coercion for workflow callers."""
    return coerce_parameter_value_for_netlogo(value)


def save_experiment_parameters(
    experiment_parameters: Mapping[str, ParameterScalar],
    output_json_path: Path,
) -> None:
    """Persist experiment parameters to JSON."""
    _save_experiment_parameters(experiment_parameters, output_json_path)


def extract_gui_parameters_to_json(model_path: Path, output_json_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Extract GUI parameters and persist them to JSON."""
    return _extract_gui_parameters_to_json(model_path, output_json_path)


def update_gui_with_experiment_parameters(
    *,
    gui_parameters_path: Path,
    experiment_parameters_path: Path,
    updated_gui_parameters_path: Path,
    updated_experiment_parameters_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Update GUI parameters with experiment values and persist artifacts."""
    return _update_gui_with_experiment_parameters(
        gui_parameters_path=gui_parameters_path,
        experiment_parameters_path=experiment_parameters_path,
        updated_gui_parameters_path=updated_gui_parameters_path,
        updated_experiment_parameters_path=updated_experiment_parameters_path,
    )


def build_parameter_narrative(
    *,
    gui_parameters_path: Path,
    experiment_parameters_path: Path,
    output_text_path: Path,
) -> str:
    """Build parameter narrative text artifact."""
    return _build_parameter_narrative(
        gui_parameters_path=gui_parameters_path,
        experiment_parameters_path=experiment_parameters_path,
        output_text_path=output_text_path,
    )


def extract_documentation_to_json(model_path: Path, output_json_path: Path) -> str:
    """Extract documentation and write JSON artifact."""
    return _extract_documentation_to_json(model_path, output_json_path)


def extract_code_to_text(model_path: Path, output_text_path: Path) -> str:
    """Extract NetLogo code and write TXT artifact."""
    return _extract_code_to_text(model_path, output_text_path)


def run_ingest_workflow(
    *,
    model_path: Path,
    experiment_parameters: Mapping[str, ParameterScalar],
    output_dir: Path,
    suffix: str = "",
) -> dict[str, Path]:
    """Run ingestion preprocessing workflow and return artifact paths."""
    json_dir = output_dir / "JSON"
    txt_dir = output_dir / "TXT"
    artifact_names = _artifact_name_map(suffix)
    experiment_parameters_json = json_dir / artifact_names["experiment_parameters_json"]
    gui_parameters_json = json_dir / artifact_names["gui_parameters_json"]
    updated_gui_parameters_json = json_dir / artifact_names["updated_gui_parameters_json"]
    updated_experiment_parameters_json = json_dir / artifact_names["updated_experiment_parameters_json"]
    narrative_txt = txt_dir / artifact_names["narrative_txt"]
    documentation_json = json_dir / artifact_names["documentation_json"]
    cleaned_documentation_json = json_dir / artifact_names["cleaned_documentation_json"]
    documentation_without_default_json = json_dir / artifact_names["documentation_without_default_json"]
    final_documentation_txt = txt_dir / artifact_names["final_documentation_txt"]
    extracted_code_txt = txt_dir / artifact_names["extracted_code_txt"]

    resolved_experiment_parameters = _resolve_experiment_parameters(model_path, experiment_parameters)
    save_experiment_parameters(dict(resolved_experiment_parameters), experiment_parameters_json)
    extract_gui_parameters_to_json(model_path, gui_parameters_json)
    update_gui_with_experiment_parameters(
        gui_parameters_path=gui_parameters_json,
        experiment_parameters_path=experiment_parameters_json,
        updated_gui_parameters_path=updated_gui_parameters_json,
        updated_experiment_parameters_path=updated_experiment_parameters_json,
    )
    _write_reference_narrative_or_build(
        model_path=model_path,
        gui_parameters_path=updated_gui_parameters_json,
        experiment_parameters_path=updated_experiment_parameters_json,
        narrative_txt=narrative_txt,
    )
    _extract_documentation_to_json(model_path, documentation_json)
    _remove_documentation_urls(documentation_json, cleaned_documentation_json)
    _remove_documentation_defaults(cleaned_documentation_json, documentation_without_default_json)
    _clean_documentation_artifacts(documentation_without_default_json, final_documentation_txt)
    _extract_code_to_text(model_path, extracted_code_txt)
    artifact_paths = {
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
    return artifact_paths
