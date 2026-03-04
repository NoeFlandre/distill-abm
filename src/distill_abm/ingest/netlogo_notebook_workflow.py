"""Notebook-equivalent NetLogo ingestion workflow helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from distill_abm.ingest.netlogo_notebook_artifacts import (
    build_parameter_narrative as _build_parameter_narrative,
)
from distill_abm.ingest.netlogo_notebook_artifacts import (
    clean_documentation_artifacts as _clean_documentation_artifacts,
)
from distill_abm.ingest.netlogo_notebook_artifacts import (
    extract_code_to_text as _extract_code_to_text,
)
from distill_abm.ingest.netlogo_notebook_artifacts import (
    extract_documentation_to_json as _extract_documentation_to_json,
)
from distill_abm.ingest.netlogo_notebook_artifacts import (
    extract_gui_parameters_to_json as _extract_gui_parameters_to_json,
)
from distill_abm.ingest.netlogo_notebook_artifacts import (
    remove_documentation_defaults as _remove_documentation_defaults,
)
from distill_abm.ingest.netlogo_notebook_artifacts import (
    remove_documentation_urls as _remove_documentation_urls,
)
from distill_abm.ingest.netlogo_notebook_artifacts import (
    save_experiment_parameters as _save_experiment_parameters,
)
from distill_abm.ingest.netlogo_notebook_artifacts import (
    update_gui_with_experiment_parameters as _update_gui_with_experiment_parameters,
)
from distill_abm.ingest.netlogo_notebook_steps import (
    NetLogoLinkProtocol,
    coerce_parameter_value_for_netlogo,
)
from distill_abm.ingest.netlogo_notebook_steps import (
    run_single_repetition as _run_single_repetition,
)

ParameterScalar = bool | int | float | str


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
    all_results.to_csv(output_csv_path, index=False)
    return all_results


def run_single_repetition(
    *,
    netlogo: NetLogoLinkProtocol,
    reporters: Sequence[str],
    max_ticks: int,
    interval: int,
    experiment_parameters: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Wrapper retained for compatibility with existing unit and caller imports."""
    return _run_single_repetition(
        netlogo=netlogo,
        reporters=reporters,
        max_ticks=max_ticks,
        interval=interval,
        experiment_parameters=experiment_parameters,
    )


def _coerce_parameter_value_for_netlogo(value: ParameterScalar) -> ParameterScalar:
    """Compatibility shim for current and legacy callers expecting local helper."""
    return coerce_parameter_value_for_netlogo(value)


def save_experiment_parameters(
    experiment_parameters: Mapping[str, ParameterScalar],
    output_json_path: Path,
) -> None:
    """Wrapper retained for notebook-compatible import paths."""
    _save_experiment_parameters(experiment_parameters, output_json_path)


def extract_gui_parameters_to_json(model_path: Path, output_json_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Wrapper retained for notebook-compatible import paths."""
    return _extract_gui_parameters_to_json(model_path, output_json_path)


def update_gui_with_experiment_parameters(
    *,
    gui_parameters_path: Path,
    experiment_parameters_path: Path,
    updated_gui_parameters_path: Path,
    updated_experiment_parameters_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Wrapper retained for notebook-compatible import paths."""
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
    """Wrapper retained for notebook-compatible import paths."""
    return _build_parameter_narrative(
        gui_parameters_path=gui_parameters_path,
        experiment_parameters_path=experiment_parameters_path,
        output_text_path=output_text_path,
    )


def extract_documentation_to_json(model_path: Path, output_json_path: Path) -> str:
    """Wrapper retained for notebook-compatible import paths."""
    return _extract_documentation_to_json(model_path, output_json_path)


def extract_code_to_text(model_path: Path, output_text_path: Path) -> str:
    """Wrapper retained for notebook-compatible import paths."""
    return _extract_code_to_text(model_path, output_text_path)


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

    save_experiment_parameters(dict(experiment_parameters), experiment_parameters_json)
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
    _extract_documentation_to_json(model_path, documentation_json)
    _remove_documentation_urls(documentation_json, cleaned_documentation_json)
    _remove_documentation_defaults(cleaned_documentation_json, documentation_without_default_json)
    _clean_documentation_artifacts(documentation_without_default_json, final_documentation_txt)
    _extract_code_to_text(model_path, extracted_code_txt)

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
