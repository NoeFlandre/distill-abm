"""Pure helper logic extracted from the Typer entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

import typer

from distill_abm.configs.loader import load_abm_config, load_experiment_settings, load_models_config
from distill_abm.configs.models import ModelEntry, SummarizerId
from distill_abm.ingest.netlogo_workflow import resolve_experiment_parameters
from distill_abm.pipeline.smoke import SmokeCase, default_branch_smoke_cases, default_smoke_cases
from distill_abm.viz.plots import MetricPlotBundle
from distill_abm.viz.viz_smoke import VizSmokeSpec

BENCHMARK_MODELS: set[tuple[str, str]] = {
    ("openrouter", "moonshotai/kimi-k2.5"),
    ("openrouter", "google/gemini-3.1-pro-preview"),
    ("openrouter", "qwen/qwen3.5-27b"),
}
SUPPORTED_SUMMARIZERS: tuple[SummarizerId, ...] = ("bart", "bert", "t5", "longformer_ext")


def load_experiment_parameters(path: Path | None) -> dict[str, bool | int | float | str]:
    """Load experiment parameter overrides from an optional JSON file."""
    if path is None:
        return {}

    if not path.exists():
        raise typer.BadParameter(f"experiment parameters file not found: {path}")
    payload = path.read_text(encoding="utf-8")
    if not payload.strip():
        return {}

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"experiment parameters file must contain JSON object: {path}") from exc

    if not isinstance(parsed, dict):
        raise typer.BadParameter("experiment parameters file must contain a JSON object at top level.")

    sanitized: dict[str, bool | int | float | str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str):
            raise typer.BadParameter("experiment parameters keys must be strings.")
        if not isinstance(value, (bool, int, float, str)):
            raise typer.BadParameter(
                f"unsupported value type for experiment parameter '{key}': {type(value).__name__}. "
                "Allowed types are bool, int, float, and str."
            )
        sanitized[key] = value

    return sanitized


def as_dict(value: object) -> dict[str, object]:
    """Normalize optional JSON payload fragments to dictionaries."""
    return value if isinstance(value, dict) else {}


def discover_configured_abms() -> tuple[str, ...]:
    """Return ABM identifiers configured in configs/abms."""
    return tuple(sorted(p.stem for p in Path("configs/abms").glob("*.yaml")))


def resolve_model_filenames(abm: str) -> tuple[str, ...]:
    """Return preferred NetLogo model filenames for an ABM."""
    candidates = [f"{abm}.nlogo", f"{abm}_model.nlogo"]
    if abm == "milk_consumption":
        candidates.append("model.nlogo")
    return tuple(dict.fromkeys(candidates))


def resolve_abm_model_path(*, abm: str, models_root: Path) -> Path:
    """Find a single NetLogo model for an ABM and fail with a clear message if absent or ambiguous."""
    candidate_roots = [models_root, models_root / f"{abm}_abm", models_root / abm]
    model_filenames = resolve_model_filenames(abm)
    matches: list[Path] = []
    for directory in candidate_roots:
        if not directory.exists():
            continue
        if directory.is_file():
            if directory.name in model_filenames:
                matches.append(directory)
            continue
        for model_name in model_filenames:
            candidate = directory / model_name
            if candidate.exists():
                matches.append(candidate)

    if not matches:
        candidates_desc = ", ".join(str(directory / name) for directory in candidate_roots for name in model_filenames)
        raise typer.BadParameter(
            f"no .nlogo file found for ABM '{abm}' in {models_root}. " f"Searched: {candidates_desc}"
        )
    matches = list(dict.fromkeys(matches))

    if len(matches) > 1:
        names = ", ".join(str(match.relative_to(models_root)) for match in matches)
        raise typer.BadParameter(f"multiple .nlogo files found for ABM '{abm}': {names}.")
    return matches[0]


def resolve_abm_experiment_parameters_path(*, model_dir: Path, abm: str, explicit: Path | None) -> Path | None:
    """Resolve per-ABM experiment-parameters file if available."""
    if explicit is not None:
        if explicit.exists():
            return explicit
        return None

    candidates = [
        model_dir / "experiment_parameters.json",
        model_dir / "experiment-parameters.json",
        model_dir / f"{abm}_experiment_parameters.json",
        model_dir / f"{abm}-experiment_parameters.json",
        model_dir / f"{abm}-experiment-parameters.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_viz_smoke_specs(*, requested_abms: list[str], models_root: Path) -> dict[str, VizSmokeSpec]:
    """Resolve ABM NetLogo simulation-and-plot specs for visualization smoke runs."""
    specs: dict[str, VizSmokeSpec] = {}
    missing_viz_config: list[str] = []
    for abm in requested_abms:
        abm_config = load_abm_config(Path("configs/abms") / f"{abm}.yaml")
        model_path = resolve_abm_model_path(abm=abm, models_root=models_root)
        if abm_config.netlogo_viz is None:
            missing_viz_config.append(abm)
            continue
        viz_config = abm_config.netlogo_viz
        experiment_parameters = resolve_experiment_parameters(
            model_path=model_path,
            experiment_parameters=viz_config.experiment_parameters,
            preferred_experiment=viz_config.experiment_name,
        )
        specs[abm] = VizSmokeSpec(
            abm=abm,
            model_path=model_path,
            experiment_name=viz_config.experiment_name,
            experiment_parameters=experiment_parameters,
            num_runs=viz_config.smoke_num_runs or viz_config.num_runs,
            max_ticks=viz_config.smoke_max_ticks or viz_config.max_ticks,
            interval=viz_config.smoke_interval or viz_config.interval,
            fallback_mode=viz_config.fallback_mode,
            fallback_csv=Path(viz_config.fallback_csv) if viz_config.fallback_csv else None,
            fallback_plot_dir=Path(viz_config.fallback_plot_dir) if viz_config.fallback_plot_dir else None,
            reporters=list(viz_config.reporters),
            plots=[
                MetricPlotBundle(
                    include_pattern=plot.reporter_pattern,
                    title=plot.title,
                    y_label=plot.y_label,
                    x_label=plot.x_label,
                    exclude_pattern=plot.exclude_pattern,
                    show_mean_line=plot.show_mean_line,
                )
                for plot in viz_config.plots
            ],
        )
    if missing_viz_config:
        joined = ", ".join(missing_viz_config)
        raise ValueError(
            f"missing netlogo_viz config for ABM(s): {joined}. "
            "Add the simulation-and-plot spec under netlogo_viz in the ABM config."
        )
    return specs


def parse_summarizers(values: list[str] | None, fallback: tuple[SummarizerId, ...]) -> tuple[SummarizerId, ...]:
    """Validate and normalize summarizer ids."""
    allowed = set(SUPPORTED_SUMMARIZERS)
    normalized = tuple(dict.fromkeys(value.strip() for value in (values or list(fallback))))
    invalid = [value for value in normalized if value not in allowed]
    if invalid:
        raise typer.BadParameter(
            "unsupported summarizer(s): " f"{', '.join(invalid)}. Allowed: {', '.join(SUPPORTED_SUMMARIZERS)}."
        )
    return cast(tuple[SummarizerId, ...], normalized)


def resolve_model_from_registry(models_path: Path, model_id: str) -> tuple[str, str]:
    """Resolve one registry model id into provider/model strings."""
    config = load_models_config(models_path)
    try:
        entry: ModelEntry = config.models[model_id]
    except KeyError as exc:
        available = ", ".join(sorted(config.models))
        raise typer.BadParameter(f"unknown model_id '{model_id}'. Available: {available}") from exc
    return entry.provider, entry.model


def resolve_scoring_reference_path(abm: str) -> Path:
    """Resolve the human scoring reference path for one supported ABM."""
    settings = load_experiment_settings(Path("configs/experiment_settings.yaml"))
    mapping = {
        "fauna": settings.ground_truth.fauna,
        "grazing": settings.ground_truth.grazing,
        "milk_consumption": settings.ground_truth.milk_consumption,
    }
    if abm not in mapping:
        raise typer.BadParameter(
            f"unsupported ABM for scoring reference: {abm}. Allowed: fauna, grazing, milk_consumption."
        )
    return Path(mapping[abm])


def resolve_additional_scoring_reference_paths(abm: str) -> dict[str, Path]:
    """Resolve optional secondary human reference paths for one supported ABM."""
    settings = load_experiment_settings(Path("configs/experiment_settings.yaml"))
    supported_abms = {"fauna", "grazing", "milk_consumption"}
    if abm not in supported_abms:
        raise typer.BadParameter(
            f"unsupported ABM for additional scoring references: {abm}. Allowed: fauna, grazing, milk_consumption."
        )

    references: dict[str, Path] = {}
    if settings.modeler_ground_truth is not None:
        modeler_mapping = {
            "fauna": settings.modeler_ground_truth.fauna,
            "grazing": settings.modeler_ground_truth.grazing,
            "milk_consumption": settings.modeler_ground_truth.milk_consumption,
        }
        resolved_modeler = modeler_mapping[abm]
        if resolved_modeler:
            references["modeler"] = Path(resolved_modeler)

    if settings.gpt5_2_short_ground_truth is not None:
        short_mapping = {
            "fauna": settings.gpt5_2_short_ground_truth.fauna,
            "grazing": settings.gpt5_2_short_ground_truth.grazing,
            "milk_consumption": settings.gpt5_2_short_ground_truth.milk_consumption,
        }
        references["gpt5.2_short"] = Path(short_mapping[abm])

    if settings.gpt5_2_long_ground_truth is not None:
        long_mapping = {
            "fauna": settings.gpt5_2_long_ground_truth.fauna,
            "grazing": settings.gpt5_2_long_ground_truth.grazing,
            "milk_consumption": settings.gpt5_2_long_ground_truth.milk_consumption,
        }
        references["gpt5.2_long"] = Path(long_mapping[abm])

    return references


def resolve_quantitative_reference_paths(abm: str) -> dict[str, Path]:
    """Resolve all reviewer-facing scoring reference families for one supported ABM."""
    return {"author": resolve_scoring_reference_path(abm), **resolve_additional_scoring_reference_paths(abm)}


def validate_model_policy(provider: str, model: str, allow_debug_model: bool) -> None:
    """Enforce the supported benchmark model policy used by the CLI."""
    key = (provider.strip().lower(), model.strip())
    if allow_debug_model and key not in BENCHMARK_MODELS:
        return
    if key not in BENCHMARK_MODELS:
        allowed = ", ".join(f"{p}:{m}" for p, m in sorted(BENCHMARK_MODELS))
        raise typer.BadParameter(
            f"unsupported benchmark model '{provider}:{model}'. Allowed benchmark models: {allowed}."
        )


def select_smoke_cases(
    case_ids: list[str] | None,
    max_cases: int | None,
    profile: Literal["matrix", "three-branches"],
) -> list[SmokeCase] | None:
    """Resolve the selected smoke matrix cases for CLI execution."""
    all_cases = default_smoke_cases() if profile == "matrix" else default_branch_smoke_cases()
    if not case_ids:
        if max_cases is None and profile == "matrix":
            return None
        if max_cases is None:
            return all_cases
        return all_cases[:max_cases]

    by_id = {case.case_id: case for case in all_cases}
    unknown = [value for value in case_ids if value not in by_id]
    if unknown:
        known = ", ".join(sorted(by_id))
        raise typer.BadParameter(f"unknown --case-id value(s): {', '.join(unknown)}. Known cases: {known}")
    seen: set[str] = set()
    selected: list[SmokeCase] = []
    for value in case_ids:
        if value in seen:
            continue
        seen.add(value)
        selected.append(by_id[value])
    if max_cases is not None:
        selected = selected[:max_cases]
    if not selected:
        raise typer.BadParameter("at least one smoke case must be selected")
    return selected
