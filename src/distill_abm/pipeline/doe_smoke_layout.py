"""Path layout helpers for DOE smoke artifacts."""

from __future__ import annotations

from pathlib import Path


def overview_dir(output_root: Path) -> Path:
    return output_root / "00_overview"


def shared_root_dir(output_root: Path) -> Path:
    return output_root / "10_shared"


def shared_global_dir(output_root: Path) -> Path:
    return shared_root_dir(output_root) / "global"


def case_index_dir(output_root: Path) -> Path:
    return output_root / "20_case_index"


def shared_abm_dir(*, output_root: Path, abm: str) -> Path:
    return shared_root_dir(output_root) / abm


def shared_inputs_dir(*, output_root: Path, abm: str) -> Path:
    return shared_abm_dir(output_root=output_root, abm=abm) / "01_inputs"


def shared_evidence_dir(*, output_root: Path, abm: str) -> Path:
    return shared_abm_dir(output_root=output_root, abm=abm) / "02_evidence"


def shared_plots_dir(*, output_root: Path, abm: str) -> Path:
    return shared_evidence_dir(output_root=output_root, abm=abm) / "plots"


def shared_prompts_dir(*, output_root: Path, abm: str) -> Path:
    return shared_abm_dir(output_root=output_root, abm=abm) / "03_prompts"


def shared_tables_dir(*, output_root: Path, abm: str) -> Path:
    return shared_evidence_dir(output_root=output_root, abm=abm) / "tables"


def shared_context_prompt_path(*, output_root: Path, abm: str, prompt_variant: str) -> Path:
    return shared_prompts_dir(output_root=output_root, abm=abm) / "context" / f"{prompt_variant}.txt"


def shared_trend_prompt_path(
    *, output_root: Path, abm: str, prompt_variant: str, evidence_mode: str, plot_index: int
) -> Path:
    return (
        shared_prompts_dir(output_root=output_root, abm=abm)
        / "trend"
        / evidence_mode
        / prompt_variant
        / f"plot_{plot_index}.txt"
    )


def shared_plot_copy_path(*, output_root: Path, abm: str, plot_index: int) -> Path:
    return shared_plots_dir(output_root=output_root, abm=abm) / f"plot_{plot_index}.png"


def shared_table_path(*, output_root: Path, abm: str, plot_index: int) -> Path:
    return shared_tables_dir(output_root=output_root, abm=abm) / f"plot_{plot_index}.csv"


def layout_guide_path(output_root: Path) -> Path:
    return overview_dir(output_root) / "README.md"
