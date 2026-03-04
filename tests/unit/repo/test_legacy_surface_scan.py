from __future__ import annotations

from pathlib import Path

FORBIDDEN_SNIPPETS = (
    "configs/notebook_experiment_settings.yaml",
    "configs/notebook_prompt_reference.yaml",
    "configs/notebook_ground_truth",
    "netlogo_notebook",
    "distill_abm.compat",
    "distill_abm.reference",
)

SCAN_ROOTS = (Path("README.md"), Path("docs"), Path("configs"), Path("src"))


def test_no_legacy_surface_references_in_active_runtime_docs_or_configs() -> None:
    violations: list[str] = []
    for root in SCAN_ROOTS:
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in paths:
            if not path.is_file():
                continue
            if path.suffix not in {".md", ".py", ".yaml", ".yml", ".txt", ""}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for snippet in FORBIDDEN_SNIPPETS:
                if snippet in text:
                    violations.append(f"{path}: contains '{snippet}'")

    assert not violations, "\n".join(violations)
