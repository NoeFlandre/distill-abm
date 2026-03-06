from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from distill_abm.ingest.netlogo_workflow import (
    _coerce_parameter_value_for_netlogo,
    _parse_java_major_version,
    _resolve_jvm_path,
    build_parameter_narrative,
    extract_code_to_text,
    extract_documentation_to_json,
    run_ingest_workflow,
    run_netlogo_experiment,
    run_single_repetition,
    save_experiment_parameters,
    update_gui_with_experiment_parameters,
)


class _FakeNetLogoLink:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.loaded_model: str | None = None
        self.tick = 0

    def load_model(self, model_path: str) -> None:
        self.loaded_model = model_path

    def command(self, command: str) -> None:
        self.commands.append(command)
        if command == "go":
            self.tick += 1

    def report(self, reporter: str) -> int:
        # deterministic per call for test assertions
        return self.tick * 10 + len(reporter)


def test_coerce_parameter_value_for_netlogo() -> None:
    assert _coerce_parameter_value_for_netlogo(True) == "true"
    assert _coerce_parameter_value_for_netlogo(False) == "false"
    assert _coerce_parameter_value_for_netlogo(7) == 7
    assert _coerce_parameter_value_for_netlogo("abc") == '"abc"'
    assert _coerce_parameter_value_for_netlogo('a"b') == '"a\\"b"'


def test_run_single_repetition_collects_reported_ticks() -> None:
    fake = _FakeNetLogoLink()
    frame = run_single_repetition(
        netlogo=fake,
        reporters=["mean-incum", "mean-alt"],
        max_ticks=90,
        interval=30,
        experiment_parameters=None,
    )

    assert frame.columns.tolist() == ["mean-incum", "mean-alt", "tick"]
    assert frame["tick"].tolist() == [0, 30, 60]
    assert fake.commands == ["setup", "go", "go", "go"]


def test_run_netlogo_experiment_loop_and_csv(tmp_path: Path) -> None:
    output_csv = tmp_path / "netlogo_experiment_results.csv"
    factory_calls: list[dict[str, Any]] = []

    def _factory(*, netlogo_home: str) -> _FakeNetLogoLink:
        factory_calls.append({"netlogo_home": netlogo_home})
        return _FakeNetLogoLink()

    frame = run_netlogo_experiment(
        netlogo_home="/fake/netlogo",
        model_path="NetLogo/model.nlogo",
        experiment_parameters={"norms": True, "number-of-agents": 1000},
        reporters=["mean-incum", "mean-alt"],
        num_runs=2,
        max_ticks=100,
        interval=50,
        output_csv_path=output_csv,
        netlogo_link_factory=_factory,
    )

    assert factory_calls == [{"netlogo_home": "/fake/netlogo"}]
    assert output_csv.exists()
    written = pd.read_csv(output_csv, sep=";")
    assert written.shape == frame.shape
    assert written.values.tolist() == frame.values.tolist()
    assert frame.shape[0] == 2
    # horizontal concat duplicates run columns across repetitions
    assert frame.columns.tolist() == ["mean-incum", "mean-alt", "tick", "mean-incum", "mean-alt", "tick"]
    assert written.columns.tolist() == ["mean-incum", "mean-alt", "tick", "mean-incum.1", "mean-alt.1", "tick.1"]
    assert frame.iloc[:, 2].tolist() == [0, 50]
    assert frame.iloc[:, 5].tolist() == [0, 50]


def test_save_update_and_narrative_workflow(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment_parameters.json"
    gui_path = tmp_path / "gui_parameters.json"
    updated_gui_path = tmp_path / "updated_gui_parameters.json"
    updated_exp_path = tmp_path / "updated_experiment_parameters.json"
    narrative_path = tmp_path / "narrative_combined.txt"

    save_experiment_parameters({"slider-a": 3, "switch-b": True}, experiment_path)
    assert json.loads(experiment_path.read_text(encoding="utf-8")) == {"slider-a": 3, "switch-b": True}

    gui_payload = {
        "sliders": [{"name": "slider-a", "min_value": 0, "max_value": 10}],
        "switches": [{"name": "switch-b", "true_value": True, "false_value": False, "default_value": False}],
        "monitors": [],
    }
    gui_path.write_text(json.dumps(gui_payload, indent=2), encoding="utf-8")

    updated_gui, remaining = update_gui_with_experiment_parameters(
        gui_parameters_path=gui_path,
        experiment_parameters_path=experiment_path,
        updated_gui_parameters_path=updated_gui_path,
        updated_experiment_parameters_path=updated_exp_path,
    )
    assert remaining == {}
    assert updated_gui["sliders"][0]["value"] == 3
    assert updated_gui["switches"][0]["value"] is True
    assert updated_gui_path.exists()
    assert updated_exp_path.exists()

    narrative = build_parameter_narrative(
        gui_parameters_path=updated_gui_path,
        experiment_parameters_path=updated_exp_path,
        output_text_path=narrative_path,
    )
    assert narrative.startswith("We have 2 parameters:")
    assert "slider-a, from 0 to 10. We set it to 3." in narrative
    assert "switch-b. We set it to True." in narrative
    assert narrative_path.read_text(encoding="utf-8") == narrative


def test_extract_documentation_and_code_to_files(tmp_path: Path) -> None:
    model_path = tmp_path / "model.nlogo"
    documentation_json_path = tmp_path / "documentation.json"
    code_txt_path = tmp_path / "extracted_code.txt"
    model_path.write_text(
        "globals [a b]\n" "to go\n" "end\n" "@#$#@#$#@\n" "## WHAT IS IT?\n\nDoc text\n" "@#$#@#$#@\n",
        encoding="utf-8",
    )

    documentation = extract_documentation_to_json(model_path, documentation_json_path)
    code = extract_code_to_text(model_path, code_txt_path)

    assert documentation == "@#$#@#$#@\n## WHAT IS IT?\n\nDoc text"
    assert code.startswith("globals [a b]")
    assert json.loads(documentation_json_path.read_text(encoding="utf-8"))["documentation"] == documentation
    assert code_txt_path.read_text(encoding="utf-8") == code


def test_run_ingest_workflow_end_to_end(tmp_path: Path) -> None:
    model_path = tmp_path / "model.nlogo"
    model_path.write_text(
        "globals [a b]\n"
        "to go\n"
        "end\n"
        "@#$#@#$#@\n"
        "## WHAT IS IT?\n\nDoc text\n"
        "@#$#@#$#@\n"
        "SLIDER 0 0 10 10 slider-a slider-a 0 10 1 5\n"
        "SWITCH 0 0 10 10 switch-b switch-b 1 0 0\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    result = run_ingest_workflow(
        model_path=model_path,
        experiment_parameters={"slider-a": 3, "switch-b": True},
        output_dir=output_dir,
    )

    assert result["experiment_parameters_json"].exists()
    assert result["gui_parameters_json"].exists()
    assert result["updated_gui_parameters_json"].exists()
    assert result["updated_experiment_parameters_json"].exists()
    assert result["narrative_txt"].exists()
    # canonical artifact names
    assert result["documentation_json"].exists()
    assert result["cleaned_documentation_json"].exists()
    assert result["documentation_without_default_json"].exists()
    assert result["final_documentation_txt"].exists()
    assert result["extracted_code_txt"].exists()
    assert "We have 2 parameters:" in result["narrative_txt"].read_text(encoding="utf-8")
    assert "We have 2 parameters:" in result["narrative_txt"].read_text(encoding="utf-8")


def test_run_ingest_workflow_uses_reference_narrative_when_available(tmp_path: Path) -> None:
    model_path = tmp_path / "model.nlogo"
    model_path.write_text(
        "globals [a b]\n"
        "to go\n"
        "end\n"
        "@#$#@#$#@\n"
        "## WHAT IS IT?\n\nDoc text\n"
        "@#$#@#$#@\n",
        encoding="utf-8",
    )
    reference = tmp_path / "reference_narrative.txt"
    reference.write_text("Reference narrative", encoding="utf-8")

    output_dir = tmp_path / "output"
    result = run_ingest_workflow(
        model_path=model_path,
        experiment_parameters={},
        output_dir=output_dir,
    )

    assert result["narrative_txt"].read_text(encoding="utf-8") == "Reference narrative"


def test_default_link_factory_raises_when_pynetlogo_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that _default_link_factory raises RuntimeError when pynetlogo cannot be imported."""
    import sys
    
    # Temporarily hide pynetlogo by removing it from sys.modules if present
    original_modules = dict(sys.modules)
    if 'pynetlogo' in sys.modules:
        del sys.modules['pynetlogo']
    
    # Mock the import to fail
    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == 'pynetlogo' or name.startswith('pynetlogo'):
            raise ImportError("simulated pynetlogo not available")
        return original_modules.get(name)
    
    monkeypatch.setattr("builtins.__import__", mock_import)
    
    from distill_abm.ingest.netlogo_workflow import _default_link_factory
    
    with pytest.raises(RuntimeError) as exc_info:
        _default_link_factory(netlogo_home="/fake/path")
    
    assert "pynetlogo" in str(exc_info.value).lower()


def test_parse_java_major_version_supports_legacy_and_modern_strings() -> None:
    assert _parse_java_major_version('java version "1.8.0_451"') == 8
    assert _parse_java_major_version('openjdk version "17.0.13" 2024-10-15') == 17
    assert _parse_java_major_version('openjdk version "21.0.5" 2024-10-15') == 21
    assert _parse_java_major_version("not a version string") is None


def test_resolve_jvm_path_prefers_modern_macos_jvm_when_default_is_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("distill_abm.ingest.netlogo_workflow.sys.platform", "darwin")
    monkeypatch.setattr(
        "distill_abm.ingest.netlogo_workflow._find_modern_macos_jvm",
        lambda: "/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home/lib/server/libjvm.dylib",
    )
    monkeypatch.setattr(
        "distill_abm.ingest.netlogo_workflow._read_java_major_version",
        lambda: 8,
    )

    assert (
        _resolve_jvm_path()
        == "/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home/lib/server/libjvm.dylib"
    )


def test_resolve_jvm_path_still_prefers_explicit_modern_jvm_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("distill_abm.ingest.netlogo_workflow.sys.platform", "darwin")
    monkeypatch.setattr(
        "distill_abm.ingest.netlogo_workflow._find_modern_macos_jvm",
        lambda: "/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home/lib/server/libjvm.dylib",
    )

    assert (
        _resolve_jvm_path()
        == "/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home/lib/server/libjvm.dylib"
    )


def test_resolve_jvm_path_returns_none_outside_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("distill_abm.ingest.netlogo_workflow.sys.platform", "linux")
    assert _resolve_jvm_path() is None
