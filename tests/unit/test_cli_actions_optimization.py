from __future__ import annotations

from pathlib import Path

from distill_abm.cli_actions import execute_smoke_optimization_gemini_chain_command


def test_execute_smoke_optimization_gemini_chain_resolves_stage_run_roots(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_execute_smoke_ingest_command_fn(**kwargs):  # type: ignore[no-untyped-def]
        output_root = Path(kwargs["output_root"])
        run_root = output_root / "runs" / "run_ingest"
        run_root.mkdir(parents=True, exist_ok=True)
        (output_root / "latest_run.txt").write_text(str(run_root), encoding="utf-8")

    def fake_execute_smoke_viz_command_fn(**kwargs):  # type: ignore[no-untyped-def]
        output_root = Path(kwargs["output_root"])
        run_root = output_root / "runs" / "run_viz"
        run_root.mkdir(parents=True, exist_ok=True)
        (output_root / "latest_run.txt").write_text(str(run_root), encoding="utf-8")

    def fake_execute_smoke_doe_command_fn(**kwargs):  # type: ignore[no-untyped-def]
        captured["doe_ingest_root"] = kwargs["ingest_root"]
        captured["doe_viz_root"] = kwargs["viz_root"]
        captured["doe_prompt_variants"] = kwargs["prompt_variants"]

    def fake_execute_smoke_full_case_suite_command_fn(**kwargs):  # type: ignore[no-untyped-def]
        captured["full_case_ingest_root"] = kwargs["ingest_root"]
        captured["full_case_viz_root"] = kwargs["viz_root"]
        output_root = Path(kwargs["output_root"])
        run_root = output_root / "runs" / "run_full_case"
        run_root.mkdir(parents=True, exist_ok=True)
        (output_root / "latest_run.txt").write_text(str(run_root), encoding="utf-8")

    def fake_execute_smoke_summarizers_command_fn(**kwargs):  # type: ignore[no-untyped-def]
        captured["summarizer_source_root"] = kwargs["source_root"]

    def fake_execute_smoke_quantitative_command_fn(**kwargs):  # type: ignore[no-untyped-def]
        captured["quantitative_source_root"] = kwargs["source_root"]

    execute_smoke_optimization_gemini_chain_command(
        models_root=tmp_path / "data",
        netlogo_home="/fake/netlogo",
        prompts_path=tmp_path / "configs" / "prompts.yaml",
        models_path=tmp_path / "configs" / "models.yaml",
        output_root=tmp_path / "results" / "gemini-3.1-pro-preview_optimization_all_abms_chain",
        evidence_modes=("plot",),
        prompt_variants=("example",),
        repetitions=(1, 2, 3),
        summarization_modes=("bart", "bert", "t5"),
        model_id="gemini_3_1_pro_preview",
        max_tokens=32768,
        resume=True,
        execute_smoke_ingest_command_fn=fake_execute_smoke_ingest_command_fn,
        execute_smoke_viz_command_fn=fake_execute_smoke_viz_command_fn,
        execute_smoke_doe_command_fn=fake_execute_smoke_doe_command_fn,
        execute_smoke_full_case_suite_command_fn=fake_execute_smoke_full_case_suite_command_fn,
        execute_smoke_summarizers_command_fn=fake_execute_smoke_summarizers_command_fn,
        execute_smoke_quantitative_command_fn=fake_execute_smoke_quantitative_command_fn,
        run_ingest_smoke_suite_fn=lambda **kwargs: None,
        resolve_viz_smoke_specs_fn=lambda *args, **kwargs: None,
        run_viz_smoke_suite_fn=lambda **kwargs: None,
        discover_abms_fn=lambda: ("fauna", "grazing", "milk_consumption"),
        resolve_model_from_registry_fn=lambda *args, **kwargs: ("openrouter", "google/gemini-3.1-pro-preview"),
        resolve_model_path_fn=lambda **kwargs: tmp_path / "data" / f"{kwargs['abm']}.nlogo",
        run_doe_smoke_suite_fn=lambda **kwargs: None,
        load_abm_config_fn=lambda *args, **kwargs: None,
        load_prompts_config_fn=lambda *args, **kwargs: None,
        create_adapter_fn=lambda *args, **kwargs: object(),
        run_full_case_suite_smoke_fn=lambda **kwargs: None,
        validate_model_policy_fn=lambda **kwargs: None,
        run_summarizer_smoke_fn=lambda **kwargs: None,
        run_quantitative_smoke_fn=lambda **kwargs: None,
    )

    output_root = tmp_path / "results" / "gemini-3.1-pro-preview_optimization_all_abms_chain"
    assert captured["doe_ingest_root"] == output_root / "01_ingest_smoke_latest" / "runs" / "run_ingest"
    assert captured["doe_viz_root"] == output_root / "02_viz_smoke_latest" / "runs" / "run_viz"
    assert captured["full_case_ingest_root"] == output_root / "01_ingest_smoke_latest" / "runs" / "run_ingest"
    assert captured["full_case_viz_root"] == output_root / "02_viz_smoke_latest" / "runs" / "run_viz"
    assert captured["doe_prompt_variants"] == ("example",)
    assert captured["summarizer_source_root"] == output_root / "04_full_case_suite_smoke_latest" / "runs" / "run_full_case"
    assert captured["quantitative_source_root"] == output_root / "05_summarizer_smoke_latest"
