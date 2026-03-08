from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from distill_abm.run_viewer import render_run_viewer, resolve_run_root


def _build_sample_run(root: Path) -> Path:
    run_root = root / 'runs' / 'run_1'
    case_root = run_root / 'cases' / '01_case'
    (case_root / '01_inputs').mkdir(parents=True)
    (case_root / '02_requests').mkdir(parents=True)
    (case_root / '03_outputs').mkdir(parents=True)
    (root / 'latest_run.txt').write_text(str(run_root), encoding='utf-8')
    (run_root / 'smoke_local_qwen_report.json').write_text(dedent('''
    {
      "success": true,
      "failed_case_ids": [],
      "started_at_utc": "2026-03-08T00:00:00+00:00",
      "finished_at_utc": "2026-03-08T00:01:00+00:00",
      "cases": [
        {
          "case_id": "01_case",
          "abm": "fauna",
          "evidence_mode": "plot",
          "prompt_variant": "none",
          "model": "nvidia/nemotron-nano-12b-v2-vl:free",
          "resumed_from_existing": false,
          "success": true,
          "error": null
        }
      ]
    }
    '''), encoding='utf-8')
    (run_root / 'run.log.jsonl').write_text('{"message":"x"}\n', encoding='utf-8')
    (case_root / '00_case_summary.json').write_text(dedent('''
    {
      "case_id": "01_case",
      "abm": "fauna",
      "evidence_mode": "plot",
      "prompt_variant": "none",
      "model": "nvidia/nemotron-nano-12b-v2-vl:free"
    }
    '''), encoding='utf-8')
    (case_root / '01_inputs' / 'context_prompt.txt').write_text('ctx prompt', encoding='utf-8')
    (case_root / '01_inputs' / 'documentation.txt').write_text('docs body', encoding='utf-8')
    (case_root / '01_inputs' / 'parameters.txt').write_text('params body', encoding='utf-8')
    (case_root / '01_inputs' / 'trend_prompt.txt').write_text('trend prompt', encoding='utf-8')
    (case_root / '01_inputs' / 'trend_evidence_plot.png').write_bytes(b'png')
    (case_root / '02_requests' / 'hyperparameters.json').write_text('{"max_tokens": 10}', encoding='utf-8')
    (case_root / '03_outputs' / 'context_output.txt').write_text('ctx out', encoding='utf-8')
    (case_root / '03_outputs' / 'trend_output.txt').write_text('trend out', encoding='utf-8')
    (case_root / '03_outputs' / 'context_trace.json').write_text('{}', encoding='utf-8')
    (case_root / '03_outputs' / 'trend_trace.json').write_text('{}', encoding='utf-8')
    return root


def test_resolve_run_root_uses_latest_run_pointer(tmp_path: Path) -> None:
    root = _build_sample_run(tmp_path)
    assert resolve_run_root(root) == root / 'runs' / 'run_1'


def test_render_run_viewer_writes_minimal_review_html(tmp_path: Path) -> None:
    root = _build_sample_run(tmp_path)
    html_path = render_run_viewer(root)

    html = html_path.read_text(encoding='utf-8')
    assert html_path == root / 'runs' / 'run_1' / 'review.html'
    assert 'Run Review' in html
    assert '01_case' in html
    assert 'docs body' in html
    assert 'params body' in html
    assert 'ctx out' in html
    assert 'trend out' in html
    assert 'trend_evidence_plot.png' in html


def test_render_run_viewer_writes_full_case_trend_sections(tmp_path: Path) -> None:
    run_root = tmp_path / 'runs' / 'run_1'
    case_root = run_root / 'cases' / '01_case'
    (case_root / '01_inputs').mkdir(parents=True)
    (case_root / '02_context').mkdir(parents=True)
    (case_root / '03_trends' / 'plot_01').mkdir(parents=True)
    (tmp_path / 'latest_run.txt').write_text(str(run_root), encoding='utf-8')
    (run_root / 'smoke_full_case_matrix_report.json').write_text(
        '{"success": true, "failed_case_ids": [], "cases": [{"case_id": "01_case", "abm": "grazing", '
        '"evidence_mode": "table", "prompt_variant": "role", "model": "m", "resumed_from_existing": false, '
        '"success": true, "error": null}]}',
        encoding='utf-8',
    )
    (run_root / 'run.log.jsonl').write_text('{"message":"x"}\n', encoding='utf-8')
    (case_root / '00_case_summary.json').write_text(
        '{"case_id":"01_case","abm":"grazing","evidence_mode":"table","prompt_variant":"role","model":"m"}',
        encoding='utf-8',
    )
    (case_root / '01_inputs' / 'context_prompt.txt').write_text('ctx prompt', encoding='utf-8')
    (case_root / '01_inputs' / 'documentation.txt').write_text('docs body', encoding='utf-8')
    (case_root / '01_inputs' / 'parameters.txt').write_text('params body', encoding='utf-8')
    (case_root / '02_context' / 'context_output.txt').write_text('ctx out', encoding='utf-8')
    (case_root / '02_context' / 'context_trace.json').write_text('{}', encoding='utf-8')
    (case_root / '03_trends' / 'plot_01' / 'trend_prompt.txt').write_text('trend prompt', encoding='utf-8')
    (case_root / '03_trends' / 'plot_01' / 'trend_output.txt').write_text('trend out', encoding='utf-8')
    (case_root / '03_trends' / 'plot_01' / 'trend_trace.json').write_text('{}', encoding='utf-8')
    (case_root / '03_trends' / 'plot_01' / 'trend_evidence_table.csv').write_text(
        'tick,metric\n0,1\n',
        encoding='utf-8',
    )

    html_path = render_run_viewer(tmp_path)

    html = html_path.read_text(encoding='utf-8')
    assert 'trend prompt' in html
    assert 'trend out' in html
    assert 'plot_01' in html
    assert 'tick,metric' in html
