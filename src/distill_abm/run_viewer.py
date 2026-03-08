"""Static HTML viewer for reviewer-friendly inspection of case-based run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


def render_run_viewer(run_root: Path, output_path: Path | None = None) -> Path:
    """Render one self-contained HTML viewer for a case-based run directory."""
    resolved_run_root = resolve_run_root(run_root)
    target_path = output_path or resolved_run_root / "review.html"
    payload = _build_viewer_payload(resolved_run_root)
    target_path.write_text(_render_html(payload), encoding="utf-8")
    return target_path


def resolve_run_root(path: Path) -> Path:
    """Resolve either a concrete run directory or a root containing latest_run.txt."""
    candidate = path
    latest_run_path = candidate / "latest_run.txt"
    if latest_run_path.exists():
        latest_text = latest_run_path.read_text(encoding="utf-8").strip()
        if latest_text:
            return Path(latest_text)
    return candidate


def _build_viewer_payload(run_root: Path) -> dict[str, Any]:
    report_path = _resolve_report_path(run_root)
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    run_log_path = run_root / "run.log.jsonl"
    cases_root = run_root / "cases"
    cases = []
    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        if (case_dir / "03_outputs").exists():
            cases.append(_build_sample_case_payload(run_root=run_root, report=report, case_dir=case_dir))
        elif (case_dir / "03_trends").exists():
            cases.append(_build_full_case_payload(run_root=run_root, report=report, case_dir=case_dir))
    return {
        "run_root": str(run_root),
        "report_path": str(report_path) if report_path.exists() else "",
        "run_log_path": str(run_log_path) if run_log_path.exists() else "",
        "success": report.get("success"),
        "failed_case_ids": report.get("failed_case_ids", []),
        "started_at_utc": report.get("started_at_utc", ""),
        "finished_at_utc": report.get("finished_at_utc", ""),
        "cases": cases,
    }


def _resolve_report_path(run_root: Path) -> Path:
    for name in ("smoke_local_qwen_report.json", "smoke_full_case_matrix_report.json"):
        candidate = run_root / name
        if candidate.exists():
            return candidate
    return run_root / "smoke_local_qwen_report.json"


def _build_sample_case_payload(*, run_root: Path, report: dict[str, Any], case_dir: Path) -> dict[str, Any]:
    case_summary = _read_optional_json(case_dir / "00_case_summary.json")
    inputs_dir = case_dir / "01_inputs"
    requests_dir = case_dir / "02_requests"
    outputs_dir = case_dir / "03_outputs"
    return {
        "case_id": case_summary.get("case_id", case_dir.name),
        "abm": case_summary.get("abm", ""),
        "evidence_mode": case_summary.get("evidence_mode", ""),
        "prompt_variant": case_summary.get("prompt_variant", ""),
        "model": case_summary.get("model", ""),
        "resumed_from_existing": _lookup_resumed_flag(
            report=report,
            case_id=case_summary.get("case_id", case_dir.name),
        ),
        "success": not (outputs_dir / "error.txt").exists(),
        "error_text": _read_optional_text(outputs_dir / "error.txt"),
        "paths": {
            "case_dir": _relative_path(run_root, case_dir),
            "context_prompt": _relative_path(run_root, inputs_dir / "context_prompt.txt"),
            "documentation": _relative_path(run_root, inputs_dir / "documentation.txt"),
            "parameters": _relative_path(run_root, inputs_dir / "parameters.txt"),
            "trend_prompt": _relative_path(run_root, inputs_dir / "trend_prompt.txt"),
            "table_csv": _relative_path(run_root, inputs_dir / "trend_evidence_table.csv"),
            "image": _relative_path(run_root, inputs_dir / "trend_evidence_plot.png"),
            "context_output": _relative_path(run_root, outputs_dir / "context_output.txt"),
            "trend_output": _relative_path(run_root, outputs_dir / "trend_output.txt"),
            "hyperparameters": _relative_path(run_root, requests_dir / "hyperparameters.json"),
            "context_trace": _relative_path(run_root, outputs_dir / "context_trace.json"),
            "trend_trace": _relative_path(run_root, outputs_dir / "trend_trace.json"),
        },
        "context_prompt_text": _read_optional_text(inputs_dir / "context_prompt.txt"),
        "documentation_text": _read_optional_text(inputs_dir / "documentation.txt"),
        "parameters_text": _read_optional_text(inputs_dir / "parameters.txt"),
        "trend_prompt_text": _read_optional_text(inputs_dir / "trend_prompt.txt"),
        "table_csv_text": _read_optional_text(inputs_dir / "trend_evidence_table.csv"),
        "context_output_text": _read_optional_text(outputs_dir / "context_output.txt"),
        "trend_output_text": _read_optional_text(outputs_dir / "trend_output.txt"),
        "hyperparameters_text": _read_optional_text(requests_dir / "hyperparameters.json"),
        "trends": [],
    }


def _build_full_case_payload(*, run_root: Path, report: dict[str, Any], case_dir: Path) -> dict[str, Any]:
    case_summary = _read_optional_json(case_dir / "00_case_summary.json")
    inputs_dir = case_dir / "01_inputs"
    context_dir = case_dir / "02_context"
    trends_root = case_dir / "03_trends"
    trend_entries = []
    for trend_dir in sorted(path for path in trends_root.iterdir() if path.is_dir()):
        trend_entries.append(
            {
                "plot_id": trend_dir.name,
                "trend_prompt_path": _relative_path(run_root, trend_dir / "trend_prompt.txt"),
                "trend_prompt_text": _read_optional_text(trend_dir / "trend_prompt.txt"),
                "trend_output_path": _relative_path(run_root, trend_dir / "trend_output.txt"),
                "trend_output_text": _read_optional_text(trend_dir / "trend_output.txt"),
                "table_csv_path": _relative_path(run_root, trend_dir / "trend_evidence_table.csv"),
                "table_csv_text": _read_optional_text(trend_dir / "trend_evidence_table.csv"),
                "image_path": _relative_path(run_root, trend_dir / "trend_evidence_plot.png"),
                "trend_trace_path": _relative_path(run_root, trend_dir / "trend_trace.json"),
                "error_text": _read_optional_text(trend_dir / "error.txt"),
            }
        )
    success = not (context_dir / "error.txt").exists() and all(not item["error_text"] for item in trend_entries)
    return {
        "case_id": case_summary.get("case_id", case_dir.name),
        "abm": case_summary.get("abm", ""),
        "evidence_mode": case_summary.get("evidence_mode", ""),
        "prompt_variant": case_summary.get("prompt_variant", ""),
        "model": case_summary.get("model", ""),
        "resumed_from_existing": _lookup_resumed_flag(
            report=report,
            case_id=case_summary.get("case_id", case_dir.name),
        ),
        "success": success,
        "error_text": _read_optional_text(context_dir / "error.txt"),
        "paths": {
            "case_dir": _relative_path(run_root, case_dir),
            "context_prompt": _relative_path(run_root, inputs_dir / "context_prompt.txt"),
            "documentation": _relative_path(run_root, inputs_dir / "documentation.txt"),
            "parameters": _relative_path(run_root, inputs_dir / "parameters.txt"),
            "context_output": _relative_path(run_root, context_dir / "context_output.txt"),
            "context_trace": _relative_path(run_root, context_dir / "context_trace.json"),
            "review_csv": _relative_path(run_root, case_dir / "review.csv"),
            "validation_state": _relative_path(run_root, case_dir / "validation_state.json"),
        },
        "context_prompt_text": _read_optional_text(inputs_dir / "context_prompt.txt"),
        "documentation_text": _read_optional_text(inputs_dir / "documentation.txt"),
        "parameters_text": _read_optional_text(inputs_dir / "parameters.txt"),
        "trend_prompt_text": "",
        "table_csv_text": "",
        "context_output_text": _read_optional_text(context_dir / "context_output.txt"),
        "trend_output_text": "",
        "hyperparameters_text": "",
        "trends": trend_entries,
    }


def _lookup_resumed_flag(*, report: dict[str, Any], case_id: str) -> bool:
    cases = report.get("cases")
    if not isinstance(cases, list):
        return False
    for case in cases:
        if isinstance(case, dict) and case.get("case_id") == case_id:
            return bool(case.get("resumed_from_existing"))
    return False


def _read_optional_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _relative_path(run_root: Path, path: Path) -> str:
    if not path.exists():
        return ""
    return path.relative_to(run_root).as_posix()


def _render_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Run Review</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --panel: rgba(255, 252, 245, 0.92);
      --panel-strong: #fffdfa;
      --line: #d8cec0;
      --text: #1e1a15;
      --muted: #6f665d;
      --ok: #1b6b4d;
      --bad: #9b2c2c;
      --accent: #204d6d;
      --accent-soft: #e7eff4;
      --shadow: 0 10px 30px rgba(42, 34, 24, 0.08);
      --radius: 14px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.5 Georgia, "Iowan Old Style", serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(32, 77, 109, 0.08), transparent 26%),
        linear-gradient(180deg, #f7f4ee 0%, var(--bg) 100%);
    }}
    .app {{ display: grid; grid-template-columns: 360px 1fr; min-height: 100vh; }}
    .sidebar {{
      border-right: 1px solid rgba(216, 206, 192, 0.9);
      background: rgba(248, 244, 237, 0.9);
      backdrop-filter: blur(12px);
      padding: 18px;
      overflow: auto;
      position: sticky;
      top: 0;
      height: 100vh;
    }}
    .main {{ padding: 24px; overflow: auto; }}
    h1, h2, h3 {{ margin: 0 0 10px; font-weight: 600; }}
    h1 {{ font-size: 26px; letter-spacing: -0.03em; }}
    .meta {{ color: var(--muted); font-size: 12px; margin-bottom: 12px; }}
    .run-head {{
      border: 1px solid rgba(216, 206, 192, 0.9);
      background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(255,255,255,0.35));
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 16px;
      margin-bottom: 14px;
    }}
    .toolbar {{ display: grid; gap: 8px; margin-bottom: 12px; }}
    input, select {{
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(255, 255, 255, 0.95);
      color: var(--text);
    }}
    .case-list {{ display: grid; gap: 8px; }}
    .case-btn {{
      border: 1px solid rgba(216, 206, 192, 0.9);
      background: var(--panel);
      border-radius: 12px;
      box-shadow: var(--shadow);
      padding: 12px;
      text-align: left;
      cursor: pointer;
      transition: transform .12s ease, border-color .12s ease, background .12s ease;
    }}
    .case-btn:hover {{ transform: translateY(-1px); }}
    .case-btn.active {{
      border-color: var(--accent);
      background: linear-gradient(180deg, var(--panel-strong), var(--accent-soft));
      box-shadow: inset 0 0 0 1px rgba(32, 77, 109, 0.22), var(--shadow);
    }}
    .case-btn .top {{ display: flex; justify-content: space-between; gap: 8px; }}
    .badge {{
      font-size: 11px;
      padding: 3px 7px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255,255,255,0.7);
    }}
    .ok {{ color: var(--ok); border-color: color-mix(in srgb, var(--ok) 45%, white); }}
    .bad {{ color: var(--bad); border-color: color-mix(in srgb, var(--bad) 45%, white); }}
    .resumed {{ color: var(--accent); }}
    .grid {{ display: grid; gap: 16px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
    .hero {{
      border: 1px solid rgba(216, 206, 192, 0.9);
      border-radius: 18px;
      background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(248,244,237,0.75));
      box-shadow: var(--shadow);
      padding: 18px;
    }}
    .hero-top {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
      margin-bottom: 10px;
    }}
    .hero-title {{
      font-size: 24px;
      line-height: 1.15;
      letter-spacing: -0.03em;
    }}
    .chip-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }}
    .chip {{
      border: 1px solid rgba(216, 206, 192, 0.9);
      border-radius: 999px;
      padding: 4px 10px;
      background: rgba(255, 255, 255, 0.7);
      color: var(--muted);
      font-size: 12px;
    }}
    .card {{
      border: 1px solid rgba(216, 206, 192, 0.9);
      border-radius: var(--radius);
      background: var(--panel);
      box-shadow: var(--shadow);
      padding: 14px;
    }}
    .label {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .04em;
      margin-bottom: 4px;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    .detail-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    img {{
      max-width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: white;
      box-shadow: 0 4px 18px rgba(42, 34, 24, 0.08);
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .empty {{ color: var(--muted); font-style: italic; }}
    .card pre {{
      max-height: 320px;
      overflow: auto;
      padding-top: 2px;
    }}
    @media (max-width: 1100px) {{
      .app {{ grid-template-columns: 1fr; }}
      .sidebar {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .summary-grid, .detail-grid {{ grid-template-columns: 1fr; }}
      .hero-top {{ flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="run-head">
        <h1>Run Review</h1>
        <div class="meta" id="run-meta"></div>
      </div>
      <div class="toolbar">
        <input id="search" type="search" placeholder="Filter cases">
        <select id="status-filter">
          <option value="all">All cases</option>
          <option value="ok">Only successful</option>
          <option value="failed">Only failed</option>
        </select>
      </div>
      <div class="case-list" id="case-list"></div>
    </aside>
    <main class="main">
      <div id="details"></div>
    </main>
  </div>
  <script id="viewer-data" type="application/json">__VIEWER_DATA__</script>
  <script>
    const data = JSON.parse(document.getElementById("viewer-data").textContent);
    const caseList = document.getElementById("case-list");
    const details = document.getElementById("details");
    const search = document.getElementById("search");
    const statusFilter = document.getElementById("status-filter");
    const runMeta = document.getElementById("run-meta");
    let currentCaseId = data.cases[0] ? data.cases[0].case_id : null;

    runMeta.innerHTML = [
      data.run_root,
      `success: ${data.success}`,
      data.run_log_path ? `<a href="${escapeHtmlAttr(relPath(data.run_log_path))}">run log</a>` : "",
      data.report_path ? `<a href="${escapeHtmlAttr(relPath(data.report_path))}">report json</a>` : ""
    ].filter(Boolean).join("<br>");

    function relPath(absoluteOrRunRelative) {{
      if (!absoluteOrRunRelative) return "";
      if (
        absoluteOrRunRelative.includes("/runs/")
        || absoluteOrRunRelative.startsWith("cases/")
      ) {{
        return absoluteOrRunRelative.replace(/^.*?openrouter_smoke_latest\\//, "");
      }}
      return absoluteOrRunRelative;
    }}

    function escapeHtml(text) {{
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    function escapeHtmlAttr(text) {{
      return escapeHtml(text).replaceAll('"', "&quot;");
    }}

    function filteredCases() {{
      const q = search.value.trim().toLowerCase();
      const status = statusFilter.value;
      return data.cases.filter((item) => {{
        const hay = [
          item.case_id,
          item.abm,
          item.evidence_mode,
          item.prompt_variant,
          item.error_text
        ].join(" ").toLowerCase();
        const matchesText = !q || hay.includes(q);
        const matchesStatus = status === "all" || (status === "ok" ? item.success : !item.success);
        return matchesText && matchesStatus;
      }});
    }}

    function renderList() {{
      const items = filteredCases();
      if (!items.some((item) => item.case_id === currentCaseId)) {{
        currentCaseId = items[0] ? items[0].case_id : null;
      }}
      caseList.innerHTML = items.map((item) => `
        <button
          class="case-btn ${item.case_id === currentCaseId ? "active" : ""}"
          data-case-id="${escapeHtmlAttr(item.case_id)}"
        >
          <div class="top">
            <strong>${escapeHtml(item.case_id)}</strong>
            <span class="badge ${item.success ? "ok" : "bad"}">${item.success ? "ok" : "failed"}</span>
          </div>
          <div>${escapeHtml(item.abm)} · ${escapeHtml(item.evidence_mode)} · ${escapeHtml(item.prompt_variant)}</div>
          <div class="meta">
            ${item.resumed_from_existing ? '<span class="resumed">reused</span>' : 'fresh'}
            ${item.error_text ? " · " + escapeHtml(item.error_text) : ""}
          </div>
        </button>
      `).join("");
      caseList.querySelectorAll(".case-btn").forEach((button) => {{
        button.addEventListener("click", () => {{
          currentCaseId = button.dataset.caseId;
          renderList();
          renderDetails();
        }});
      }});
      renderDetails();
    }}

    function renderDetails() {{
      const item = data.cases.find((entry) => entry.case_id === currentCaseId);
      if (!item) {{
        details.innerHTML = '<div class="card empty">No matching case.</div>';
        return;
      }}
      details.innerHTML = `
        <div class="grid">
          <div class="hero">
            <div class="hero-top">
              <div>
                <div class="hero-title">${escapeHtml(item.case_id)}</div>
                <div class="meta">${escapeHtml(item.abm)}</div>
              </div>
              <span class="badge ${item.success ? "ok" : "bad"}">${item.success ? "ok" : "failed"}</span>
            </div>
            <div class="chip-row">
              <span class="chip">evidence: ${escapeHtml(item.evidence_mode)}</span>
              <span class="chip">prompt: ${escapeHtml(item.prompt_variant)}</span>
              <span class="chip">model: ${escapeHtml(item.model || "n/a")}</span>
              <span class="chip">${item.resumed_from_existing ? "reused from prior run" : "fresh in this run"}</span>
            </div>
          </div>
          <div class="summary-grid">
            ${summaryCard("Case dir", item.paths.case_dir)}
            ${summaryCard("Context prompt", item.paths.context_prompt)}
            ${summaryCard("Trend prompt", item.paths.trend_prompt)}
            ${summaryCard("Run log", data.run_log_path ? relPath(data.run_log_path) : "n/a")}
          </div>
          ${
            item.error_text
              ? `<div class="card"><span class="label">Error</span><pre>${escapeHtml(item.error_text)}</pre></div>`
              : ""
          }
          <div class="detail-grid">
            ${textCard("Parameters", item.paths.parameters, item.parameters_text)}
            ${textCard("Documentation", item.paths.documentation, item.documentation_text)}
            ${textCard("Context Prompt", item.paths.context_prompt, item.context_prompt_text)}
            ${textCard("Context Output", item.paths.context_output, item.context_output_text)}
            ${item.trends && item.trends.length
              ? textCard("Validation State", item.paths.validation_state, "")
              : textCard("Trend Prompt", item.paths.trend_prompt, item.trend_prompt_text)}
            ${item.trends && item.trends.length
              ? textCard("Case Review CSV", item.paths.review_csv, "")
              : textCard("Trend Output", item.paths.trend_output, item.trend_output_text)}
            ${item.trends && item.trends.length
              ? ""
              : textCard("Hyperparameters", item.paths.hyperparameters, item.hyperparameters_text)}
            ${item.trends && item.trends.length
              ? ""
              : textCard("Evidence Table", item.paths.table_csv, item.table_csv_text)}
          </div>
          <div class="detail-grid">
            ${item.trends && item.trends.length
              ? textCard("Trace Files", item.paths.context_trace, "")
              : imageCard("Evidence Plot", item.paths.image)}
            ${item.trends && item.trends.length
              ? linkCard("Case Files", [
                  ["context trace", item.paths.context_trace],
                  ["review csv", item.paths.review_csv],
                  ["validation state", item.paths.validation_state]
                ])
              : linkCard("Trace Files", [
                  ["context trace", item.paths.context_trace],
                  ["trend trace", item.paths.trend_trace]
                ])}
          </div>
          ${item.trends && item.trends.length ? renderTrendSections(item.trends) : ""}
        </div>
      `;
    }}

    function summaryCard(label, value) {{
      return `
        <div class="card">
          <span class="label">${escapeHtml(label)}</span>
          <div>${escapeHtml(value || "n/a")}</div>
        </div>
      `;
    }}

    function textCard(label, path, text) {{
      const body = text ? `<pre>${escapeHtml(text)}</pre>` : '<div class="empty">not present</div>';
      const link = path ? `<div class="meta"><a href="${escapeHtmlAttr(path)}">${escapeHtml(path)}</a></div>` : "";
      return `<div class="card"><span class="label">${escapeHtml(label)}</span>${link}${body}</div>`;
    }}

    function imageCard(label, path) {{
      if (!path) {{
        return `
          <div class="card">
            <span class="label">${escapeHtml(label)}</span>
            <div class="empty">not present</div>
          </div>
        `;
      }}
      return `
        <div class="card">
          <span class="label">${escapeHtml(label)}</span>
          <div class="meta"><a href="${escapeHtmlAttr(path)}">${escapeHtml(path)}</a></div>
          <img src="${escapeHtmlAttr(path)}" alt="${escapeHtmlAttr(label)}">
        </div>
      `;
    }}

    function linkCard(label, links) {{
      const items = links
        .filter(([, path]) => path)
        .map(([name, path]) => `<div><a href="${escapeHtmlAttr(path)}">${escapeHtml(name)}</a></div>`)
        .join("");
      return `
        <div class="card">
          <span class="label">${escapeHtml(label)}</span>
          ${items || '<div class="empty">not present</div>'}
        </div>
      `;
    }}

    function renderTrendSections(trends) {{
      return `
        <div class="grid">
          ${trends.map((trend) => `
            <div class="card">
              <span class="label">${escapeHtml(trend.plot_id)}</span>
              ${
                trend.error_text
                  ? `<div class="meta" style="color: var(--bad);">${escapeHtml(trend.error_text)}</div>`
                  : ""
              }
              <div class="detail-grid">
                ${textCard("Trend Prompt", trend.trend_prompt_path, trend.trend_prompt_text)}
                ${textCard("Trend Output", trend.trend_output_path, trend.trend_output_text)}
                ${textCard("Evidence Table", trend.table_csv_path, trend.table_csv_text)}
                ${imageCard("Evidence Plot", trend.image_path)}
              </div>
              <div class="meta" style="margin-top: 10px;">
                ${trend.trend_trace_path ? `<a href="${escapeHtmlAttr(trend.trend_trace_path)}">trend trace</a>` : ""}
              </div>
            </div>
          `).join("")}
        </div>
      `;
    }}

    search.addEventListener("input", renderList);
    statusFilter.addEventListener("change", renderList);
    renderList();
  </script>
</body>
</html>
"""
    return template.replace("__VIEWER_DATA__", data_json)
