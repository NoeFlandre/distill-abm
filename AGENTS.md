# AGENTS.md

Operating instructions for coding agents.

---

## Core Principle

Deliver **working, verified, minimal, maintainable changes**. Code is cheap to generate; good code is not. Optimize for correctness, clarity, and reviewability.

---

## Non-Negotiable Rules

1. **Never assume generated code works.** If it hasn't been executed, it's unverified. Always run what you change.
2. **First run the tests.** Before touching code: find the test commands, run them, understand the baseline.
3. **Prefer red/green TDD.** Write a failing test → confirm it fails → implement the smallest fix → confirm it passes → expand.
4. **Keep changes minimal and reviewable.** Small PRs, small commits, clear boundaries. Split large work into reviewable slices.
5. **Don't hand off unreviewed code.** Before handoff: code works, you've reviewed it, the change is minimal, the description is accurate.
6. **Stop and ask when ambiguity is material.** Don't guess on prompt semantics, evaluation criteria, data formats, or anything the human said not to guess about. Ask one concise question, wait, then continue.
7. **New smoke runs must use the standard run scaffolding.** By default every smoke workflow should be run-separated and resumable, with `runs/run_<timestamp>/`, `latest_run.txt`, per-run logs, review artifacts, and reuse of previously successful work. Do not invent one-off layouts unless explicitly justified.
8. **Read the technology documentation when behavior is provider- or framework-specific.** Before changing external API, SDK, model-provider, or framework integrations, check the relevant official docs so transport details and parameters are not guessed.

---

## Operating Loop

### 1. Understand
Restate the goal. Identify constraints, relevant files, existing tests, and how success will be verified.

### 2. Inspect
Read implementation files, tests, configs, docs, and nearby patterns. Tests are often the fastest route to understanding.

### 3. Baseline
Run existing tests. If there's a runnable app, get it running.

### 4. Test First
Encode expected behavior in a test before implementing. If automation isn't feasible, define a manual test procedure.

### 5. Implement Minimally
Smallest change that satisfies the requirement. No speculative abstraction. Respect YAGNI.

### 6. Execute & Verify
After each change: run tests, run manual checks, inspect outputs, fix issues.

### 7. Document
Leave evidence of what changed, why, how it was verified, and any tradeoffs. Prefer artifacts (transcripts, screenshots, demo docs) over claims.

### 8. Reduce Cognitive Debt
If the result is hard to understand, create walkthroughs, notes, or interactive explanations immediately.

---

## Testing Policy

- **Automated tests are part of the implementation**, not optional polish.
- Use the **narrowest test** that proves the point (unit → integration → e2e).
- **Convert discovered bugs** into permanent regression tests.
- **Browser tests** are worth maintaining for critical UI flows.

### Manual Testing

Automated tests are necessary but not sufficient. Also perform:

- **Libraries/functions:** Exercise with `python -c`, temporary scripts, direct calls with edge-case inputs.
- **APIs:** Start dev server, use `curl`, test happy paths and error cases, check status codes and payloads.
- **Web UIs:** Use browser automation (Playwright, Rodney, etc.). Test layout, interactions, state transitions, error messages, accessibility.
- **Always inspect screenshots** when available—fastest way to catch visual bugs.
- **Show your work:** Record commands run, outputs observed, screenshots inspected, conclusions drawn.

---

## Evidence & Demos

Don't just claim you tested—**produce evidence**.

### Showboat Pattern
Create executable demo documents (Markdown) that capture:
- Narrative notes
- Commands executed with their output (use `exec`, not hand-typed results)
- Screenshots and images

Place in `notes/` (e.g., `notes/api-demo.md`, `notes/feature-walkthrough.md`). **Never fabricate results**—rerun if reality changes.

### Rodney Pattern
For web apps, use browser automation CLI to:
- Drive multi-step browser sessions
- Inspect/modify DOM state, run page JS
- Capture screenshots for verification and documentation

### Combined Workflow
Drive browser with Rodney → capture screenshots → document steps in Showboat → produce a durable demo artifact.

### Tool Discovery
Run `--help` on any CLI tool before using it. Read what it does and how to compose its commands.

---

## Documentation & Explanation

- **Leave evidence, not claims:** command transcripts, screenshots, sample outputs, demo Markdown.
- **Walkthroughs:** For nontrivial features, produce file-by-file explanations with real code excerpts (obtained mechanically, not from memory).
- **Interactive explanations:** For complex algorithms, build small HTML demos, visualizers, or stepping tools to make invisible mechanics visible.
- **Preserve artifacts:** Good explanations become documentation, debugging aids, and teaching resources.

---

## Collaboration & PRs

- Reviewer time is precious. Don't waste it on validation you should have done.
- PR descriptions must be **truthful**: goal, scope, how tested, tradeoffs, links to demos/issues.
- Use **one branch + one worktree per PR/task** so isolated work stays reviewable and cleanup is straightforward.
- Split work aggressively into smaller commits and PRs.

---

## Handling Uncertainty

- **When unsure:** Read files, run code, probe APIs, drive the browser, capture evidence.
- **When design is unclear:** Build cheap prototypes and spikes. Keep useful ones.
- **When behavior is subtle:** Add logging, create repros, build visualizers, generate walkthroughs.

---

## Defaults for Small Tools & Prototypes

For single-file demos, utilities, and visual explainers:
- One self-contained HTML file, vanilla JS, minimal CSS
- No React or build steps unless clearly needed
- Mobile-usable inputs, clear headings, readable typography
- Favor direct manipulation, drag-drop, previews, download buttons where useful

---

## Knowledge Hoarding

Capture solutions so the team compounds knowledge:
- Reusable scripts, proof-of-concept repos, TIL notes, demo pages, build scripts, research memos
- **Working examples > vague claims.** "Here's a working example" beats "it should be possible."

### Repository Memory

- **Smoke workflows are pipeline audits, not side experiments.** New smoke runs should reuse the core pipeline contracts: run-separated outputs, resumability, logging, review artifacts, and the same prompt/evidence/report logic used by the implementation they audit.
- **Context reuse is expected whenever prompts are identical.** In the current DOE/full-case setup, context variation is much smaller than trend variation: context effectively changes by ABM and `role` on/off; `example` and `insights` are trend-only. Cache and reuse context outputs by the fully rendered prompt, not by loose labels.
- **Keep primary scoring stable and add secondary references additively.** The primary author ground truth remains the main score contract. Optional modeler references belong in `experiment_settings.modeler_ground_truth` and should add parallel score blocks/columns/metadata, not replace the primary scores.
- **Be careful with integration tests that touch scoring.** If the goal is reference scoring or metadata/report wiring, prefer `full_text_only` unless the test is explicitly about summarizers. This avoids accidentally invoking heavy summarizer/model dependencies and keeps the test focused.
- **Actively clean stale processes.** Long-running smokes, monitors, and stuck test processes can make the machine lag badly. Periodically check for stale `uv`, `pytest`, monitor, and smoke processes and stop the ones that are no longer useful. Do not leave background process cleanup as an afterthought.
- **Change evidence semantics at the shared helper layer.** If reviewer or product guidance changes what an evidence mode means, implement it once in the core pipeline/evidence helpers so DOE, smoke workflows, and real runs stay aligned.
- **`table` means plot-relevant statistical evidence.** The non-plot evidence mode should be computed only from the simulation columns relevant to the target trend description, plus an optional step/time column. Do not dump unrelated simulation columns into prompts.
- **Bound heavy statistical routines on repeated-simulation series.** If one reporter pattern matches many replicate series, keep the selected slice for auditability but run the expensive signal analyses on a representative aggregate series (currently the tick-wise mean) so DOE and other audit runs stay fast and finish reliably.
- **Keep the LLM runtime API-only.** Production models are `moonshotai/kimi-k2.5`, `google/gemini-3.1-pro-preview`, and OpenRouter `qwen/qwen3.5-27b`. `nvidia/nemotron-nano-12b-v2-vl:free` and `mistral-medium-latest` are debug-only. Do not reintroduce local-model runtime paths unless a task explicitly asks for them.
- **Provider-specific scheduling belongs in the shared runners.** If a provider has a clear request budget, encode pacing and worker limits in the shared smoke/runtime runners rather than in one-off scripts or ad hoc shell throttling.
- **Long-running smoke suites should ship a simple HTML dashboard.** Logs and CSVs remain the source of truth, but top-level all-ABM smoke suites should also expose a small static dashboard for quick operator review during and after the run.
- **Long-running smoke suites should expose one stable root-level progress contract while they are running.** Prefer a root `suite_progress.json` plus a terminal-first monitor path that reads it and the nested ABM run state, so humans can monitor the suite from one place without drilling into nested ABM run roots.
- **Keep monitor/rendering boundaries explicit.** `pipeline/local_qwen_monitor.py` should stay focused on terminal rendering and interaction, while filesystem snapshot collection belongs in `pipeline/local_qwen_monitor_snapshots.py`.
- **Keep stable artifact writers centralized when schemas repeat.** Shared contracts like per-case full-case review CSVs, suite current-view sync, and paired JSON/Markdown smoke reports should be written through small helper modules so layout changes happen in one place.
- **Treat viewer payload building as a separate typed boundary.** `run_viewer_payloads.py` owns the JSON payload contract for `review.html`; avoid rebuilding that structure ad hoc inside HTML rendering code.

---

## Writing Policy

- Don't ghostwrite personal opinion in someone else's voice unless asked.
- Do improve documentation, notes, and mechanical text.
- Do proofread for typos, grammar, logic, factual errors, weak arguments, broken links.

---

## Definition of Done

A task is done when (as applicable):
- [ ] Right files changed, change is minimal but sufficient
- [ ] Tests added/updated and passing
- [ ] Behavior manually exercised, edge cases checked
- [ ] Docs/comments updated if behavior changed
- [ ] Implementation is understandable
- [ ] Evidence exists of what was tested and what happened

---

## Quick Checklists

**Start of task:** What's being asked? What constraints? What files? What tests exist? How will I verify? What artifact will I leave?

**During implementation:** Test first? Smallest change? No speculative complexity? Executed the code? Checked edge cases? Captured evidence?

**Pre-handoff:** Reviewed my own code? Description matches reality? Tests documented? Demo artifact if helpful? Change small enough for review? Understandable by future humans/agents?

---

## Compact Policy

> Build the smallest correct change. Run existing tests first. Prefer red/green TDD. Never assume code works without executing it. Manually test as appropriate: direct execution, curl for APIs, browser automation for UIs. Convert found bugs into automated tests. Keep changes simple, reviewable, documented. Produce evidence artifacts (demo docs, screenshots, transcripts). When something is hard to understand, create walkthroughs or interactive explanations. Preserve useful scripts, demos, and examples for reuse.
