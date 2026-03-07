# Manual validation evidence

*2026-03-06T18:21:24Z by Showboat 0.6.1*
<!-- showboat-id: b49c7cf9-5e44-464e-a82e-77d61dc1628d -->

This document records a concrete verification pass for the current repository state. It uses Showboat so the key commands and their observed outputs are preserved together. Rodney was not used for this pass because the repository does not currently expose a browser UI surface that needs browser automation.

```bash
uv run ruff check src tests
```

```output
All checks passed!
```

```bash
uv run mypy src tests
```

```output
Success: no issues found in 96 source files
```

```bash
uv run pytest -q | perl -pe 's/ in [0-9]+\.[0-9]+s$/ in <elapsed>/'
```

```output
........................................................................ [ 27%]
........................................................................ [ 54%]
........................................................................ [ 81%]
................................................                         [100%]
264 passed in <elapsed>
```

The main pre-LLM debug surfaces are the ingest smoke, visualization smoke, and DOE smoke result trees under results/. The DOE smoke is strictly pre-LLM: it materializes prompts, evidence references, and request plans without creating adapters, checking local model runtime availability, or sending any model request.

Evidence locations: results/ingest_smoke_latest/, results/viz_smoke_latest/, results/doe_smoke_latest/, and results/agent_validation/latest/. Reviewers can inspect `00_overview/` first, then `10_shared/global/`, then `10_shared/<abm>/`, and finally `20_case_index/` only when they need compact case-level or request-level indexing.

```bash
python -c "import json; from pathlib import Path; data=json.loads(Path('results/doe_smoke_latest/00_overview/doe_smoke_report.json').read_text()); print('total_cases=', data['total_cases']); print('total_planned_requests=', data['total_planned_requests']); print('failed_cases=', len(data['failed_case_ids'])); print('shared_abms=', ','.join(sorted(data['abm_shared'])));"
```

```output
total_cases= 3240
total_planned_requests= 42120
failed_cases= 1080
shared_abms= fauna,grazing,milk_consumption
```
