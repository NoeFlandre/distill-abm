# AGENTS.md

## Purpose

This repository is maintained with an **agentic engineering** workflow focused on improving software quality without unnecessary churn.

The default mindset is:

- **Code is cheap. Good code is not.**
- Optimize for **correctness, clarity, simplicity, maintainability, documentation, and testability**.
- Prefer **small, verified, low-risk changes** over ambitious rewrites.
- Do not confuse output volume with engineering progress.

---

## Core Operating Principles

### 1. Preserve behavior unless explicitly asked otherwise
Agents working in this codebase should **not add features, remove features, or change intended behavior** unless the task explicitly requires it.

When improving code:

- preserve external behavior
- preserve public interfaces unless there is a clear, approved reason to change them
- treat existing tests, docs, and usage patterns as the current behavioral contract
- if behavior appears incorrect, confirm it with evidence before changing it

### 2. Start with reality, not assumptions
Before making changes:

1. inspect the repository structure
2. read the relevant documentation
3. read any local guidance files
4. locate the commands for:
   - tests
   - lint
   - type-checking
   - build
5. run the existing tests first

Do not propose broad changes before understanding how the system actually works.

### 3. Use the codebase as the source of truth
Prefer:

- existing patterns already used in the repository
- established helpers and utilities
- existing naming conventions
- existing test styles
- existing architecture decisions

Avoid introducing novelty when the repository already has a workable precedent.

### 4. Keep changes small and reviewable
Break work into the smallest meaningful increments.

For each increment:

- identify the quality problem
- write or update tests when needed
- make the smallest safe change
- verify the result
- document what changed and why

Avoid large rewrites unless they are clearly justified and safely verifiable.

### 5. Use TDD where behavior matters
When touching logic that could affect behavior, prefer:

1. write a failing test for the intended behavior or regression
2. confirm it fails for the expected reason
3. implement the smallest fix
4. rerun the tests until green
5. refactor while keeping tests green

Tests are not just verification; they are a tool for understanding and controlling change.

### 6. Explain the code back to humans
A good agent reduces cognitive debt.

As you work:

- summarize how the relevant subsystem works
- identify invariants and assumptions
- note risky areas and unclear boundaries
- leave behind code and documentation that are easier for a human to understand

Do not leave behind "mystery improvements".

### 7. Verify continuously
Never claim a change works without running the relevant checks.

Prefer narrow verification first, then broader verification:

- targeted tests
- file/module level checks
- full test suite
- lint/type-check/build, if relevant

Be explicit about what was and was not verified.

### 8. Documentation must track reality
When code changes, update any affected:

- README sections
- docstrings
- inline comments
- examples
- architecture notes
- developer workflows
- test names and descriptions

Documentation should explain **why** the code is structured the way it is, not merely restate what the code obviously does.

### 9. Favor simplicity over cleverness
Prefer code that is:

- obvious
- local
- composable
- easy to test
- easy to delete
- easy to explain

Avoid:

- speculative abstractions
- deeply nested control flow
- magic behavior
- hidden coupling
- unnecessary indirection

### 10. Be honest about uncertainty
If something is unclear:

- say what you know
- say what you inferred
- say what you verified
- say what remains uncertain

Do not bluff understanding.

---

## Default Agent Workflow

When assigned a task, follow this order unless the task explicitly says otherwise.

### Phase 1: Recon
- Read this file and any other repository guidance.
- Inspect the relevant files and module boundaries.
- Identify test/lint/type-check/build commands.
- Run the existing tests first.
- Summarize the current state before editing.

### Phase 2: Plan
- Propose a short, concrete plan.
- Prefer low-risk, high-value quality improvements first.
- Reuse existing patterns from the repository.

### Phase 3: Execute in small loops
For each loop:

1. define the specific problem
2. add or adjust tests when appropriate
3. make the smallest safe change
4. run verification
5. summarize the outcome

### Phase 4: Report
At the end, provide:

- what changed
- what behavior was intentionally preserved
- tests added or updated
- checks run and their outcomes
- docs updated
- remaining risks or follow-up opportunities

---

## What "Quality Improvement" Means Here

Quality work may include:

- simplifying complex functions
- reducing duplication
- improving naming
- improving type safety
- clarifying boundaries between modules
- adding missing regression tests
- improving error handling
- tightening validation
- improving developer documentation
- removing dead or confusing internal code where behavior is preserved

Quality work does **not** automatically justify:

- redesigning the architecture
- replacing stable code with novel abstractions
- changing API shape
- altering user-facing behavior
- introducing large dependency churn

---

## Guardrails

Unless explicitly requested, do **not**:

- add features
- remove features
- silently change behavior
- rewrite large subsystems without a strong reason
- introduce unnecessary dependencies
- leave placeholder implementations
- leave broad TODOs without explanation
- skip tests when tests are available
- claim success without verification

### Protected Documentation Files

The following files should **not be modified** unless explicitly asked to do so:

- `docs/TESTING_REPORT.md`
- `docs/GROUND_TRUTHS_GPT5.2.md`

If a task requires modifying these files, you **must ask for confirmation** from the human operator before proceeding.

---

## Preferred Change Style

When editing code, prefer:

- smaller functions
- clearer names
- explicit data flow
- fewer hidden side effects
- better test coverage around risky logic
- comments that explain intent, not trivia
- reuse of existing patterns already present in the codebase

---

## Preferred Reporting Style

When updating the human operator:

- be concise
- be concrete
- reference actual files and commands
- separate facts from guesses
- show evidence from tests and checks
- mention tradeoffs and risks

Good status updates should help a human quickly answer:

- What did the agent learn?
- What changed?
- Why was it safe?
- How was it verified?
- What still needs attention?

---

## Decision Rule

When multiple valid options exist, choose the one that is:

1. simplest
2. easiest to verify
3. most consistent with the existing codebase
4. easiest for a future human maintainer to understand

---

## Final Standard

Leave the codebase:

- cleaner than you found it
- better tested than before
- better explained than before
- no more complex than necessary
- behaviorally consistent unless change was explicitly required
