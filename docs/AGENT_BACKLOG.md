# Agent Backlog

This document lists the highest-value repository improvements that make future agent work safer, faster, and easier to verify. It is intentionally short and pragmatic. Items here are not feature requests; they are maintainability and verification improvements.

## Highest Priority

### 1. Canonical Quality Gate Command

The repository already has `validate-workspace`, but there is still room to make it the single obvious post-change command for most edits.

Desired end state:

- one command for the common local quality loop
- clear profiles for narrow versus broad verification
- predictable machine-readable output
- explicit mapping from changed subsystem to recommended profile

Why it matters:

- reduces ad hoc command selection
- lowers the chance that an agent verifies the wrong thing
- shortens review handoff

### 2. Explicit Output-Quality Validators

Some workflows already detect malformed or obviously unavailable model outputs. The next step is to centralize and strengthen those validators.

Desired end state:

- one reusable validator surface for LLM outputs
- explicit failure categories such as:
  - truncated
  - malformed structured output
  - generic unavailable response
  - pathological length
- validator results written into run state so resume can reuse them directly

Why it matters:

- closes the loop between generation, human review, and rerun
- makes bad outputs fail consistently instead of being handled piecemeal

### 3. Reviewer Decisions Feeding Resume

The current run model supports resume and reuse. It should eventually support reviewer decisions as first-class state.

Desired end state:

- human marks case or plot as:
  - accepted
  - rejected
  - rerun
  - wrong evidence
  - bad output
- next resume uses those decisions directly

Why it matters:

- reduces manual cleanup
- makes long runs more practical
- makes review reproducible instead of informal

## Medium Priority

### 4. Golden Tests for Prompts and Run Layout

The codebase now relies heavily on prompt rendering, run layout, and machine-readable review artifacts.

Desired end state:

- golden tests for representative rendered prompts
- golden tests for case folder layouts
- regression tests for `review.csv`, `review.html`, and run logs

Why it matters:

- makes safe refactoring much easier
- catches accidental contract drift early

### 5. Clearer Runtime State Machine

Run and case statuses exist, but they are not yet modeled as one explicit state machine across workflows.

Desired end state:

- explicit states such as:
  - pending
  - running
  - succeeded
  - failed
  - invalid
  - accepted
  - needs_retry

Why it matters:

- simplifies resume logic
- improves monitor behavior
- makes reporting easier to reason about

### 6. Further Separation of Concerns

Large workflow modules have already been reduced, but some still mix orchestration, persistence, validation, and rendering.

Desired end state:

- clearer module boundaries between:
  - orchestration
  - persistence
  - validation
  - rendering
- fewer cross-imported private helpers

Why it matters:

- lowers change risk
- makes narrow testing easier

## Lower Priority

### 7. Stronger Typing for Intermediate Payloads

Some internal payloads are still anonymous dictionaries.

Desired end state:

- typed models for key request, trace, and review payloads
- less `dict[str, Any]` in workflow internals

Why it matters:

- improves static analysis
- reduces field-name drift and implicit contracts

### 8. More Artifact Invariant Tests

The repository already has useful artifact checks. Additional invariants would make refactors safer.

Desired end state:

- stronger assertions for:
  - required files per run type
  - required fields in report files
  - consistency between logs, review rows, and case folders

Why it matters:

- helps preserve the review/debug workflow without manual spot checks
