# Checkpoint

## Invariants

- **Goal:** Remove leftover README author notes after API examples.
- **Acceptance:** no “Adjust the example response”; JSON examples remain; unittest 6/6; Chrome N/A.
- **Non-goals:** hashed lockfile; rewriting examples; skip_specs.
- **Constraints:** `main`; push allowed.
- **Decisions:** delete notes; keep examples; skip design.

## Current State

- **Phase:** propose complete; next apply
- **Hypothesis:** delete notes → phrase gone; tests still 0
- **Expected signal:** grep miss on Adjust; unittest 6/6
- **Rollback:** restore paragraph
- **Tasks:** 1.1 pending
- **Retry count:** 0
- **Confidence:** high
- **Next action:** apply 1.1

## Facts

- README lines 106–108 leftover notes
- test_parse_transcription_readme_segments matches the success example

## Assumptions

- None

## Open questions

- None

## Events

- 2026-09-04 selected README leftover notes (docs correctness over hashed lockfile)
