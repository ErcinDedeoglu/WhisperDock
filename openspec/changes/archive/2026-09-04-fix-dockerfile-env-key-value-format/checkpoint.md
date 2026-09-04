# Checkpoint

## Invariants

- **Goal:** Convert Dockerfile ENV to `key=value` so LegacyKeyValueFormat warnings go away.
- **Acceptance:** `--check` 0 LegacyKeyValueFormat; unittest 6/6; GHA no annotation on Dockerfile:5/6; Chrome N/A.
- **Non-goals:** combining ENV, skip comments, unused socket, other lints.
- **Constraints:** `main`; push allowed; no secrets; no force-push.
- **Decisions:** Two-line `=` conversion; verify with `docker build --check`.

## Current State

- **Phase:** propose complete; next apply
- **Hypothesis:** `ENV KEY=1` → `--check` clean; GHA annotation gone
- **Expected signal:** no LegacyKeyValueFormat locally or on GHA
- **Rollback:** restore space-separated ENV
- **Tasks:** 1.1 pending; 2.1 pending
- **Retry count:** 0
- **Confidence:** high local `--check`; GHA unproven
- **Next action:** apply 1.1

## Facts

- Baseline `docker build --check` = 2 LegacyKeyValueFormat warnings (lines 5–6)
- Unittest 6/6 before edit

## Assumptions

- Values `1` parse identically with `=`

## Open questions

- None

## Events

- 2026-09-04 selected LegacyKeyValueFormat from GHA 33848114073
- 2026-09-04 research: Docker LegacyKeyValueFormat + ENV reference
