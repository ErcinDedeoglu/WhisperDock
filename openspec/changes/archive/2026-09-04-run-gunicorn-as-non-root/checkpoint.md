# Checkpoint

## Invariants

- **Goal:** Run gunicorn as non-root uid 10001 (`app`).
- **Acceptance:** inspect User=`app`; id uid!=0; unittest 6/6; Chrome N/A.
- **Non-goals:** gosu, chown /app, gunicorn body limits.
- **Constraints:** `main`; push allowed; do not tag `dublok/*` locally.
- **Decisions:** explicit 10001; `--no-log-init`; no chown.

## Current State

- **Phase:** propose complete; next apply
- **Hypothesis:** USER app → inspect User=app; id not 0
- **Expected signal:** uid 10001
- **Rollback:** remove useradd/USER
- **Tasks:** pending
- **Retry count:** 0
- **Confidence:** medium until rebuild
- **Next action:** apply 1.1

## Facts

- Current image User empty; id root
- Bind port 5000

## Assumptions

- /tmp writable by other; binaries world-executable

## Open questions

- None

## Events

- 2026-09-04 selected non-root USER (security; gunicorn-as-root evidence)
