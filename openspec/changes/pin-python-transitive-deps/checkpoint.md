# Checkpoint

## Invariants

- **Goal:** Freeze Flask/gunicorn transitives in `src/requirements.txt` and install `--no-deps -r`.
- **Acceptance:** eight == pins; no inline Flask pip; image metadata matches; unittest 6/6; Chrome N/A.
- **Non-goals:** hashes, pip-tools, apt pins.
- **Constraints:** `main`; push allowed; do not tag `dublok/*` locally.
- **Decisions:** freeze without hashes; `--no-deps`; COPY requirements before app.py.

## Current State

- **Phase:** propose complete; next apply
- **Hypothesis:** freeze + `--no-deps -r` → image versions match freeze
- **Expected signal:** metadata 3.1.3/26.2.0/3.1.8/...
- **Rollback:** restore inline pip; delete requirements.txt
- **Tasks:** all pending
- **Retry count:** 0
- **Confidence:** medium until image rebuild
- **Next action:** apply 1.1

## Facts

- Freeze from whisperdock-local:pinned: blinker 1.9.0 click 8.5.0 Flask 3.1.3 gunicorn 26.2.0 itsdangerous 2.2.0 Jinja2 3.1.6 MarkupSafe 3.0.3 Werkzeug 3.1.8

## Assumptions

- Those eight packages are the full runtime set gunicorn needs

## Open questions

- None

## Events

- 2026-09-04 selected unpinned transitives; rejected hashes this slice (arch wheels)
