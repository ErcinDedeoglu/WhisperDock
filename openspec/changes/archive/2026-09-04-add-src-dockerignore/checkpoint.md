# Checkpoint

## Invariants

- **Goal:** Context root ignores unittest and pycache.
- **Acceptance:** probe COPY test_app.py fails; whisper-cli; unittest 8/8; Chrome N/A.
- **Non-goals:** Ignoring whisper tests/examples.
- **Constraints:** Push `main` allowed; do not tag/push `dublok/*` locally; local tag `whisperdock-local:pinned`.
- **Decisions:** skip_specs; ignore three patterns only.

## Current State

- **Phase:** apply
- **Hypothesis:** src/.dockerignore → probe COPY fails; service image still builds.
- **Expected signal:** probe fail; cli-ok.
- **Tasks:** 1.1 pending, 2.1 pending
- **Retry count:** 0
- **Next action:** Write dockerignore, probe, rebuild.

## Events

- 2026-09-04: Proposed add-src-dockerignore with skip_specs.
