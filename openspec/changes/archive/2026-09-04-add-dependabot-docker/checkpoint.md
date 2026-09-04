# Checkpoint

## Invariants

- **Goal:** Dependabot docker watches `/src` weekly.
- **Acceptance:** YAML has docker + `/src`; unittest 8/8; no Docker Images run; Chrome N/A.
- **Non-goals:** github-actions, pip, Dockerfile edits.
- **Constraints:** Push `main` allowed; do not tag/push `dublok/*` locally.
- **Decisions:** directory `/src`; weekly.

## Current State

- **Phase:** apply
- **Hypothesis:** dependabot.yml docker /src → keys present; Docker Images does not start.
- **Expected signal:** grep keys; no 🐳 run on this SHA.
- **Tasks:** 1.1 pending, 2.1 pending
- **Retry count:** 0
- **Next action:** Write dependabot.yml.

## Events

- 2026-09-04: Proposed add-dependabot-docker.
