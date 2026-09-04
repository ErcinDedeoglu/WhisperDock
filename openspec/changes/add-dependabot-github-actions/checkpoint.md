# Checkpoint

## Invariants

- **Goal:** Dependabot github-actions at `/` weekly; docker `/src` kept.
- **Acceptance:** YAML has both ecosystems; unittest 8/8; no Docker Images; Chrome N/A.
- **Non-goals:** SHA pins, pip.
- **Constraints:** Push `main` allowed; do not tag/push `dublok/*` locally.
- **Decisions:** directory `/` for actions.

## Current State

- **Phase:** apply
- **Hypothesis:** second updates entry → keys present; Docker Images does not start.
- **Expected signal:** grep; no 🐳 run.
- **Tasks:** 1.1 pending, 2.1 pending
- **Retry count:** 0
- **Next action:** Edit dependabot.yml.

## Events

- 2026-09-04: Proposed add-dependabot-github-actions.
