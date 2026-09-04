# Checkpoint

## Invariants

- **Goal:** Dependabot pip watches `/src` weekly.
- **Acceptance:** YAML has pip `/src`; unittest 8/8; no Docker Images; Chrome N/A.
- **Non-goals:** Changing package versions; SHA-pinning actions.
- **Constraints:** Push `main` allowed; do not tag/push `dublok/*` locally.
- **Decisions:** pip `/src` weekly.

## Current State

- **Phase:** apply
- **Hypothesis:** third updates entry → keys present; Docker Images does not start.
- **Expected signal:** grep; Dependabot pip scan; no 🐳.
- **Tasks:** 1.1 pending, 2.1 pending
- **Retry count:** 0
- **Next action:** Edit dependabot.yml.

## Events

- 2026-09-04: Proposed add-dependabot-pip.
