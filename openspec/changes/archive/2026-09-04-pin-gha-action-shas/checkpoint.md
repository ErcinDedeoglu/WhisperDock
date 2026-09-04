# Checkpoint

## Invariants

- **Goal:** All workflow `uses:` are 40-char SHAs with `# vN`.
- **Acceptance:** hex SHAs; unittest 8/8; Docker Images green; Chrome N/A.
- **Non-goals:** Changing majors.
- **Constraints:** Push `main` allowed; do not tag/push `dublok/*` locally.
- **Decisions:** SHAs from gh api commits/<tag>; comments for Dependabot.

## Current State

- **Phase:** apply
- **Hypothesis:** SHA pins → 40-hex uses; 🐳 succeeds.
- **Expected signal:** grep; GHA success.
- **Tasks:** 1.1 pending, 2.1 pending
- **Retry count:** 0
- **Next action:** Edit both workflow files.

## Events

- 2026-09-04: Proposed pin-gha-action-shas.
