# Checkpoint

## Invariants

- Goal: automatic Docker Hub publish only on `main`; keep `workflow_dispatch`.
- Acceptance: YAML parse `branches == ['main']`, no `'**'`, dispatch present; sync-whisper still dispatches; Chrome N/A.
- Non-goals: tag scheme, secrets, sync-whisper edits, probe push.
- Constraints: do not push until this change is archived on `main`. Size S.
- Decision: `on.push.branches: [main]` + keep dispatch; no job-level main `if`.

## Current State

- Phase: sync complete; next archive
- Hypothesis: main-only push filter stops feature-branch publishes; dispatch remains
- Expected signal: parse assert exit 0
- Tasks: 1.1–2.1 done
- Retry count: 0
- Confidence: high local YAML; remote GHA unproven until first allowed push
- Verification: local pass; Chrome N/A; CI n/a no push
- Next action: OpenSpec archive restrict-docker-publish-to-main

## Facts / Assumptions / Open questions

- Facts: current `branches: ['**']`; sync-whisper uses `gh workflow run`
- Assumptions: restricting push to main is enough because dispatch covers tag branches
- Open questions: none

## Events

- 2026-09-04 propose: artifacts created; product/workflow code unchanged
- 2026-09-04 apply 1.1: branches=['main']; parse OK
- 2026-09-04 apply 2.1: sync-whisper still dispatches; build-push push: true
- 2026-09-04 verify: YAML_OK; unittest 6/6; validate --strict ok
- 2026-09-04 sync: created openspec/specs/docker-image-publish/spec.md
