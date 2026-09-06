# Checkpoint

## Invariants

- Goal: skip Docker Images on docs-only main pushes.
- Non-goals: gunicorn -w 4; tag scheme; sync-whisper.
- Decision: paths allowlist src/** + this workflow.

## Current State

- Phase: propose
- Hypothesis: paths filter → parse OK; apply SHA runs GHA; archive SHA does not.
- Next: validate, apply 1.1.

## Events

- 2026-09-04: evidence GHA 33853268414 docs-only rebuild 2m46s
