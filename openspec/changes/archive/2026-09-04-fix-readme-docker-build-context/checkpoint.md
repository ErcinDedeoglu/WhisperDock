# Checkpoint

## Invariants

- Goal: README source-build command uses `src` context.
- Acceptance: README has `docker build -t whisperdock src`; no `docker build -t whisperdock .`; Chrome N/A.
- Non-goals: move Dockerfile, GHA edits, socket import, pip pins.
- Constraints: docs-only. Size S. `skip_specs: true`.
- Decision: `docker build -t whisperdock src` not `-f src/Dockerfile .`.

## Current State

- Phase: verify green; next archive (sync n/a)
- Hypothesis: `src` context matches GHA COPY layout
- Expected signal: README grep as above
- Tasks: 1.1 done
- Retry count: 0
- Confidence: high
- Verification: local pass; Chrome N/A; sync n/a
- Next action: OpenSpec archive fix-readme-docker-build-context
- Design: skipped (docs-only; instruction “create only if” none apply)

## Events

- 2026-09-04 propose: artifacts; design skipped; specs skipped
- 2026-09-04 apply 1.1: README docker build uses src context
- 2026-09-04 verify: README_OK; unittest 6/6; validate --strict ok
