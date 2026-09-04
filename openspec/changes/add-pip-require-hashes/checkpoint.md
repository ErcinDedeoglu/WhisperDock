# Checkpoint

## Invariants

- **Goal:** Hash-check frozen pip packages on amd64 and arm64.
- **Acceptance:** hashes in requirements; `--require-hashes` in Dockerfile; local rebuild + unittest; GHA success; Chrome N/A.
- **Non-goals:** pip-tools; hashing all MarkupSafe wheels.
- **Constraints:** `main`; push allowed; do not tag `dublok/*` locally.
- **Decisions:** `--only-binary :all:`; two MarkupSafe linux cp312 hashes.

## Current State

- **Phase:** propose complete; next apply
- **Hypothesis:** multi-hash + --require-hashes → local and GHA installs succeed; versions unchanged
- **Expected signal:** rebuild OK; metadata match
- **Rollback:** unhashed freeze; drop --require-hashes
- **Tasks:** all pending
- **Retry count:** 0
- **Confidence:** medium until rebuild
- **Next action:** apply 1.1

## Facts

- Freeze already lists eight packages
- MarkupSafe is the only arch-specific wheel

## Assumptions

- GHA manylinux2014_x86_64 and local manylinux2014_aarch64 remain the selected wheels

## Open questions

- None

## Events

- 2026-09-04 selected hash-checking over gunicorn size (security; named next experiment)
