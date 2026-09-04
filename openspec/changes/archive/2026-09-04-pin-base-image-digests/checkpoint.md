# Checkpoint

## Invariants

- **Goal:** Pin python bookworm, slim-bookworm, and static-ffmpeg to OCI index digests.
- **Acceptance:** three `@sha256:` refs in Dockerfile; whisper-cli; unittest 8/8; Chrome N/A.
- **Non-goals:** Dependabot; single-arch digests.
- **Constraints:** Push `main` allowed; do not tag/push `dublok/*` locally; local tag `whisperdock-local:pinned`.
- **Decisions:** Keep tags; pin index digests from imagetools inspect.

## Current State

- **Phase:** apply
- **Hypothesis:** tag@sha256 pins → Dockerfile has three digests; image builds.
- **Expected signal:** grep three sha256 strings; cli-ok.
- **Tasks:** 1.1 pending, 2.1 pending
- **Retry count:** 0
- **Next action:** Edit Dockerfile refs and rebuild.

## Events

- 2026-09-04: Proposed pin-base-image-digests.
