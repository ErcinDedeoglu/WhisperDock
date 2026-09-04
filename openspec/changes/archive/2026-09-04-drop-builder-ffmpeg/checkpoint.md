# Checkpoint

## Invariants

- **Goal:** Builder apt omits unused Debian ffmpeg/libsndfile1; whisper-cli still builds.
- **Acceptance:** no `ffmpeg=` in builder apt; whisper-cli executable; unittest 8/8; Chrome N/A.
- **Non-goals:** Dropping git; runtime changes.
- **Constraints:** Push `main` allowed; do not tag/push `dublok/*` locally; local tag `whisperdock-local:pinned`.
- **Decisions:** Keep git; drop ffmpeg and libsndfile1 from builder.

## Current State

- **Phase:** apply
- **Hypothesis:** three-package builder apt → whisper-cli still builds; no builder ffmpeg=.
- **Expected signal:** Dockerfile grep; cli-ok after rebuild.
- **Tasks:** 1.1 pending, 2.1 pending
- **Retry count:** 0
- **Next action:** Edit builder apt RUN and rebuild.

## Events

- 2026-09-04: Proposed drop-builder-ffmpeg.
