# Checkpoint

## Invariants

- **Goal:** Runtime ffmpeg without Debian libllvm15/Mesa; whisper-cli still works.
- **Acceptance:** libllvm15 absent; ffmpeg on PATH; size < 1.06 GB; unittest 8/8; Chrome N/A.
- **Non-goals:** Builder ffmpeg, ffprobe, compiling ffmpeg, johnvansickle.
- **Constraints:** Push `main` allowed; do not tag/push `dublok/*` locally; local tag `whisperdock-local:pinned`.
- **Decisions:** `mwader/static-ffmpeg:9.0.1` COPY `/ffmpeg`; runtime apt only libgomp1.

## Current State

- **Phase:** apply
- **Hypothesis:** static ffmpeg + libgomp1 only → no libllvm15; ffmpeg works; size < 1.06 GB.
- **Expected signal:** libllvm15 query exit 1; ffmpeg on PATH; smaller image.
- **Tasks:** 1.1 pending, 2.1 pending
- **Retry count:** 0
- **Next action:** Apply Dockerfile runtime change and rebuild.

## Events

- 2026-09-04: Proposed static-ffmpeg-drop-mesa.
