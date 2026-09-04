# Checkpoint

## Invariants

- **Goal:** Runtime image without `g++`; whisper-cli and CPython 3.12 remain.
- **Acceptance:** `dpkg-query -W g++` non-zero; whisper-cli executable; python `(3, 12)`; size < 2.21 GB; unittest 8/8; Chrome N/A.
- **Non-goals:** Slim builder, Alpine, Trixie, digest pins.
- **Constraints:** Push `main` allowed; do not tag/push `dublok/*` locally; local tag `whisperdock-local:pinned`.
- **Decisions:** Runtime `python:3.12-slim-bookworm`; builder stays bookworm.

## Current State

- **Phase:** apply
- **Hypothesis:** slim-bookworm runtime → no g++; cli works; size < 2.21 GB; unittest 0.
- **Expected signal:** g++ query exit 1; cli-ok; `(3, 12)`; smaller image.
- **Tasks:** 1.1 pending, 2.1 pending
- **Retry count:** 0
- **Next action:** Apply 1.1 Dockerfile FROM change and rebuild.

## Events

- 2026-09-04: Proposed slim-runtime-drop-compiler. Research: Hub `3.12-slim` is Trixie; use slim-bookworm.
