## Context

Builder apt currently installs five pinned packages. Runtime ffmpeg is static 9.0.1. whisper.cpp examples ffmpeg option is OFF. See proposal.md.

## Goals / Non-Goals

**Goals:**
- Builder apt is only packages required to compile whisper-cli.

**Non-Goals:**
- Removing git. Changing runtime.

## Decisions

1. **Drop `ffmpeg` and `libsndfile1` from builder**
   - Not linked into whisper-cli (`WHISPER_COMMON_FFMPEG` OFF).
   - Keep git: ggml CMake records commit via `find_program(GIT_EXE)`.

## Risks / Trade-offs

- [Builder cache bust / longer CI this once] → expected; runtime layers stay cached.

## Migration Plan

Edit one RUN. Rebuild `whisperdock-local:pinned`. Rollback: restore the two pins.

## Open Questions

None.
