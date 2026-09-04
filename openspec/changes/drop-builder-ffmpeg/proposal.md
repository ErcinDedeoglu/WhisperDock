## Why

The builder stage still apt-installs Debian `ffmpeg` and `libsndfile1`. whisper.cpp compiles with `WHISPER_COMMON_FFMPEG` off; runtime already ships static ffmpeg 9.0.1. Those packages only inflate the builder layer and CI time.

## What Changes

- Builder apt drops `ffmpeg` and `libsndfile1`.
- Builder still pins `build-essential`, `cmake`, and `git`.

## Capabilities

### New Capabilities

### Modified Capabilities

- `container-runtime`: builder apt pins are `build-essential`, `cmake`, and `git` only; Debian ffmpeg is not a builder install.

## Impact

- `src/Dockerfile` builder `RUN apt-get` line.
- No API or runtime image change.

## Problem

Builder installs `ffmpeg=7:5.1.9-0+deb12u1` and `libsndfile1=1.2.0-1+deb12u1` though compile does not need them.

## Non-goals

- Dropping git. Changing runtime. Slim builder.

## Hypothesis

If builder apt is only the three compile packages, then `make` still produces whisper-cli, the Dockerfile builder RUN has no `ffmpeg=`, unittest still exits 0.

## Expected signal

- Builder apt line contains `build-essential=12.9`, `cmake=3.25.1-1`, `git=1:2.39.5-0+deb12u3`.
- Builder apt line does not contain `ffmpeg=`.
- `test -x /app/whisper/build/bin/whisper-cli` in the rebuilt image.

## Research

- n/a — version-agnostic unused-package removal. Docker: https://docs.docker.com/build/building/best-practices/#dont-install-unnecessary-packages
- whisper.cpp `WHISPER_COMMON_FFMPEG` defaults OFF in `src/whisper/CMakeLists.txt`.

## Chosen / rejected

- Chosen: drop ffmpeg and libsndfile1 from builder only.
- Rejected: dropping git (cmake records git hash). Slim builder.

## Rollback

Restore the two package pins on the builder apt line.

## Acceptance

- unittest 8/8 including holdout missing-file JSON 400.
- Chrome: N/A — no UI.
