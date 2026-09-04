## Why

The runtime stage still uses `python:3.12-bookworm` (`buildpack-deps`), which ships `g++` 4:12.2.0-3. Multi-stage already dropped cmake; the compiler remains because it is in the fat base, not because we install it.

## What Changes

- Runtime `FROM` becomes `python:3.12-slim-bookworm` (Debian 12, not floating `python:3.12-slim` which tracks Trixie).
- Builder stays `python:3.12-bookworm`.
- Runtime still apt-installs pinned ffmpeg, libgomp1, libsndfile1.

## Capabilities

### New Capabilities

### Modified Capabilities

- `container-runtime`: published runtime MUST NOT install `g++`; MUST keep whisper-cli and CPython 3.12.

## Impact

- `src/Dockerfile` runtime FROM line only.
- No API change. Rebuild publishes a smaller image without a C++ compiler.

## Problem

Hub `0b47593` and local `whisperdock-local:pinned` report `g++` installed.

## Non-goals

- Slim builder, Alpine, Trixie, digest pins.
- Removing ffmpeg from runtime.

## Hypothesis

If runtime is `python:3.12-slim-bookworm`, then `dpkg-query -W g++` exits non-zero, whisper-cli still runs, image size is below 2.21 GB, unittest still exits 0.

## Expected signal

- `dpkg-query -W g++` non-zero.
- `test -x /app/whisper/build/bin/whisper-cli` exits 0.
- `python -c "import sys; print(sys.version_info[:2])"` prints `(3, 12)`.
- Image smaller than 2.21 GB.

## Research

- https://hub.docker.com/_/python — `3.12-slim-bookworm` exists; `3.12-slim` is Trixie.
- https://github.com/docker-library/python/blob/master/3.12/slim-bookworm/Dockerfile — `debian:bookworm-slim`; gcc/g++ used then purged.
- https://github.com/docker-library/python/issues/882 — slim has no gcc on PATH.

## Chosen / rejected

- Chosen: `python:3.12-slim-bookworm` runtime only.
- Rejected: `python:3.12-slim` (Trixie; apt pins break). Alpine (musl vs glibc whisper .so). Slim builder (needs compilers).

## Rollback

Revert the runtime FROM line to `python:3.12-bookworm`.

## Acceptance

- unittest 8/8 including holdout missing-file JSON 400.
- Chrome: N/A — no UI.
