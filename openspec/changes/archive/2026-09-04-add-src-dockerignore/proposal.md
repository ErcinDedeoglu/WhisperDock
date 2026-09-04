## Why

`docker build` context is `src/`. There is no `src/.dockerignore`. Nested `src/whisper/.dockerignore` is not applied. The daemon still receives `test_app.py` and `__pycache__`, which the image never copies.

## What Changes

- Add `src/.dockerignore` excluding `test_app.py`, `__pycache__`, and `*.pyc`.

## Capabilities

### New Capabilities

### Modified Capabilities

skip_specs: no runtime behavior change.

## Impact

- `src/.dockerignore` only. GHA still rebuilds because the path is under `src/**`.

## Problem

Build context root has no ignore file.

## Non-goals

- Ignoring whisper tests/examples (cmake `add_subdirectory`). Nested whisper/.dockerignore.

## Hypothesis

If `src/.dockerignore` lists `test_app.py`, then `COPY test_app.py` in a probe Dockerfile fails, the service image still builds, unittest still exits 0.

## Expected signal

- Probe `COPY test_app.py` fails.
- `test -x` whisper-cli succeeds.

## Research

- https://docs.docker.com/build/concepts/context/#dockerignore-files — ignore file is at context root.
- https://docs.docker.com/build/building/best-practices/#exclude-with-dockerignore

## Chosen / rejected

- Chosen: ignore unittest and pycache only.
- Rejected: ignoring whisper/tests or examples (cmake requires them).

## Rollback

Delete `src/.dockerignore`.

## Acceptance

- unittest 8/8 including holdout missing-file JSON 400.
- Chrome: N/A — no UI.
