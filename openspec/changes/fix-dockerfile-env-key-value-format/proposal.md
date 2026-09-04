## Why

GHA run 33848114073 annotated `src/Dockerfile` lines 5–6: LegacyKeyValueFormat (`ENV key value`). Docker documents that space-separated ENV is deprecated; the equals form is required.

## What Changes

- Change `ENV PYTHONDONTWRITEBYTECODE 1` and `ENV PYTHONUNBUFFERED 1` to `ENV PYTHONDONTWRITEBYTECODE=1` and `ENV PYTHONUNBUFFERED=1`.
- Require the service Dockerfile ENV instructions to use `key=value`.

Not **BREAKING**. Same env names and values.

## Capabilities

### New Capabilities

- (none)

### Modified Capabilities

- `container-runtime`: Dockerfile ENV instructions MUST use `key=value`, not the legacy space form.

## Impact

- `src/Dockerfile` two ENV lines.
- No app, unittest, or workflow trigger changes.

## Problem

Legacy `ENV key value` emits BuildKit warnings and is deprecated.

## Non-goals

- Combining both ENV into one instruction.
- Unused `import socket`.
- Apt/pip pinning, `# check=skip`, or other Dockerfile lints.

## Hypothesis

If both ENV lines use `=`, then `docker build --check -f src/Dockerfile src` reports 0 LegacyKeyValueFormat warnings, unittest still exits 0, and a rebuilt image still has those two env vars set to `1`.

## Expected signal

- Dockerfile has `ENV PYTHONDONTWRITEBYTECODE=1` and `ENV PYTHONUNBUFFERED=1` and no `ENV … 1` without `=`.
- `docker build --check` has no LegacyKeyValueFormat.
- GHA publish-docker has no LegacyKeyValueFormat annotation on those lines.
- Holdout: missing-file POST still 400 JSON.

## Research

Official pattern: https://docs.docker.com/reference/build-checks/legacy-key-value-format/ (`ENV key=value`; space form deprecated)
Why current code is worse: `ENV PYTHONDONTWRITEBYTECODE 1` / `ENV PYTHONUNBUFFERED 1`; GHA + `docker build --check` warn twice
Chosen approach: add `=` on those two lines
Rejected alternative: `# check=skip=LegacyKeyValueFormat` (hides the lint; syntax stays deprecated)
Proof plan: `docker build --check` 0 warnings; unittest exit 0; Chrome: N/A — no UI

Supporting official sources:
- https://docs.docker.com/reference/dockerfile/#env (`ENV MY_CAT=fluffy`; space form is alternative/legacy)
- https://docs.docker.com/go/dockerfile/rule/legacy-key-value-format/

## Chosen and rejected approaches

- **Chosen:** Two-line `key=value` conversion. Smallest reversible surface.
- **Rejected:** Single `ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1` (unrelated merge).
- **Rejected:** Skip the build check.

## Rollback

Restore the two ENV lines without `=`.

## Acceptance checks

- `cd src && python3 -m unittest test_app.py -v` exits 0
- `docker build --check -f src/Dockerfile src` has no LegacyKeyValueFormat
- Dockerfile ENV lines use `=`
- Next GHA publish-docker run succeeds without LegacyKeyValueFormat on Dockerfile:5/6
- Chrome: N/A — no UI
