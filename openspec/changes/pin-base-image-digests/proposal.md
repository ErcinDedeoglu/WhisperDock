## Why

`FROM python:3.12-bookworm`, `FROM python:3.12-slim-bookworm`, and `COPY --from=mwader/static-ffmpeg:9.0.1` are mutable tags. A publisher rebuild can change the image under the same tag with no audit trail.

## What Changes

- Pin all three refs as `name:tag@sha256:<index-digest>` using current OCI index digests.

## Capabilities

### New Capabilities

### Modified Capabilities

- `container-runtime`: Dockerfile FROM and COPY --from for python bookworm, slim-bookworm, and static-ffmpeg MUST include `@sha256:` index digests.

## Impact

- `src/Dockerfile` three image references.
- No API change. Rebuilds stay multi-arch.

## Problem

Floating tags can silently change bases and ffmpeg between CI runs.

## Non-goals

- Dependabot config. Pinning apt or pip (already pinned). Platform-specific (single-arch) digests.

## Hypothesis

If the three refs include the current index digests, then the Dockerfile contains those three `@sha256:` strings, the image still builds, unittest still exits 0.

## Expected signal

- Dockerfile contains `python:3.12-bookworm@sha256:581429e3df12d76e6af4be5ab7d0e7fc2013eb57dc23d2de691411c8efdbb970`
- Dockerfile contains `python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254`
- Dockerfile contains `mwader/static-ffmpeg:9.0.1@sha256:54e55b0cb8f672870fc38ceb2e6c411855cb3b39c505f5f3b2505ee01ed5f2b7`

## Research

- https://docs.docker.com/build/building/best-practices/#pin-base-image-versions — `FROM alpine:3.21@sha256:…`
- https://docs.docker.com/dhi/core-concepts/digests/ — pin the manifest-list/index digest for multi-platform.

## Chosen / rejected

- Chosen: keep tag + index digest (`name:tag@sha256:`).
- Rejected: digest-only (harder to read). Single-arch digests (breaks GHA amd64+arm64).

## Rollback

Remove the `@sha256:…` suffixes.

## Acceptance

- unittest 8/8 including holdout missing-file JSON 400.
- Chrome: N/A — no UI.
