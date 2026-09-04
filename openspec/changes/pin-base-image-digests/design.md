## Context

Three floating tags. GHA builds linux/amd64 and linux/arm64. Digests must be OCI index digests from `docker buildx imagetools inspect`. See proposal.md.

## Goals / Non-Goals

**Goals:**
- Immutable base and ffmpeg refs that still resolve multi-arch.

**Non-Goals:**
- Dependabot. Single-arch pins.

## Decisions

1. **Tag plus index digest**
   - `python:3.12-bookworm@sha256:581429e3df12d76e6af4be5ab7d0e7fc2013eb57dc23d2de691411c8efdbb970`
   - `python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254`
   - `mwader/static-ffmpeg:9.0.1@sha256:54e55b0cb8f672870fc38ceb2e6c411855cb3b39c505f5f3b2505ee01ed5f2b7`
   - Index mediaType `application/vnd.oci.image.index.v1+json`; includes linux/amd64 and linux/arm64.

## Risks / Trade-offs

- [No automatic security rebuilds of bases] → explicit digest updates; tags remain for humans.

## Migration Plan

Edit three refs. Rebuild `whisperdock-local:pinned`. Rollback: drop `@sha256:` suffixes.

## Open Questions

None.
