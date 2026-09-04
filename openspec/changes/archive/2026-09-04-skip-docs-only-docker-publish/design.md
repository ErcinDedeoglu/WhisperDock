## Context

`on.push.branches: [main]` with no `paths`. Docker build context is `src/`. Docs archive commits still published (GHA 33853268414). `workflow_dispatch` is how `sync-whisper.yml` rebuilds.

## Goals / Non-Goals

**Goals:** skip automatic publish when no image-source file changed.

**Non-Goals:** tag scheme; required-check skip jobs; gunicorn workers.

## Decisions

1. **`paths` allowlist, not `paths-ignore`**
   - Why: official docs forbid both on one event. Allowlist matches the Docker context plus the workflow itself (so this file's edits still run).
2. **Leave `workflow_dispatch` unfiltered**
   - Why: GitHub does not apply path filters to dispatch.

## Risks / Trade-offs

- [README-only docker instruction change] → no rebuild; image bytes unchanged; accept.
- [First apply commit] → includes the workflow file so it still runs; skip is proven on the later docs-only archive push.
