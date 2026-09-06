## Context

GHA `context: ./src`. Docker reads `.dockerignore` only at that root. See proposal.md.

## Goals / Non-Goals

**Goals:**
- Exclude host unittest and pycache from the build context.

**Non-Goals:**
- Trimming whisper examples/tests.

## Decisions

1. **`src/.dockerignore` with `test_app.py`, `__pycache__`, `*.pyc`**
   - Safe: Dockerfile never COPY those paths.
   - Probe: stdin Dockerfile `COPY test_app.py` must fail.

## Risks / Trade-offs

- [GHA republishes on src/**] → accepted; ignore file is in the context path.

## Migration Plan

Add the file. Rebuild `whisperdock-local:pinned`. Rollback: delete it.

## Open Questions

None.
