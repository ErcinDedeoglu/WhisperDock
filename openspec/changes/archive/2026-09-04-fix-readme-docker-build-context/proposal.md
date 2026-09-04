## Why

README Getting Started tells users to `docker build -t whisperdock .` from the repo root. There is no root Dockerfile. The image is built from `src/` (`COPY whisper`, `COPY app.py`), matching GitHub Actions `context: ./src`.

## What Changes

- Replace the documented source-build command with `docker build -t whisperdock src`.
- Keep `docker run -p 5000:5000 whisperdock`.

Not **BREAKING** for the service API. Docs-only.

## Capabilities

### New Capabilities
- (none — `skip_specs: true`; README build command is not a service behavior contract)

### Modified Capabilities
- (none)

## Impact

- `README.md` Getting Started “Building from Source” only.
- No Dockerfile, app, or workflow edits.

## Problem

The published build command cannot work from the repository root.

## Non-goals

- Moving `src/Dockerfile` to the repo root.
- Editing GHA context.
- Removing unused `import socket` or pinning pip.

## Hypothesis

If README uses `docker build -t whisperdock src`, then the documented command matches GHA context and Dockerfile COPY paths.

## Expected signal

README contains `docker build -t whisperdock src` and does not contain `docker build -t whisperdock .`.

## Research

n/a — version-agnostic docs path (skip rule). GHA already documents the context: `.github/workflows/publish-docker.yml` `context: ./src`.

## Chosen and rejected approaches

- **Chosen:** `docker build -t whisperdock src` (default Dockerfile in that context).
- **Rejected:** `-f src/Dockerfile .` (COPY paths would still look at repo-root `whisper/` / `app.py` and fail).
- **Rejected:** move Dockerfile to root (would require GHA + COPY rewrites).

## Rollback

Restore the previous README command.

## Acceptance checks

- README has `docker build -t whisperdock src`
- README does not have `docker build -t whisperdock .`
- Chrome: N/A — no UI
