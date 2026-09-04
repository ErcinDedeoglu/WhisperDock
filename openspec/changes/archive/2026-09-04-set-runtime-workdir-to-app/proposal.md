## Why

The last Dockerfile `WORKDIR` is `/app/whisper` (build cwd). Runtime `Config.WorkingDir` is `/app/whisper`. Gunicorn `-w 4` is **not** this change: idle 96 MiB, 4 concurrent transcribes 103 MiB.

## What Changes

- `WORKDIR /app` after `COPY app.py` so the service cwd is the app root.

Not **BREAKING** for `/transcribe` (paths are absolute; gunicorn already `--chdir /app`).

## Capabilities

### New Capabilities

- (none)

### Modified Capabilities

- `container-runtime`: the image working directory MUST be `/app`.

## Impact

- `src/Dockerfile` one `WORKDIR /app`.

## Problem

Containers start in the whisper.cpp tree, not the Flask app directory.

## Non-goals

- gunicorn `-w 4`; `--chdir` removal; chown `/app`.

## Hypothesis

If Dockerfile ends with `WORKDIR /app` after copying `app.py`, then `docker inspect` WorkingDir is `/app`, unittest still exits 0, and `/health` still 200.

## Expected signal

- Image `Config.WorkingDir` is `/app`.
- Holdout missing-file 400 JSON.

## Research

Official pattern: https://docs.docker.com/reference/dockerfile/#workdir (last WORKDIR is the container default cwd)
Why current code is worse: last WORKDIR is `/app/whisper`; inspect WorkingDir=/app/whisper
Chosen approach: `WORKDIR /app` after `COPY app.py`
Rejected alternative: keep `--chdir` as the only cwd (WORKDIR still wrong for exec/health)
Proof plan: inspect WorkingDir=/app; unittest exit 0; Chrome: N/A — no UI

## Chosen and rejected approaches

- **Chosen:** restore `WORKDIR /app` before USER/CMD.
- **Rejected:** `docker run -w /app` (does not fix the image).
- **Rejected:** drop gunicorn `--chdir` this slice.

## Rollback

Remove the extra `WORKDIR /app`.

## Acceptance checks

- unittest including holdout missing-file 400 JSON
- `docker inspect` WorkingDir is `/app`
- Chrome: N/A — no UI
