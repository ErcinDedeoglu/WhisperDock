## Why

The image has no `HEALTHCHECK` (`Config.Healthcheck` is null). Docker cannot tell a stuck gunicorn from a live one. The only HTTP route is POST `/transcribe`; a probe that POSTs it is the wrong signal (400, and `curl -f` would fail). Official HEALTHCHECK is a command that exits 0 when the process can serve HTTP.

Gunicorn request-size vs Flask 16 MB was measured and is **not** this change: 17 MB POST already returns JSON 413 in ~10 ms under gunicorn.

## What Changes

- Add GET `/health` that returns HTTP 200 JSON without running ffmpeg or whisper.
- Add Dockerfile `HEALTHCHECK` that GETs that route with stdlib Python (no curl package).

Not **BREAKING** for `/transcribe`.

## Capabilities

### New Capabilities

- (none)

### Modified Capabilities

- `transcribe-api`: GET `/health` MUST return 200 JSON.
- `container-runtime`: the service image MUST declare a HEALTHCHECK against `/health`.

## Impact

- `src/app.py`, `src/test_app.py`, `src/Dockerfile`.

## Problem

Orchestrators see the container as running even if gunicorn is wedged; no liveness HTTP route exists.

## Non-goals

- Readiness that runs whisper/ffmpeg.
- Installing curl/wget.
- gunicorn body-size flags (no such setting; Flask 16 MB already 413).

## Hypothesis

If GET `/health` returns 200 JSON and Dockerfile HEALTHCHECK probes it, then unittest covers `/health`, rebuilt image Healthcheck is non-null, and a started container becomes `healthy`.

## Expected signal

- GET `/health` → 200 `application/json` with `status`.
- `docker inspect` Healthcheck Test is present.
- Container `State.Health.Status` becomes `healthy`.

## Research

Official pattern: https://docs.docker.com/reference/dockerfile/#healthcheck (CMD exit 0/1; `--interval` `--timeout` `--start-period` `--retries`)
Why current code is worse: Healthcheck=null; only POST `/transcribe`
Chosen approach: GET `/health` + `HEALTHCHECK` exec-form `python -c urllib`
Rejected alternative: probe POST `/transcribe` (400 is not healthy; would need curl `-f` exception)
Proof plan: unittest GET `/health`; inspect Healthcheck; `docker ps` healthy; Chrome: N/A — no UI

Supporting: https://werkzeug.palletsprojects.com/en/stable/request_data/ (gunicorn has no body-size setting; Flask 16 MB already applied)

## Chosen and rejected approaches

- **Chosen:** liveness GET `/health` + Python urllib HEALTHCHECK.
- **Rejected:** install curl just for HEALTHCHECK.
- **Rejected:** TCP `nc -z` (does not prove WSGI answers).

## Rollback

Remove `/health` and HEALTHCHECK.

## Acceptance checks

- unittest including `/health` 200 JSON
- image Healthcheck non-null
- running container reaches healthy
- holdout missing-file 400 JSON
- Chrome: N/A — no UI
