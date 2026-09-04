## Context

`docker inspect whisperdock-local:pinned` `Healthcheck=null`. `app.py` has only POST `/transcribe`. Gunicorn 17 MB POST already 413 JSON (`size_upload=2562484`, 10 ms) — no gunicorn body-size flag exists in 26.2.0 `--help`. Tiny WAV as `USER app` already 200 `{"transcription":[]}`.

## Goals / Non-Goals

**Goals:** liveness HTTP 200; Docker HEALTHCHECK exit 0 on that route.

**Non-Goals:** whisper readiness; curl apt package; gunicorn `--limit-request-*` body (none).

## Decisions

1. **GET `/health` → `jsonify(status="ok")`**
   - Why: `curl -f` / urllib require 2xx. Must not POST `/transcribe`.
2. **HEALTHCHECK exec-form Python urllib to `http://127.0.0.1:5000/health`**
   - Why: bookworm Python image has `python`, not necessarily curl. urllib raises on HTTP errors so exit is non-zero.
   - Timing: `--interval=30s --timeout=5s --start-period=15s --retries=3` (gunicorn answered POST in 2s locally).
3. **Not TCP-only**
   - Rejected: port open does not prove WSGI.

## Risks / Trade-offs

- [HEALTHCHECK as USER app] → python and 127.0.0.1:5000 are available; no extra files.
- [start-period too short] → 15s > observed 2s worker boot.
