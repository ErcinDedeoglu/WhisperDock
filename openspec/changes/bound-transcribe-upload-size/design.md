## Context

See proposal.md. Flask 3.1.2: `app.config['MAX_CONTENT_LENGTH']` is `None`. `/transcribe` reads `request.files` then writes two `NamedTemporaryFile(delete=False)` paths and runs ffmpeg. Existing JSON errors live in the view; there is no 413 handler. Tests use `app.test_client()` with `TESTING` left false, so error handlers run.

## Goals / Non-Goals

**Goals:**

- Abort oversized bodies at Flask request parsing with JSON 413.
- Prove with unittest that ffmpeg/`file.save` never run for that case (no new temps).

**Non-Goals:**

- WSGI/proxy limits, gunicorn flags, MIME allowlists, env-configurable cap.

## Decisions

1. **`app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000`**
   - Why: Flask 3.1 fileuploads example; config docs raise 413 `RequestEntityTooLarge`.
   - Rejected: 25 MiB (not documented); per-request `Request.max_content_length` (one route only, extra surface).

2. **`@app.errorhandler(413)` returning `jsonify(error=...), 413`**
   - Why: default 413 is HTML; transcribe-api requires JSON. Flask errorhandling JSON pattern. Status must be set on the return.
   - Rejected: `@app.errorhandler(HTTPException)` (too broad). Rejected: try/except in the view (abort can happen before the view).

3. **Test posts `io.BytesIO` of `limit + 1` bytes as `huge.wav`**
   - Why: test_client returns 413 without the werkzeug-dev-server connection reset.
   - Rejected: live gunicorn POST this slice (out of size S).

## Risks / Trade-offs

- [TESTING=True would skip handlers] → Do not enable `app.testing` in this slice.
- [16 MB SI vs 16 MiB] → Spec and config both use 16 000 000 bytes.
- [Clients sending >16 MB that used to transcribe] → Intentional; document in README only if a later polish finding asks.
- [Any git push publishes production] → Do not push.

## Migration Plan

- Local: set config + handler + test; run unittest.
- Rollback: revert those two files.
- Production image only after an authorized push.

## Open Questions

None.
