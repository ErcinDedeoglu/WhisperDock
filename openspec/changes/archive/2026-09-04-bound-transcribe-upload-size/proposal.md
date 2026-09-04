## Why

POST `/transcribe` has no Flask `MAX_CONTENT_LENGTH` (default `None`), so a client can stream an unbounded body into `file.save` and ffmpeg. That is a resource-use gap on Flask 3.1.x (installed 3.1.2). Oversized uploads must fail as JSON 413 before conversion, matching the existing JSON error contract.

## What Changes

- Set `app.config['MAX_CONTENT_LENGTH']` to `16 * 1000 * 1000` (16 MB, Flask 3.1 file-upload example).
- Register a 413 handler that returns `application/json` with an `error` field (not HTML).
- Add a unittest that POSTs a body larger than the limit and asserts 413 JSON and `leaked_tmp=[]`.

Not **BREAKING** for in-limit clients. Bodies over 16 MB that previously reached ffmpeg will now get 413.

## Capabilities

### New Capabilities
- (none)

### Modified Capabilities
- `transcribe-api`: oversized uploads are a JSON 413 client error and must not create temp files or start ffmpeg.

## Impact

- `src/app.py` config + error handler; `src/test_app.py` one test.
- No Dockerfile, gunicorn flags, whisper.cpp, or publish-workflow changes.
- No origin push.

## Problem

Unbounded request bodies can exhaust disk/CPU via `NamedTemporaryFile` + ffmpeg.

## Non-goals

- Gunicorn/`--limit-request-line` or proxy 413.
- Auth, MIME allowlists, or changing the 16 MB figure after research.
- Pushing a branch (`publish-docker.yml` on `**`).

## Hypothesis

If `MAX_CONTENT_LENGTH` is `16 * 1000 * 1000` and `@app.errorhandler(413)` returns `jsonify(error=...)`, then a test-client POST whose body exceeds that limit is HTTP 413 `application/json` with `error` set, `leaked_tmp=[]`, and existing tests stay green.

## Expected signal

Oversized POST: status 413, content-type JSON, no new `tmp*` files. Holdout: missing-file still 400 `{"error": "No file part"}`.

## Research

Official pattern: https://flask.palletsprojects.com/en/stable/patterns/fileuploads/ (Flask 3.1.x)
Why current code is worse: `MAX_CONTENT_LENGTH` is `None`; `file.save` then ffmpeg run with no size gate
Chosen approach: config `16 * 1000 * 1000` plus `@app.errorhandler(413)` + `jsonify`
Rejected alternative: view-level `request.content_length` check (Flask already aborts before the view)
Proof plan: `python3 -m unittest test_app.py` exit 0 from `src/`; Chrome: N/A — no UI

Supporting official sources:
- https://flask.palletsprojects.com/en/stable/config/#MAX_CONTENT_LENGTH
- https://flask.palletsprojects.com/en/stable/web-security/#resource-use
- https://flask.palletsprojects.com/en/stable/errorhandling/#returning-api-errors-as-json

## Chosen and rejected approaches

- **Chosen:** Flask `MAX_CONTENT_LENGTH` + JSON 413 handler. Smallest reversible surface; official 3.1 pattern.
- **Rejected:** catch `RequestEntityTooLarge` only inside `transcribe_audio` (misses abort before the view).
- **Rejected:** 25 MiB custom cap (not the documented 16 MB example).

## Rollback

Revert `src/app.py` and `src/test_app.py`. No migrations, no image publish.

## Acceptance checks

- `cd src && python3 -m unittest test_app.py -v` exits 0
- Oversized POST test asserts 413 JSON `error` and empty temp-dir delta
- Missing-file holdout still 400 JSON `"No file part"`
- Chrome: N/A — no UI
