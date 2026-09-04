## Why

POST `/transcribe` with invalid media raises uncaught `subprocess.CalledProcessError` from `ffmpeg ... check=True`. Flask then returns HTML `500 Internal Server Error` (`text/html`) instead of the documented JSON error body, and both `NamedTemporaryFile(delete=False)` paths leak. Observed on the Flask test client: garbage `bad.wav` → status 500, `text/html`, two leftover `tmp*` files.

## What Changes

- Catch ffmpeg conversion failure (`CalledProcessError`) and missing-binary (`FileNotFoundError`) inside `transcribe_audio` and return JSON `{"error": ...}` with an HTTP 4xx/5xx status, not HTML.
- Always unlink the upload and converted temp paths after the request (success, ffmpeg failure, whisper failure, or missing binary).
- Add in-repo tests that drive the shipped Flask `/transcribe` view and `parse_transcription` (stdlib `unittest`, no new dependencies).

Not **BREAKING**: documented success JSON and the existing `"error": "Error in transcription"` shape stay. Clients that parsed HTML 500 on bad media will now receive JSON.

## Capabilities

### New Capabilities
- `transcribe-api`: HTTP `/transcribe` request validation, media conversion errors, transcription errors, JSON response bodies, and temp-file cleanup for this Flask service.

### Modified Capabilities
- (none — `openspec/specs/` is empty)

## Impact

- `src/app.py` `transcribe_audio` only (plus a new stdlib test module next to it).
- No Dockerfile, GitHub Actions, whisper.cpp, dependency, or public Docker Hub changes.
- Docker image behavior changes only after a later authorized rebuild; local Flask test-client is the verify path.

## Problem

Invalid uploads and subprocess launch failures are unhandled exceptions, so the API contract (JSON errors) is violated and temp files accumulate.

## Non-goals

- Changing whisper-cli flags, model path, or `parse_transcription` regex.
- Adding upload size limits, auth, or content-type allowlists.
- Upgrading `python:3.8`, pinning Flask, or editing `publish-docker.yml`.
- Pushing a branch (that workflow publishes `dublok/whisperdock` on every push).

## Hypothesis

If `transcribe_audio` catches ffmpeg `CalledProcessError` / `FileNotFoundError` and unlinks temps in `finally`, then a Flask test-client POST of bytes `not-audio` as `bad.wav` returns `application/json` with an `error` field (not `text/html`) and the temp-dir delta of new `tmp*` names is empty.

## Expected signal

`GARBAGE_AUDIO status=400` (or 500 only if classified as server error) `type=application/json` body contains `"error"`; `leaked_tmp=[]`. Holdout: missing-file still `400` JSON `"No file part"`; `parse_transcription` still returns two segments for the README-style timestamp dump.

## Research

Official pattern: https://flask.palletsprojects.com/en/stable/errorhandling/ (Flask 3.1.x — installed local Flask 3.1.2; Docker `pip install Flask` unpinned on python:3.8)
Why current code is worse: unhandled exceptions become generic HTML 500; `delete=False` temps are never unlinked on that path
Chosen approach: handle the known subprocess failures in the view with `jsonify(error=...)` (same pattern as missing-file) and `os.unlink` in `finally`
Rejected alternative: `@app.errorhandler(Exception)` JSON wrapper — Flask 3.1 docs warn this is `except Exception` and captures all HTTP codes
Proof plan: `python3 -m unittest src.test_app -v` exit 0; Flask test-client POST garbage WAV asserts JSON `error` and zero leaked temps; Chrome: N/A — no UI

Supporting official sources:
- https://flask.palletsprojects.com/en/stable/errorhandling/#returning-api-errors-as-json (`jsonify(error=...), status`)
- https://docs.python.org/3.8/library/tempfile.html (`NamedTemporaryFile(delete=False)` then `os.unlink`)
- https://docs.python.org/3.8/library/subprocess.html (`check=True` raises `CalledProcessError`)

## Chosen and rejected approaches

- **Chosen:** view-local try/except/finally around ffmpeg + whisper-cli + temp unlink. Smallest reversible surface; matches existing `jsonify(error=...)` returns.
- **Rejected:** global Flask errorhandler for `Exception` or 500 (too broad; does not unlink temps created before the raise).
- **Rejected:** drop `check=True` and send a failed/empty wav to whisper-cli (hides client errors as transcription 500).

## Rollback

Revert the `src/app.py` and test-file diff. No migrations, no image publish, no schema.

## Acceptance checks

- `python3 -m unittest discover -s src -p 'test_*.py' -v` exits 0
- That run's output includes `GARBAGE_AUDIO` JSON error (or equivalent test name asserting `application/json` + `error` key) and no leaked `tmp*` files
- Missing-file holdout still 400 JSON `"No file part"`
- Chrome: N/A — no UI
