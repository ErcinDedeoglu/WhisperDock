## Context

See proposal.md for motivation. `src/app.py` already returns `jsonify(error=...)` for missing uploads, then writes two `NamedTemporaryFile(delete=False)` paths, runs ffmpeg with `check=True`, and only unlinks after ffmpeg returns. Flask 3.1 converts the uncaught `CalledProcessError` into HTML 500. Image runtime is Python 3.8; local verify is Flask 3.1.2 on Python 3.12 without `/app/whisper/build/bin/whisper-cli`.

## Goals / Non-Goals

**Goals:**
- Map known conversion and launch failures to the existing JSON error envelope
- Unlink both temp paths on every exit from the handler
- Prove that with stdlib tests that POST the real Flask view

**Non-Goals:**
- App-wide JSON error middleware
- Changing ffmpeg flags, whisper-cli flags, or parse regex
- New packages, Docker rebuild as a required gate, or CI publish

## Decisions

1. **Handle failures in the view, not `@app.errorhandler(Exception)`.** Flask 3.1 documents returning API errors as `jsonify(...)` and warns that a generic `Exception` handler is `except Exception`. The missing-file path already returns JSON from the view. Alternative rejected: global 500 JSON handler (too broad, does not unlink temps).

2. **HTTP 400 for ffmpeg `CalledProcessError`, HTTP 500 for missing binary / whisper non-zero.** Flask 3.1: 400–499 are client request-data errors; 500–599 are server/application errors. Invalid media is client data. Alternative rejected: keep 500 for ffmpeg so the status stays "as today" — today is HTML 500, not a JSON contract.

3. **Keep `NamedTemporaryFile(delete=False)` and unlink in `finally`.** Python 3.8 docs: `delete=False` requires `os.unlink` after close because ffmpeg/whisper need a filesystem path. Alternative rejected: `delete=True` (file may vanish before the child process reads it).

4. **Tests: `unittest` + Flask `test_client` importing shipped `app`.** No new test runner. Drive `/transcribe` and `parse_transcription`; do not reimplement either. Holdout: missing-file 400 JSON still holds.

## Risks / Trade-offs

- [Risk] Local verify cannot run whisper-cli, so success-path 200 is unproven here → Mitigation: unit-test `parse_transcription` on real timestamp dumps; document Docker as launcher-unavailable; do not treat HTTP 200-from-mock as success.
- [Risk] ffmpeg stderr still prints to the worker log on bad media → Mitigation: acceptable; not in scope. Optional `capture_output=True` would hide operator signal.
- [Risk] Publishing an image requires push, which triggers `publish-docker.yml` → Mitigation: do not push; `CI/CD: n/a`.

## Migration Plan

Ship `src/app.py` + tests only. Rollback is git revert of that diff. No data migration.

## Open Questions

None. Status for invalid media is 400 (resolved above).
