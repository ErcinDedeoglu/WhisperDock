# Checkpoint: return-json-for-transcribe-media-errors

## Invariants

- **Goal:** One system-improve slice: JSON errors + temp cleanup on `/transcribe` media conversion failure.
- **Acceptance:** OpenSpec explore→propose→apply (S)→verify→sync→archive; hypothesis/expected signal match; Chrome: N/A — no UI; no push to main; no Docker Hub publish; no secrets.
- **Non-goals:** parse regex, whisper flags, Python 3.8 upgrade, CI workflow, MAX_CONTENT_LENGTH, extra slices.
- **Constraints:** Smallest root-cause diff in `src/app.py`; stdlib tests only; do not push (`publish-docker.yml` pushes `dublok/whisperdock` on every branch).
- **Decisions:** View-local try/except/finally; 400 for ffmpeg CalledProcessError; 500 for FileNotFoundError / whisper non-zero; unlink temps in finally.

## Current State

- **Phase:** archive complete.
- **Hypothesis:** Catch ffmpeg CalledProcessError / FileNotFoundError and unlink temps in finally → garbage POST returns JSON `error` and leaked_tmp=[].
- **Expected signal:** GARBAGE_AUDIO status=400 type=application/json body has `error`; leaked_tmp=[]; holdout missing-file 400 JSON `No file part`. **Observed:** unittest twice + gunicorn twice.
- **Rollback:** revert `src/app.py` and `src/test_app.py`.
- **Task states:** 1.1 done, 2.1 done, 3.1 done.
- **Files changed:** `src/app.py`, `src/test_app.py`, `openspec/specs/transcribe-api/spec.md`, archived change tree.
- **Verification:** pass; retry count 0.
- **Failed approaches:** none.
- **Evidence paths:** `{SCRATCH}/discover-baseline.log`, `{SCRATCH}/gate-1.log`, `{SCRATCH}/gate-2.log`, `{SCRATCH}/behavior-test.log`, `{SCRATCH}/docker-gunicorn.log`
- **Confidence:** high for JSON error + temp cleanup on invalid media; success-path whisper-cli not executed here.
- **Next action:** stop (one slice). Do not push.

## Facts / Assumptions / Open questions

- **Facts:** Before: HTML 500 + 2 leaked temps. After: JSON 400 `{"error":"Error in transcription"}` leaked_tmp=[] on Flask test-client (x2) and gunicorn in `dublok/whisperdock:latest` with mounted `app.py` (x2). Missing-file holdout still 400 JSON. No push. Chrome N/A.
- **Assumptions:** ffmpeg on PATH for unittest. Docker amd64 image under qemu needs a few seconds before gunicorn accepts connections.
- **Open questions:** none.

## Events

- 2026-09-04: Discover selected this finding (correctness > reliability). Research: Flask 3.1 errorhandling JSON API errors; Python 3.8 tempfile unlink; subprocess CalledProcessError.
- 2026-09-04: Explore read-only. Propose created change `return-json-for-transcribe-media-errors`.
- 2026-09-04: Apply 1.1 tests red (HTML 500 + leak). Apply 2.1 handler. Tests green.
- 2026-09-04: Verify unittest x2 and gunicorn garbage POST x2. Sync main spec. Archive `2026-09-04-return-json-for-transcribe-media-errors`.
