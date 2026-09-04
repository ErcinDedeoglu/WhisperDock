# Lessons

## 2026-09-04 — return-json-for-transcribe-media-errors

- **Date:** 2026-09-04
- **Change:** return-json-for-transcribe-media-errors
- **Finding:** POST `/transcribe` with invalid media raised uncaught ffmpeg `CalledProcessError`, returning HTML 500 and leaking two `NamedTemporaryFile(delete=False)` paths.
- **Hypothesis:** Catch ffmpeg `CalledProcessError` / `FileNotFoundError` and unlink temps in `finally` → garbage POST is JSON with `error` and `leaked_tmp=[]`.
- **Action:** View-local try/except/finally in `transcribe_audio`; 400 for conversion failure, 500 for missing whisper-cli; stdlib unittest driving shipped Flask view.
- **Evidence:** Discover HTML 500 + 2 leaked temps. After: `GARBAGE_AUDIO status=400 type=application/json leaked_tmp=[] body='{"error":"Error in transcription"}'` on unittest (gate-1, gate-2) and gunicorn in `dublok/whisperdock:latest` with mounted `app.py` (two POSTs). Holdout missing-file still `{"error":"No file part"}`.
- **Outcome:** Hypothesis confirmed. Change archived. No push (`publish-docker.yml` would publish `dublok/whisperdock`).
- **Failure mode:** none this slice. First docker curl without waiting for gunicorn workers: connection reset (qemu/amd64 on arm64).
- **Confidence:** high for invalid-media JSON + cleanup; whisper success path not run here.
- **Applicability:** Any Flask JSON API that shells out with `check=True` and `NamedTemporaryFile(delete=False)` without `finally`.
- **Superseded lesson:** none
- **Pattern-Key:** json-api-uncaught-subprocess-leaks-delete-false-temps
- **Next experiment:** Rank remaining backlog — Python 3.8 EOL image base (reliability), or `MAX_CONTENT_LENGTH` (security). Do not push a branch while `on.push.branches: '**'` publishes production images.

## 2026-09-04 — upgrade-python-image-to-3-12

- **Date:** 2026-09-04
- **Change:** upgrade-python-image-to-3-12
- **Finding:** Published image ran CPython 3.8.20 (`FROM python:3.8`); 3.8 EOL 2024-10-07.
- **Hypothesis:** `FROM python:3.12-bookworm` → local image `sys.version_info[:2]==(3,12)` and unittest still exit 0.
- **Action:** Dockerfile + README 3.12; local tag `whisperdock-local:py312` (not `dublok/*`); unittest 5/5; synced `container-runtime`; archived.
- **Evidence:** `whisperdock-local:py312` python 3.12.14; Flask 3.1.3 / gunicorn 26.2.0; unittest OK; `openspec validate --specs` 2 passed. No push.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-upgrade-python-image-to-3-12`. Local commit f35e97d not pushed.
- **Failure mode:** none. Size-M wait-for-human in checkpoint was overridden by standing loop (local apply, no push).
- **Confidence:** high local 3.12; remote/CI unproven.
- **Applicability:** Docker Python bases still on 3.8; unpinned pip on EOL interpreters.
- **Superseded lesson:** none
- **Pattern-Key:** eol-cpython-docker-base-blocks-current-wheels
- **Next experiment:** Bound upload size via `MAX_CONTENT_LENGTH` in `src/app.py`. Still do not push.
