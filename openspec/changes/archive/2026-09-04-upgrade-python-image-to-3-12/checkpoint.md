# Checkpoint: upgrade-python-image-to-3-12

## Invariants

- **Goal:** One system-improve slice: move the service image off EOL CPython 3.8 onto supported CPython 3.12.
- **Acceptance:** Dockerfile FROM 3.12-bookworm; README 3.12; local image python 3.12; unittest exit 0; Chrome N/A; no push.
- **Non-goals:** Pin Flask/gunicorn; slim/alpine/trixie; Python 3.13+; MAX_CONTENT_LENGTH; CI workflow; `/transcribe` contract changes.
- **Constraints:** Size M. Local apply authorized by standing loop. Do not push.
- **Decisions:** `FROM python:3.12-bookworm`; leave pip unpinned; README prerequisite 3.12; new capability `container-runtime`.

## Current State

- **Phase:** archive.
- **Hypothesis:** `FROM python:3.12-bookworm` → container `sys.version_info[:2] == (3, 12)` and unittest still exits 0.
- **Expected signal:** Before image python 3.8.20; after local image 3.12.x; `python3 -m unittest test_app.py -v` exit 0; holdout missing-file 400 JSON.
- **Rollback:** revert Dockerfile FROM and README Python line.
- **Task states:** 1.1–2.2 done.
- **Files changed:** `src/Dockerfile` FROM python:3.12-bookworm; `README.md` Python 3.12; OpenSpec + loop files.
- **Verification:** re-run unittest 5/5 EXIT 0; image 3.12.14; Chrome N/A; CI/CD n/a (no push); retry 0.
- **Failed approaches:** none.
- **Evidence paths:** published image `dublok/whisperdock:latest` python 3.8.20 / Flask 3.0.3 / gunicorn 23.0.0; unittest OK on host 3.12.9; MAX_CONTENT_LENGTH None (deferred).
- **Confidence:** high for local 3.12 image + unittest. Chrome N/A. No push.
- **Next action:** none for this change. Compound lesson then next backlog item (MAX_CONTENT_LENGTH). Do not push.

## Facts / Assumptions / Open questions

- **Facts:** PEP 569 EOL 2024-10-07; devguide 3.8 unsupported, 3.12 security through 2028-10; Flask 3.1.3 requires >=3.9; gunicorn 26.2.0 requires >=3.10; no active change besides this one after propose; gates: unittest 5/5 OK.
- **Assumptions:** bookworm apt packages `build-essential cmake git libsndfile1 ffmpeg` remain available; unpinned pip on 3.12 installs current Flask/gunicorn without CMD flag changes.
- **Open questions:** none. Loop assumed local apply is authorized; push still forbidden.

## Events

- 2026-09-04: Preflight on `main` @ e7253f2. No active change. Lessons: do not push while publish-on-all-branches.
- 2026-09-04: Discover — unittest pass. Ranked Python 3.8 EOL (reliability) over MAX_CONTENT_LENGTH (security).
- 2026-09-04: Research — PEP 569, devguide versions, Docker Hub python, Flask 3.1 drop 3.8, gunicorn >=3.10.
- 2026-09-04: Explore read-only. Size M. Propose `upgrade-python-image-to-3-12`. Stop before apply.
- 2026-09-04: Loop start restored checkpoint. Applied 1.1 only (`FROM python:3.12-bookworm`). Next 1.2.
- 2026-09-04: Applied 1.2 README Python 3.12; `rg` shows no 3.8. Next 2.1.
- 2026-09-04: 2.1 pass. Image `whisperdock-local:py312` (not dublok/*). Python 3.12.14. Flask 3.1.3 gunicorn 26.2.0. Next 2.2.
- 2026-09-04: 2.2 pass. unittest 5/5 OK. Holdout missing-file JSON 400. Next verify/sync.
- 2026-09-04: Verify re-run unittest OK. Local commit f35e97d. Not pushed. Next sync.
- 2026-09-04: Synced `openspec/specs/container-runtime/spec.md`. `openspec validate --specs` 2 passed. Next archive. No push.
- 2026-09-04: Archive to `openspec/changes/archive/2026-09-04-upgrade-python-image-to-3-12`. Specs already synced. No push.
