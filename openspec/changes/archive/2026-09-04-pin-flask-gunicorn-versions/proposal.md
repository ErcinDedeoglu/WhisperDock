## Why

`src/Dockerfile` installs Flask and gunicorn with unpinned `pip install Flask gunicorn`. Host Python has Flask 3.1.2 / gunicorn 23.0.0; `whisperdock-local:py312` has Flask 3.1.3 / gunicorn 26.2.0. Rebuilds resolve whatever PyPI serves, so the WSGI stack can jump major versions without a code change.

## What Changes

- Pin the Dockerfile pip install to `Flask==3.1.3` and `gunicorn==26.2.0` (versions already running in `whisperdock-local:py312`).
- Require the service image to install those exact versions, not floating latest.

Not **BREAKING** for `/transcribe`. Rebuilds stop tracking unpinned PyPI latest.

## Capabilities

### New Capabilities

- (none)

### Modified Capabilities

- `container-runtime`: Flask and gunicorn in the service image MUST be exact pinned versions (`Flask==3.1.3`, `gunicorn==26.2.0`).

## Impact

- `src/Dockerfile` pip install line only.
- No `app.py`, unittest, whisper.cpp, or workflow changes.
- Origin push to `main` is allowed (publish-docker is main-only).

## Problem

Unpinned pip install makes image Flask/gunicorn a moving target across rebuilds.

## Non-goals

- `requirements.txt`, hash-checking, pip-tools, poetry, or uv.
- Pinning apt packages, pip itself, or the `python:3.12-bookworm` digest.
- Upgrading to Flask 3.2.x.
- Removing unused `import socket`.
- Changing gunicorn worker/timeout flags or `/transcribe` behavior.

## Hypothesis

If the Dockerfile uses `pip install Flask==3.1.3 gunicorn==26.2.0`, then a rebuild of that layer reports those versions via `importlib.metadata`, the Dockerfile contains both `==` pins, and `python3 -m unittest test_app.py` still exits 0.

## Expected signal

- Dockerfile has `Flask==3.1.3` and `gunicorn==26.2.0` and no unpinned `pip install Flask gunicorn`.
- Local image: Flask 3.1.3, gunicorn 26.2.0.
- Holdout: missing-file POST still 400 JSON `{"error":"No file part"}`.

## Research

Official pattern: https://pip.pypa.io/en/stable/topics/repeatable-installs/ (pip 26.x; pin with `==`)
Why current code is worse: `RUN pip install Flask gunicorn` (no version); host 3.1.2/23.0.0 vs image 3.1.3/26.2.0
Chosen approach: one-line `==` pins of the two direct deps already proven in `whisperdock-local:py312`
Rejected alternative: requirements.txt with hashes / pip freeze of transitives (pip's next repeatability level; extra file and `--no-deps` scope)
Proof plan: unittest exit 0 from `src/`; image metadata assert; Chrome: N/A — no UI

Supporting official sources:
- https://pip.pypa.io/en/stable/reference/requirement-specifiers/ (`SomeProject == 1.3`)
- https://github.com/hadolint/hadolint/wiki/DL3013 (pin versions in pip)
- https://docs.docker.com/build/building/best-practices/#apt-get (version pinning reduces unanticipated package changes)
- https://flask.palletsprojects.com/en/stable/deploying/gunicorn/ (Flask 3.1.x + gunicorn)

## Chosen and rejected approaches

- **Chosen:** Inline `Flask==3.1.3 gunicorn==26.2.0` on the existing Dockerfile `RUN`. Smallest reversible surface; matches hadolint DL3013 and pip `==` pinning.
- **Rejected:** New `requirements.txt` (two packages; extra COPY; same pin strength for direct deps).
- **Rejected:** Hash-checking + freeze of transitive deps (correct next step, out of this slice).
- **Rejected:** Pin host-venv 3.1.2/23.0.0 (image already runs 3.1.3/26.2.0 and passed unittest).

## Rollback

Revert the `src/Dockerfile` pip install line.

## Acceptance checks

- `cd src && python3 -m unittest test_app.py -v` exits 0
- Dockerfile contains `Flask==3.1.3` and `gunicorn==26.2.0` and does not contain `pip install Flask gunicorn` without `==`
- Local rebuild image: Flask 3.1.3 and gunicorn 26.2.0 via `importlib.metadata.version`
- Holdout: POST `/transcribe` with no file part still 400 JSON `{"error": "No file part"}`
- Chrome: N/A — no UI
