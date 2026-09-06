## Why

The published image `dublok/whisperdock:latest` runs CPython 3.8.20 (`FROM python:3.8`). Python 3.8 reached end-of-life on 2024-10-07 (PEP 569); it receives no further security fixes. The Dockerfile also does unpinned `pip install Flask gunicorn`. Current Flask 3.1.x requires Python >=3.9 and gunicorn 26.x requires Python >=3.10, so rebuilds stay stuck on frozen 3.8-era wheels or become increasingly unsupportable.

## What Changes

- Replace `FROM python:3.8` with the official `python:3.12-bookworm` image so the container interpreter is a still-supported CPython (security support through 2028-10).
- Update the README development prerequisite from Python 3.8 to Python 3.12.
- Add a container-runtime requirement that the service image MUST run a non-EOL CPython 3.12.x.
- **Not breaking** for the HTTP `/transcribe` JSON contract. Runtime version inside the image changes.

## Capabilities

### New Capabilities

- `container-runtime`: The Docker image used to run the transcription service must use a supported CPython line and an explicit Debian suite tag.

### Modified Capabilities

- None. `/transcribe` request/response behavior is unchanged.

## Impact

- `src/Dockerfile` base image (and therefore every rebuild of whisper-cli, Flask, and gunicorn).
- `README.md` listed Python version.
- Unpinned `pip install Flask gunicorn` will resolve to current PyPI versions compatible with 3.12 (Flask 3.1.x, gunicorn 26.x) instead of the frozen 3.8-era Flask 3.0.3 / gunicorn 23.0.0 in today's published image.
- No change to `src/app.py` request handlers, CI publish triggers, or Docker Hub push behavior.

## Non-goals

- Pinning Flask/gunicorn versions.
- Switching to slim/alpine bases, multi-stage builds, or Debian trixie.
- Jumping to Python 3.13/3.14.
- Adding `MAX_CONTENT_LENGTH`, changing `publish-docker.yml`, or pushing to `origin` (any push publishes `dublok/whisperdock`).
- Changing `/transcribe` behavior or whisper-cli flags.

## Hypothesis

Changing `FROM python:3.8` to `FROM python:3.12-bookworm` yields a locally built image whose `sys.version_info[:2] == (3, 12)` and whose existing unittest suite plus garbage `/transcribe` JSON error path still pass.

## Expected signal

- Before: `docker run --rm --entrypoint python dublok/whisperdock:latest -c "import sys; print(sys.version)"` → `3.8.20`.
- After: locally tagged image prints Python 3.12.x; `python3 -m unittest test_app.py -v` exits 0; holdout missing-file POST remains `400` JSON `{"error":"No file part"}`.

## Research

Official pattern: https://devguide.python.org/versions/ (CPython 3.8 end-of-life 2024-10-07; 3.12 security support through 2028-10). Image tag pattern: https://hub.docker.com/_/python (specify Debian suite; default image when compiling native deps). Flask 3.1 dropped 3.8: https://flask.palletsprojects.com/en/stable/changes/. gunicorn 26 requires Python >=3.10: https://pypi.org/project/gunicorn/.
Why current code is worse: `FROM python:3.8` + published image 3.8.20; PEP 569 froze 3.8 after 3.8.20.
Chosen approach: `python:3.12-bookworm` (matches host unittest 3.12.9; still-supported; explicit suite).
Rejected alternative: pin Flask==3.0.3 and gunicorn==23.0.0 on 3.8 (leaves EOL interpreter); `python:3.14` (newer than host tests, more whisper-compile risk).
Proof plan: unittest exit 0; docker run python version assert 3.12; Chrome: N/A — no UI.

## Rollback

Revert `src/Dockerfile` `FROM` line and the README Python prerequisite.

## Acceptance checks

- `cd src && python3 -m unittest test_app.py -v` exits 0.
- Local image: `python -c "import sys; assert sys.version_info[:2] == (3, 12)"` exits 0.
- Holdout: POST `/transcribe` with no file part still 400 JSON `{"error":"No file part"}`.
- Chrome: N/A — no UI.
- Do not push; `publish-docker.yml` publishes `dublok/whisperdock` on every branch. CI/CD and Docker Hub environment: n/a this slice.
