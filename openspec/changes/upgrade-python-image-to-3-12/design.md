## Context

See proposal.md for motivation. `src/Dockerfile` currently starts `FROM python:3.8` and later `pip install Flask gunicorn` with no pins. The published `dublok/whisperdock:latest` image reports Python 3.8.20, Flask 3.0.3, gunicorn 23.0.0. Host unittest already runs on CPython 3.12.9. Whisper-cli is compiled in the same image (`make CMAKE_ARGS="-DGGML_NATIVE=OFF"`). `publish-docker.yml` pushes `dublok/whisperdock` on every git push, so this slice must not push.

## Goals / Non-Goals

**Goals:**

- One `FROM` line change to a supported official Python image with an explicit Debian suite.
- Keep existing apt build deps, whisper make flags, gunicorn CMD, and `/transcribe` handlers.
- Prove the new image interpreter is 3.12 and that existing tests still pass.

**Non-Goals:**

- Dependency pinning, slim/alpine, multi-stage, Python 3.13+.
- CI workflow edits or Docker Hub publishes.

## Decisions

1. **Base tag `python:3.12-bookworm`**
   - Why: CPython 3.12 is in security support until 2028-10 (devguide). Docker official images recommend naming the Debian suite when installing extra packages. Bookworm is the current stable suite used by existing 3.12 tags. Host tests already run 3.12.9.
   - Rejected: `python:3.8` pin-old-wheels — EOL interpreter. `python:3.14` / `python:3` — moving target, newer than host tests. `python:3.12-slim` — official docs prefer the default image when compiling native code; we already install `build-essential`/`cmake`. `python:3.12-trixie` — newer Debian, extra apt risk this slice does not need.

2. **Leave `pip install Flask gunicorn` unpinned**
   - Why: this slice’s root cause is the interpreter, not lockfiles. On 3.12, current Flask 3.1.x and gunicorn 26.x are installable (`requires_python` >=3.9 and >=3.10).
   - Rejected: introducing a requirements.txt here — separate finding.

3. **README prerequisite only**
   - Update the documented Python 3.8 prerequisite to 3.12. No API doc changes.

4. **Size M — stop after propose unless apply is explicit**
   - Docker rebuild + whisper compile is a compatibility concern. Autonomous apply waits.

## Risks / Trade-offs

- [whisper.cpp fails to compile on bookworm/3.12] → Keep `CMAKE_ARGS="-DGGML_NATIVE=OFF"`; if compile fails, do not archive; revert `FROM`.
- [qemu/amd64 on arm64 host flakes gunicorn] → Known from prior lesson; wait for workers before HTTP checks; unittest does not need the image for JSON-error tests.
- [Unpinned pip pulls a breaking gunicorn 26] → Holdout `/transcribe` missing-file JSON 400; if gunicorn CLI flags change, stop and replan rather than drive-by CMD edits unless required to boot.
- [Any git push publishes production images] → Do not push this slice.

## Migration Plan

- Local: edit Dockerfile + README, rebuild image tagged locally (not `dublok/whisperdock`), run unittest + python version assert.
- Rollback: restore `FROM python:3.8` and README “Python 3.8”.
- Production image updates only when a human authorizes a push.

## Open Questions

None. Apply is deferred until explicit authorization because this change is size M.
