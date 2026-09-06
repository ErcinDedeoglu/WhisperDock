## Why

The service image runs gunicorn as uid 0 (`User=` empty; `id` reports root). Flask 3.1 gunicorn docs say gunicorn must not run as root because the app would run as root. Docker best practices: if the service can run without privileges, use `USER`. Port 5000 is unprivileged.

## What Changes

- Create a system user/group `app` with explicit UID/GID 10001.
- `USER app` before CMD.

Not **BREAKING** for `/transcribe`. Bind stays `0.0.0.0:5000`.

## Capabilities

### New Capabilities

- (none)

### Modified Capabilities

- `container-runtime`: the service process MUST NOT run as root.

## Impact

- `src/Dockerfile` only (groupadd/useradd/USER).
- No app.py, requirements, or CMD flag changes.

## Problem

gunicorn and Flask run as root inside the published image.

## Non-goals

- Dropping capabilities, read-only rootfs, gosu, changing the listen port.
- gunicorn body-size vs Flask 16 MB.

## Hypothesis

If the image sets `USER app` (uid 10001), then `docker inspect` User is `app`, `docker run --entrypoint id` is not uid 0, unittest still exits 0, and GHA still publishes.

## Expected signal

- Dockerfile has `USER app` and no later USER root.
- Image Config.User is `app`.
- Holdout missing-file 400 JSON.

## Research

Official pattern: https://docs.docker.com/build/building/best-practices/#user (`groupadd -r` / `useradd --no-log-init -r`; explicit UID/GID)
Why current code is worse: no USER; inspect User empty; process is root
Chosen approach: `app` uid/gid 10001 (`useradd --no-log-init`, not `-r`) then `USER app`
Rejected alternative: keep root (Flask: gunicorn as root is not secure)
Proof plan: inspect User; `id` not root; unittest exit 0; Chrome: N/A — no UI

Supporting: https://flask.palletsprojects.com/en/stable/deploying/gunicorn/ (Flask 3.1.x Binding Externally)

## Chosen and rejected approaches

- **Chosen:** dedicated `app` user 10001 + `USER app`.
- **Rejected:** numeric `USER 10001` without useradd (harder to read in `id`).
- **Rejected:** gosu (extra binary; no need to start as root).

## Rollback

Remove groupadd/useradd/USER.

## Acceptance checks

- unittest 6/6
- `docker inspect` User is `app`
- `docker run --entrypoint id` uid != 0
- Chrome: N/A — no UI
