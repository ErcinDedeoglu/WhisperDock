## Why

`src/Dockerfile` pins only `Flask==3.1.3 gunicorn==26.2.0`. Transitives still float: the proven image has blinker 1.9.0, click 8.5.0, itsdangerous 2.2.0, Jinja2 3.1.6, MarkupSafe 3.0.3, Werkzeug 3.1.8. A rebuild can pull newer transitives without a code change.

## What Changes

- Add `src/requirements.txt` with `pip freeze` of those eight packages from `whisperdock-local:pinned`.
- Install with `pip install --no-cache-dir --no-deps -r /app/requirements.txt`.
- Copy `requirements.txt` before `app.py` so app edits do not bust the pip layer.

Not **BREAKING** for `/transcribe`.

## Capabilities

### New Capabilities

- (none)

### Modified Capabilities

- `container-runtime`: the image MUST install the frozen Flask/gunicorn dependency set (direct + transitives), not unpinned transitives.

## Impact

- `src/requirements.txt` (new), `src/Dockerfile` pip/COPY.
- No app.py or workflow trigger changes.

## Problem

Direct-dep pins still let Werkzeug/Jinja2/etc. float.

## Non-goals

- Hash-checking (platform wheels differ amd64 vs arm64).
- pip-tools, poetry, uv as new tools.
- Pinning apt, pip, or the Python base digest.

## Hypothesis

If Dockerfile installs `--no-deps -r requirements.txt` containing the eight frozen `==` pins, then a rebuild reports those exact versions via `importlib.metadata`, the Dockerfile has no inline `pip install Flask`, and unittest still exits 0.

## Expected signal

- `src/requirements.txt` lists Flask==3.1.3, gunicorn==26.2.0, and the six transitives from freeze.
- Local image metadata matches those versions.
- Holdout: missing-file POST still 400 JSON.

## Research

Official pattern: https://pip.pypa.io/en/stable/topics/repeatable-installs/ (pip 26.x; pin with `==` including transitives from `pip freeze`; `--no-deps` extra insurance; works across OSes/arch)
Why current code is worse: only two direct pins; transitives unresolved at rebuild
Chosen approach: freeze file + `--no-deps -r`
Rejected alternative: `--hash` on every line (pip secure-installs all-or-nothing; MarkupSafe wheels are arch-specific, so one-arch hashes break the other)
Proof plan: unittest exit 0; image metadata == freeze; Chrome: N/A — no UI

Supporting official sources:
- https://pip.pypa.io/en/stable/cli/pip_freeze/
- https://pip.pypa.io/en/stable/topics/secure-installs/ (hash mode deferred)

## Chosen and rejected approaches

- **Chosen:** `pip freeze` of the proven image into `src/requirements.txt`; install `--no-deps -r`. Cross-arch; smallest next pip repeatability level.
- **Rejected:** hashes this slice (GHA linux/amd64 vs local linux/arm64 wheels).
- **Rejected:** pip-tools/poetry (new dependency).

## Rollback

Restore inline `pip install Flask==3.1.3 gunicorn==26.2.0`; delete `src/requirements.txt`.

## Acceptance checks

- `cd src && python3 -m unittest test_app.py -v` exits 0
- Image `importlib.metadata` matches the eight frozen versions
- Dockerfile uses `-r` and does not `pip install Flask==` inline
- Holdout missing-file 400 JSON
- Chrome: N/A — no UI
