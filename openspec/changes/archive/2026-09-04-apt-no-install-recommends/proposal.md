## Why

`apt-get install -y` omits `--no-install-recommends`. Apt lists are removed in a **second** RUN, so the first layer still stores `/var/lib/apt/lists`. Docker apt-get best practice is one RUN: update, install `--no-install-recommends`, `rm -rf /var/lib/apt/lists/*`.

## What Changes

- Merge the two apt RUN instructions.
- Add `--no-install-recommends`.

Not **BREAKING** for `/transcribe` if ffmpeg and build tools still install.

## Capabilities

### New Capabilities

- (none)

### Modified Capabilities

- `container-runtime`: apt install MUST use `--no-install-recommends` and MUST delete apt lists in the same RUN.

## Impact

- `src/Dockerfile` apt RUN only. Rebuilds all later layers.

## Problem

Recommended packages and apt lists bloat the image.

## Non-goals

- Pinning apt versions; multi-stage slim; dropping build-essential after compile.

## Hypothesis

If apt is one RUN with `--no-install-recommends` and list cleanup, then the Dockerfile has no separate `apt-get clean` RUN, ffmpeg still exists, unittest still exits 0.

## Expected signal

- Dockerfile install line contains `--no-install-recommends`.
- `rm -rf /var/lib/apt/lists/*` is in that same RUN.
- `ffmpeg -version` works in the image.

## Research

Official pattern: https://docs.docker.com/build/building/best-practices/#apt-get (`-y --no-install-recommends` and `rm -rf /var/lib/apt/lists/*` in the same RUN)
Why current code is worse: no `--no-install-recommends`; lists deleted in a later layer
Chosen approach: merge RUN + `--no-install-recommends`
Rejected alternative: keep two RUNs and only add the flag (lists still in layer 1)
Proof plan: grep Dockerfile; image `ffmpeg -version`; unittest exit 0; Chrome: N/A — no UI

## Chosen and rejected approaches

- **Chosen:** official one-RUN apt pattern.
- **Rejected:** `apt-get clean` extra (Debian images already clean).

## Rollback

Restore the two RUN blocks.

## Acceptance checks

- unittest holdout missing-file 400 JSON
- Dockerfile has `--no-install-recommends` and same-RUN list cleanup
- Chrome: N/A — no UI
