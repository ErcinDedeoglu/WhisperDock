## Why

Apt packages are unpinned (`build-essential`, `cmake`, `ffmpeg`, `git`, `libsndfile1`). Rebuilds can pull newer bookworm versions. Docker apt-get best practice is version pinning for cache-busting and reproducibility.

Proven image (`whisperdock-local:pinned` after `--no-install-recommends`):
`build-essential=12.9` `cmake=3.25.1-1` `ffmpeg=7:5.1.9-0+deb12u1` `git=1:2.39.5-0+deb12u3` `libsndfile1=1.2.0-1+deb12u1`.

## What Changes

- Pin those five packages to the proven versions.

Not **BREAKING** if those versions remain in bookworm.

## Capabilities

### New Capabilities

- (none)

### Modified Capabilities

- `container-runtime`: apt packages MUST be version-pinned.

## Impact

- `src/Dockerfile` apt install lines.

## Problem

Unpinned apt install is a moving target across rebuilds.

## Non-goals

- Multi-stage drop of build-essential.
- Pinning every transitive apt package.

## Hypothesis

If the five packages are pinned to the proven versions, then the Dockerfile contains those `=`, a rebuild reports the same `dpkg-query` versions, ffmpeg still runs, unittest still exits 0.

## Expected signal

- Dockerfile has `ffmpeg=7:5.1.9-0+deb12u1` and the other four pins.
- Image `dpkg-query` matches.

## Research

Official pattern: https://docs.docker.com/build/building/best-practices/#apt-get (version pinning `package-foo=1.3.*`)
Why current code is worse: unpinned names only
Chosen approach: pin the five direct packages to versions from the current image
Rejected alternative: multi-stage this slice (M; shared-lib copy risk)
Proof plan: dpkg-query match; unittest exit 0; Chrome: N/A — no UI

## Chosen and rejected approaches

- **Chosen:** pin the five install names.
- **Rejected:** pin every dependency (unmaintainable).
- **Rejected:** multi-stage (next experiment).

## Rollback

Remove `=version` suffixes.

## Acceptance checks

- unittest holdout missing-file 400 JSON
- dpkg-query matches the five pins
- Chrome: N/A — no UI
