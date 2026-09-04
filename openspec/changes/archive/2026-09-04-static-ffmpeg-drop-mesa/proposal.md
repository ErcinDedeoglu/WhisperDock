## Why

Debian `ffmpeg` on slim-bookworm depends on `libavdevice59`, which pulls `libllvm15` (107 MB) and Mesa DRI. The service only resamples audio to 16 kHz mono WAV. The published image is 1.06 GB mostly because of that video stack.

## What Changes

- Runtime copies a pinned static `ffmpeg` from `mwader/static-ffmpeg:9.0.1`.
- Runtime apt drops `ffmpeg` and `libsndfile1`; keeps `libgomp1` for whisper-cli.
- Builder apt pins stay as they are.

## Capabilities

### New Capabilities

### Modified Capabilities

- `container-runtime`: published runtime MUST NOT install `libllvm15`; MUST provide `ffmpeg` on PATH.

## Impact

- `src/Dockerfile` runtime apt + one `COPY --from`.
- No API change. `app.py` still calls `ffmpeg`.

## Problem

Hub/local `whisperdock-local:pinned` is 1.06 GB with `libllvm15` and `libgl1-mesa-dri` installed.

## Non-goals

- Replacing builder ffmpeg. Copying ffprobe. Compiling ffmpeg. Alpine. Digest pins of the static image.

## Hypothesis

If runtime COPY `--from=mwader/static-ffmpeg:9.0.1 /ffmpeg /usr/local/bin/ffmpeg` and apt installs only `libgomp1`, then `dpkg-query -W libllvm15` exits non-zero, `ffmpeg -version` works, image size is below 1.06 GB, unittest still exits 0.

## Expected signal

- `dpkg-query -W libllvm15` non-zero.
- `command -v ffmpeg` succeeds.
- Image smaller than 1.06 GB.

## Research

- https://github.com/wader/static-ffmpeg/blob/master/README.md — `COPY --from=mwader/static-ffmpeg:9.0.1 /ffmpeg /usr/local/bin/`; multi-arch amd64/arm64 since 5.0.1-3.
- https://hub.docker.com/r/mwader/static-ffmpeg — static PIE, no external deps.
- https://johnvansickle.com/ffmpeg/ — rejected (amd64-only).

## Chosen / rejected

- Chosen: pinned `mwader/static-ffmpeg:9.0.1` COPY of `/ffmpeg` only.
- Rejected: johnvansickle (no arm64). Compiling ffmpeg (huge). `apt remove libavdevice59` (removes debian ffmpeg).

## Rollback

Restore runtime apt `ffmpeg` + `libsndfile1`; delete the COPY --from line.

## Acceptance

- unittest 8/8 including holdout missing-file JSON 400.
- Chrome: N/A — no UI.
