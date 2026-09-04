## Why

The published image is 2.34 GB and ships `build-essential`, `cmake`, `git`, and the whisper.cpp source. Runtime only needs `whisper-cli`, four `.so` files, the model, ffmpeg, libgomp1, and Flask. ldd of whisper-cli: libwhisper/libggml* plus libstdc++/libgomp/libc.

## What Changes

- Builder stage: current compile.
- Runtime stage: python:3.12-bookworm + pinned ffmpeg/libsndfile1/libgomp1; COPY only whisper-cli, libs, model, app.

Not **BREAKING** for `/transcribe` if those files land at the same paths.

## Capabilities

### New Capabilities

- (none)

### Modified Capabilities

- `container-runtime`: the runtime image MUST NOT install `cmake`; MUST contain whisper-cli at `/app/whisper/build/bin/whisper-cli`.

## Impact

- `src/Dockerfile` structure.

## Problem

Compilers and C++ sources in production increase size and attack surface.

## Non-goals

- slim/alpine base; dropping ffmpeg.

## Hypothesis

If runtime COPY only whisper-cli + libs + model, then `cmake` is not installed, whisper-cli still runs, image size is below 2.34 GB, unittest still exits 0.

## Expected signal

- `dpkg-query -W cmake` exits non-zero.
- `/app/whisper/build/bin/whisper-cli` executable.
- Image smaller than 2.34 GB.

## Research

Official pattern: https://docs.docker.com/build/building/multi-stage/ (COPY --from builder; compilers stay in builder)
Why current code is worse: 2.34 GB; gcc in the final image
Chosen approach: two-stage bookworm; COPY whisper-cli + four .so + model
Rejected alternative: apt purge in the same image (earlier layers still contain compilers)
Proof plan: which g++ fails; whisper-cli -h; docker images size; unittest; Chrome: N/A — no UI

Supporting: https://docs.docker.com/build/building/best-practices/#dont-install-unnecessary-packages

## Chosen and rejected approaches

- **Chosen:** multi-stage COPY of runtime artifacts.
- **Rejected:** purge after make (layers remain).

## Rollback

Revert to single-stage Dockerfile.

## Acceptance checks

- unittest holdout missing-file 400 JSON
- runtime has no g++; whisper-cli exists
- Chrome: N/A — no UI
