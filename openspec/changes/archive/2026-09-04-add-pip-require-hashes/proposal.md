## Why

`src/requirements.txt` pins versions but has no `--hash`. pip still trusts PyPI bytes. Official hash-checking mode verifies sha256 of every downloaded archive. The prior slice deferred hashes because MarkupSafe wheels are arch-specific; multiple `--hash` lines are ORed, so amd64 and arm64 can both be listed.

## What Changes

- Add sha256 `--hash` lines for every frozen package (py3-none-any wheels; MarkupSafe cp312 manylinux aarch64 + x86_64).
- Dockerfile: `pip install --no-cache-dir --no-deps --require-hashes --only-binary :all: -r /app/requirements.txt`.

Not **BREAKING** for `/transcribe`.

## Capabilities

### New Capabilities

- (none)

### Modified Capabilities

- `container-runtime`: frozen requirements MUST include sha256 hashes; pip install MUST use `--require-hashes`.

## Impact

- `src/requirements.txt`, `src/Dockerfile` pip RUN.
- No app.py changes.

## Problem

Version pins do not verify artifact integrity.

## Non-goals

- pip-tools, poetry, wheelhouse, hashing apt packages.
- Listing every MarkupSafe wheel (only the two linux cp312 manylinux used here).

## Hypothesis

If requirements include hashes for both linux/amd64 and linux/arm64 MarkupSafe wheels plus py3-none-any for the rest, and pip uses `--require-hashes --only-binary :all: --no-deps`, then local rebuild succeeds, image versions stay frozen, unittest exits 0, and GHA amd64 publish succeeds.

## Expected signal

- `src/requirements.txt` contains `--hash=sha256:` for each package.
- Dockerfile contains `--require-hashes`.
- Local + Hub metadata still `3.1.3 26.2.0 3.1.8 3.1.6 3.0.3 1.9.0 8.5.0 2.2.0`.
- Holdout missing-file 400 JSON.

## Research

Official pattern: https://pip.pypa.io/en/stable/topics/secure-installs/ (pip 26.x; `--require-hashes`; multiple hashes per package for platforms; `--only-binary :all:`)
Why current code is worse: freeze without hashes; pip trusts PyPI content
Chosen approach: PyPI sha256 for py3-none-any + MarkupSafe cp312 manylinux aarch64/x86_64; `--require-hashes --only-binary :all: --no-deps`
Rejected alternative: hash only the local arm64 MarkupSafe wheel (breaks GHA amd64)
Proof plan: local docker rebuild; unittest exit 0; GHA success; Chrome: N/A — no UI

Supporting: https://pip.pypa.io/en/stable/topics/repeatable-installs/#hash-checking

## Chosen and rejected approaches

- **Chosen:** multi-hash requirements + `--require-hashes --only-binary :all:`.
- **Rejected:** single-arch hashes.
- **Rejected:** hashing all 89 MarkupSafe wheels.

## Rollback

Restore unhashed freeze file and previous pip RUN.

## Acceptance checks

- unittest 6/6 from `src/`
- local image metadata matches freeze
- GHA publish-docker succeeds
- Chrome: N/A — no UI
