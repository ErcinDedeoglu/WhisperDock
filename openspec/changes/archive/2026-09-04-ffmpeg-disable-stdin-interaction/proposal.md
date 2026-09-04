## Why

`/transcribe` runs `ffmpeg -y -i <file>` with no `-nostdin`. ffmpeg enables stdin interaction by default when stdin is not the input. Under gunicorn that can stall a worker waiting for console commands.

## What Changes

- Pass `-nostdin` on the ffmpeg argv (before `-i`).

Not **BREAKING** for `/transcribe` JSON contract.

## Capabilities

### New Capabilities

- (none)

### Modified Capabilities

- `transcribe-api`: media conversion MUST invoke ffmpeg with `-nostdin`.

## Impact

- `src/app.py` ffmpeg argv only.

## Problem

ffmpeg may read stdin in a WSGI worker.

## Non-goals

- subprocess `timeout=`; whisper-cli flags; gunicorn `--timeout`.

## Hypothesis

If ffmpeg argv includes `-nostdin`, then unittest still exits 0 including garbage-audio JSON 400, and the ffmpeg list in `app.py` contains `-nostdin`.

## Expected signal

- `app.py` ffmpeg argv has `-nostdin` before `-i`.
- Holdout missing-file 400 JSON.

## Research

Official pattern: https://ffmpeg.org/ffmpeg-all.html (`-stdin` on by default unless stdin is the input; disable with `-nostdin`)
Why current code is worse: `ffmpeg -y -i` only; stdin is not the input file
Chosen approach: add `-nostdin` to the existing argv
Rejected alternative: `stdin=subprocess.DEVNULL` only (shell-less equivalent; flag is the documented ffmpeg switch)
Proof plan: unittest exit 0; grep argv; Chrome: N/A — no UI

## Chosen and rejected approaches

- **Chosen:** `-nostdin` on ffmpeg argv.
- **Rejected:** `ffmpeg ... < /dev/null` (needs a shell).
- **Rejected:** subprocess timeout this slice.

## Rollback

Remove `-nostdin`.

## Acceptance checks

- unittest including holdout missing-file 400 JSON
- ffmpeg argv contains `-nostdin`
- Chrome: N/A — no UI
