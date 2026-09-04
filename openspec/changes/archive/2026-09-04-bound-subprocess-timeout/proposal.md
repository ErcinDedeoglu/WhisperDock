## Why

ffmpeg and whisper-cli `subprocess.run` have no `timeout`. A hung child occupies a gunicorn sync worker until `--timeout 300` kills the worker (connection reset), not JSON 500.

## What Changes

- `timeout=240` on both `subprocess.run` calls (below gunicorn 300).
- Catch `subprocess.TimeoutExpired` and return JSON 500 `Error in transcription`.

Not **BREAKING** for existing JSON errors.

## Capabilities

### New Capabilities

- (none)

### Modified Capabilities

- `transcribe-api`: conversion and transcription subprocesses MUST have a timeout shorter than the gunicorn worker timeout; timeout SHALL be JSON 500.

## Impact

- `src/app.py`, `src/test_app.py`.

## Problem

Hung ffmpeg/whisper has no app-level deadline; gunicorn kills the worker instead.

## Non-goals

- Changing gunicorn `--timeout 300`.
- ffmpeg `-timelimit`.

## Hypothesis

If both `subprocess.run` use `timeout=240` and `TimeoutExpired` returns JSON 500, then unittest including a timeout case exits 0 and holdout missing-file stays 400 JSON.

## Expected signal

- Both `subprocess.run` calls include `timeout=240`.
- TimeoutExpired → 500 `{"error":"Error in transcription"}`.
- Holdout missing-file 400 JSON.

## Research

Official pattern: https://docs.python.org/3.12/library/subprocess.html#subprocess.run (`timeout` raises `TimeoutExpired`)
Why current code is worse: no timeout; gunicorn `--timeout 300` kills the worker
Chosen approach: `timeout=240` + catch `TimeoutExpired` as JSON 500
Rejected alternative: raise gunicorn timeout only (still no JSON; child can outlive the request)
Proof plan: unittest timeout path 500 JSON; holdout 400; Chrome: N/A — no UI

Supporting: https://docs.gunicorn.org/en/stable/settings.html#timeout (silent workers killed after timeout seconds)

## Chosen and rejected approaches

- **Chosen:** subprocess timeout 240s, JSON 500 on TimeoutExpired.
- **Rejected:** gunicorn `--timeout 0`.
- **Rejected:** ffmpeg `-timelimit` only (does not cover whisper-cli).

## Rollback

Remove `timeout=` and the TimeoutExpired handlers.

## Acceptance checks

- unittest including timeout 500 JSON and holdout missing-file 400 JSON
- Chrome: N/A — no UI
