## Context

Both `subprocess.run` calls omit `timeout`. gunicorn CMD uses `--timeout 300`. Python 3.12 `subprocess.run(..., timeout=)` raises `TimeoutExpired` and kills the child.

## Goals / Non-Goals

**Goals:** app-level deadline 240s; JSON 500 on timeout.

**Non-Goals:** gunicorn flag change.

## Decisions

1. **`timeout=240` on ffmpeg and whisper-cli**
   - Why: below 300 so the view can answer before the worker is SIGKILLed.
2. **TimeoutExpired → 500 JSON**
   - Why: same contract as missing whisper-cli; not a client media error.

## Risks / Trade-offs

- [Long valid audio] → 16 MB wav can exceed 240s of CPU; 500 is preferred to a dead worker.
