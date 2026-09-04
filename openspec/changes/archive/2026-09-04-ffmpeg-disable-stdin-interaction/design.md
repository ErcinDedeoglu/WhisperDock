## Context

`subprocess.run(["ffmpeg", "-y", "-i", temp_path, ...], check=True)` with no timeout and no `-nostdin`. ffmpeg docs: stdin interaction on unless stdin is the input. Input is a file path.

## Goals / Non-Goals

**Goals:** ffmpeg argv includes `-nostdin`.

**Non-Goals:** whisper timeout; gunicorn timeout.

## Decisions

1. **`-nostdin` immediately after `-y`**
   - Why: official disable switch; no shell.
2. **Not DEVNULL-only**
   - Rejected as the primary fix; flag is what ffmpeg documents.

## Risks / Trade-offs

- [Existing tests already finish] → hang is latent; unittest still proves conversion path.
