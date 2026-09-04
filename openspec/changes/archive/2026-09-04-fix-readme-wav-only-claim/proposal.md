## Why

README API Usage says “Ensure your audio file is in WAV format with a sample rate of 16kHz.” The service already converts with `ffmpeg -i … -ar 16000 -ac 1 -vn`. That sentence is false and makes users pre-convert.

## What Changes

- Replace the WAV-only sentence with: the service converts readable audio to 16 kHz mono WAV; uploads are capped at 16 MB.

Not **BREAKING**.

## Capabilities

### New Capabilities

- (none — `skip_specs: true`; README copy is not a service behavior contract)

### Modified Capabilities

- (none)

## Impact

- `README.md` API Usage one sentence.

## Problem

Docs require a format the API already converts.

## Non-goals

- Changing ffmpeg argv; `/health` docs; curl example filename.

## Hypothesis

If that sentence is replaced, then README no longer requires WAV 16 kHz, unittest still exits 0.

## Expected signal

- Phrase `WAV format with a sample rate of 16kHz` absent.
- Conversion (`-ar 16000`) still mentioned.

## Research

Official pattern: https://ffmpeg.org/ffmpeg.html (`-i` input; `-ar` sample rate; `-ac` channels)
Why current code is worse: README WAV-only vs `ffmpeg -i` + `-ar 16000 -ac 1`
Chosen approach: one-sentence correction
Rejected alternative: list every ffmpeg demuxer
Proof plan: grep README; unittest exit 0; Chrome: N/A — no UI

## Chosen and rejected approaches

- **Chosen:** describe conversion, not a client WAV requirement.
- **Rejected:** delete the sentence with no replacement (loses 16 kHz/16 MB facts).

## Rollback

Restore the WAV-only sentence.

## Acceptance checks

- README lacks `WAV format with a sample rate of 16kHz`
- unittest holdout missing-file 400 JSON
- Chrome: N/A — no UI
