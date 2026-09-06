# Checkpoint: bound-sync-whisper-job-timeout

## Goal
Set timeout-minutes: 10 on both sync-whisper.yml jobs.

## Acceptance
- Both jobs timeout-minutes == 10
- unittest 8/8
- Chrome: N/A
- Docker Images: N/A (path filter)
- Do not dispatch sync-whisper

## Non-goals
Running the sync; persist-credentials; publish-docker.yml

## Constraints
- Origin push main allowed
- Changing sync-whisper.yml does not start Docker Images
- Unittest from src/

## Active phase
propose

## Hypothesis
timeout-minutes 10 on both jobs → YAML ok; unittest 8/8; no 🐳 run.

## Expected signal
yaml-ok 10/10; unittest 8/8; gh run list no new Docker Images for this SHA

## Tasks
- 1.1 pending
- 2.1 pending

## Last action
Wrote OpenSpec artifacts.

## Next action
Validate, apply, unittest.
