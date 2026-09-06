# Checkpoint: bound-publish-job-timeout

## Goal
Set timeout-minutes: 20 on linux-build-and-push.

## Acceptance
- Parsed timeout-minutes == 20
- unittest 8/8
- Chrome: N/A
- GHA 🐳 success under 20m; Hub cli-ok

## Non-goals
sync-whisper.yml; step timeouts

## Constraints
- Changing publish-docker.yml starts Docker Images
- Hub verify: --platform linux/amd64, /app/whisper/build/bin/whisper-cli
- Unittest from src/

## Active phase
propose

## Hypothesis
timeout-minutes 20 → Docker Images still succeeds (~2.5m observed).

## Expected signal
YAML 20; GHA success; Hub cli-ok

## Tasks
- 1.1 pending
- 2.1 pending

## Last action
Wrote OpenSpec artifacts.

## Next action
Validate, apply, unittest.
