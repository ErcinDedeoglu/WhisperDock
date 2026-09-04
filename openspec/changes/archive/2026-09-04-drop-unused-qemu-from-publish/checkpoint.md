# Checkpoint: drop-unused-qemu-from-publish

## Goal
Remove unused docker/setup-qemu-action from amd64-only publish.

## Acceptance
- No setup-qemu-action in publish-docker.yml
- platforms still linux/amd64
- unittest 8/8
- Chrome: N/A
- GHA 🐳 success; Hub tag cli-ok

## Non-goals
Removing Buildx; adding arm64

## Constraints
- Changing publish-docker.yml starts Docker Images
- Hub verify: --platform linux/amd64, /app/whisper/build/bin/whisper-cli
- Unittest from src/

## Active phase
propose

## Hypothesis
Delete QEMU step → Docker Images still succeeds on ubuntu-latest amd64.

## Expected signal
No setup-qemu-action; GHA success; Hub cli-ok

## Tasks
- 1.1 pending
- 2.1 pending

## Last action
Wrote OpenSpec artifacts.

## Next action
Validate, apply, unittest.
