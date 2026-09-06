# Checkpoint: disable-publish-checkout-credentials

## Goal
Set persist-credentials: false on publish-docker.yml checkout.

## Acceptance
- Parsed checkout with.persist-credentials == false
- unittest 8/8
- Chrome: N/A
- GHA 🐳 success; Hub tag cli-ok

## Non-goals
sync-whisper.yml git push jobs

## Constraints
- Changing publish-docker.yml starts Docker Images
- Do not tag/push dublok/* locally
- Unittest from src/
- Hub verify: --platform linux/amd64, /app/whisper/build/bin/whisper-cli

## Active phase
propose

## Hypothesis
persist-credentials false → Docker Images still succeeds (Hub uses Docker secrets).

## Expected signal
YAML persist-credentials false; GHA success; Hub cli-ok

## Tasks
- 1.1 pending
- 2.1 pending

## Last action
Wrote OpenSpec artifacts.

## Next action
Validate, apply YAML, unittest.
