# Checkpoint: serialize-publish-per-ref

## Goal
Serialize publish runs per workflow+ref without cancelling in-flight Hub pushes.

## Acceptance
- concurrency.group includes github.workflow and github.ref
- cancel-in-progress is false
- unittest 8/8
- Chrome: N/A
- GHA 🐳 success; Hub cli-ok

## Non-goals
cancel-in-progress true; sync-whisper.yml

## Constraints
- Changing publish-docker.yml starts Docker Images
- Hub verify: --platform linux/amd64, /app/whisper/build/bin/whisper-cli
- Unittest from src/
- PyYAML: ${{ }} stays a string; false parses as False

## Active phase
propose

## Hypothesis
Per-ref concurrency with cancel-in-progress false → Docker Images still succeeds.

## Expected signal
YAML ok; GHA success; Hub cli-ok

## Tasks
- 1.1 pending
- 2.1 pending

## Last action
Wrote OpenSpec artifacts. Adapted from cancel-true to cancel-false per research.

## Next action
Validate, apply, unittest.
