## Why

Overlapping `main` pushes can run two Docker publishes at once and race Hub tags. GitHub default allows concurrent workflow runs.

## What Changes

- Add workflow-level `concurrency` with `group: ${{ github.workflow }}-${{ github.ref }}` and `cancel-in-progress: false`.

## Capabilities

### New Capabilities

### Modified Capabilities

- `docker-image-publish`: publish runs for the same ref MUST be serialized.

## Impact

- `.github/workflows/publish-docker.yml`. Changing this file starts Docker Images.

## Problem

Two in-flight `docker/build-push-action` jobs can push `dublok/whisperdock:main` out of order or overlap.

## Non-goals

- Cancelling in-flight Hub pushes. Changing `sync-whisper.yml`. `queue: max`.

## Hypothesis

If concurrency is scoped per workflow+ref and does not cancel in-progress, a single main push still completes Docker Images successfully.

## Expected signal

- Parsed `concurrency.group` contains `github.workflow` and `github.ref`.
- `cancel-in-progress` is false.
- GHA 🐳 success; Hub tag cli-ok.

## Research

- https://docs.github.com/en/actions/using-jobs/using-concurrency — group `${{ github.workflow }}-${{ github.ref }}`; cancel-in-progress cancels the running run.
- Deploy pipelines should queue (`false`), not kill in-flight registry writes.

## Chosen / rejected

- Chosen: workflow-level group per workflow+ref, `cancel-in-progress: false` (serialize Hub pushes; do not abort a push).
- Rejected: `cancel-in-progress: true` (can leave a partial Hub tag). Group without `github.workflow` (would collide with other workflows). Group without ref (tag dispatch would block main).

## Rollback

Delete the `concurrency:` block.

## Acceptance

- unittest 8/8 including holdout missing-file JSON 400.
- Chrome: N/A — no UI.
