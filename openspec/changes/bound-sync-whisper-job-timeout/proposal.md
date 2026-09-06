## Why

Both jobs in `sync-whisper.yml` omit `timeout-minutes`. GitHub's default is 360. A hung `git clone` of whisper.cpp can consume six hours of runner time.

## What Changes

- Set `timeout-minutes: 10` on `sync-by-tag` and `sync-latest-commit`.

## Capabilities

### New Capabilities

- `whisper-sync`: how the whisper.cpp sync workflow is allowed to run.

### Modified Capabilities

## Impact

- `.github/workflows/sync-whisper.yml`. This file is not in `publish-docker.yml` path filters, so Docker Images does not start.

## Problem

Uncapped sync jobs hide hung clones until the six-hour platform limit.

## Non-goals

- Dispatching the sync workflow (would mutate `src/whisper` / `main`). Changing publish-docker.yml. persist-credentials.

## Hypothesis

If both jobs have `timeout-minutes == 10`, YAML parses those values and unittest still exits 0.

## Expected signal

- Parsed `sync-by-tag` and `sync-latest-commit` each have `timeout-minutes == 10`.
- Unittest 8/8. Docker Images does not start for this SHA.

## Research

- https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idtimeout-minutes — default 360; positive integer.

## Chosen / rejected

- Chosen: job-level `10` on both jobs (clone+commit, not a docker build).
- Rejected: `20` (publish budget; sync is cheaper). Dispatching the workflow this slice (mutates production tree).

## Rollback

Delete both `timeout-minutes` keys.

## Acceptance

- unittest 8/8 including holdout missing-file JSON 400.
- Chrome: N/A — no UI.
- Docker Images: N/A — path filter excludes this file.
