## Why

The publish job has no `timeout-minutes`. GitHub's default is 360. Observed successful runs finish in about 2.5 minutes. A hung `docker build` can consume six hours of runner time.

## What Changes

- Set `timeout-minutes: 20` on `linux-build-and-push` in `.github/workflows/publish-docker.yml`.

## Capabilities

### New Capabilities

### Modified Capabilities

- `docker-image-publish`: the publish job MUST declare a timeout below the 360-minute default.

## Impact

- `.github/workflows/publish-docker.yml`. Changing this file starts Docker Images.

## Problem

Uncapped jobs hide hung builds until the six-hour platform limit.

## Non-goals

- Changing `sync-whisper.yml`. Step-level timeouts. Raising the GitHub-hosted 360 cap.

## Hypothesis

If the job has `timeout-minutes == 20`, Docker Images still succeeds because recent runs complete in under 3 minutes.

## Expected signal

- Parsed job `timeout-minutes` is 20.
- GHA 🐳 success under 20 minutes; Hub tag cli-ok.

## Research

- https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idtimeout-minutes — default 360; positive integer.

## Chosen / rejected

- Chosen: job-level `20` (~8× observed 2.5 min; room for a cold whisper.cpp compile).
- Rejected: `360` (no-op). `5` (too tight for a cache miss). Step-only timeout (job can still hang across steps).

## Rollback

Delete `timeout-minutes`.

## Acceptance

- unittest 8/8 including holdout missing-file JSON 400.
- Chrome: N/A — no UI.
