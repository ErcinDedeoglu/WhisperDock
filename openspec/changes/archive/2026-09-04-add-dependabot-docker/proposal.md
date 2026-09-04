## Why

Base images are pinned by digest. Without Dependabot, those pins never move. There is no `.github/dependabot.yml`. README already documents `/health`.

## What Changes

- Add `.github/dependabot.yml` with `package-ecosystem: docker` and `directory: /src`, weekly.

## Capabilities

### New Capabilities

### Modified Capabilities

- `docker-image-publish`: Dependabot SHALL watch the service Dockerfile in `/src`.

## Impact

- `.github/dependabot.yml` only. Does not match `src/**`, so Docker Images should not run.

## Problem

Pinned digests have no automated bump path.

## Non-goals

- github-actions or pip ecosystems. Changing Dockerfile.

## Hypothesis

If Dependabot docker scans `/src` weekly, then the YAML contains `package-ecosystem: docker` and `directory: /src`, unittest still exits 0, and `🐳 Docker Images` does not start.

## Expected signal

- File exists with those two keys.
- Unittest 8/8.
- No Docker Images run for the docs/config commit.

## Research

- https://docs.github.com/code-security/dependabot/dependabot-version-updates/configuration-options-for-the-dependabot.yml-file — docker ecosystem; directory is the Dockerfile folder; weekly example.
- https://docs.docker.com/build/building/best-practices/#pin-base-image-versions — Dependabot `package-ecosystem: "docker"`.

## Chosen / rejected

- Chosen: docker `/src` weekly only.
- Rejected: directory `/` (Dockerfile is under src/). github-actions this slice.

## Rollback

Delete `.github/dependabot.yml`.

## Acceptance

- unittest 8/8 including holdout missing-file JSON 400.
- Chrome: N/A — no UI.
