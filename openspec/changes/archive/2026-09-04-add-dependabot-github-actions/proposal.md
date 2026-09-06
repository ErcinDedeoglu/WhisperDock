## Why

Workflows pin `actions/checkout@v7` and docker/* actions by major tag. Dependabot only watches Docker. Those action majors can drift without PRs.

## What Changes

- Add a second `updates` entry: `package-ecosystem: github-actions`, `directory: /`, weekly.

## Capabilities

### New Capabilities

### Modified Capabilities

- `docker-image-publish`: Dependabot SHALL also watch GitHub Actions in `/`.

## Impact

- `.github/dependabot.yml` only. Docker Images should not run.

## Problem

No github-actions ecosystem in Dependabot.

## Non-goals

- Pinning actions to SHAs. pip ecosystem. Changing workflows.

## Hypothesis

If Dependabot includes github-actions at `/`, then the YAML contains `package-ecosystem: github-actions` and `directory: /`, unittest still exits 0, and `🐳 Docker Images` does not start.

## Expected signal

- Those keys present.
- Unittest 8/8.
- No Docker Images run for this SHA.

## Research

- https://docs.github.com/code-security/dependabot/dependabot-version-updates/configuration-options-for-the-dependabot.yml-file — github-actions uses `directory: "/"` (not `/.github/workflows`).

## Chosen / rejected

- Chosen: second updates entry, weekly, directory `/`.
- Rejected: replacing the docker entry. Pinning action SHAs this slice.

## Rollback

Remove the github-actions updates block.

## Acceptance

- unittest 8/8 including holdout missing-file JSON 400.
- Chrome: N/A — no UI.
