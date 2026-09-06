## Why

Workflows use mutable tags (`actions/checkout@v7`, `docker/*@v4`/`@v7`). A retagged action can run different code with Docker Hub credentials. GitHub says a full-length commit SHA is the only immutable action pin.

## What Changes

- Pin every `uses:` in `publish-docker.yml` and `sync-whisper.yml` to the current tag SHA with a same-line `# vN` comment for Dependabot.

## Capabilities

### New Capabilities

### Modified Capabilities

- `docker-image-publish`: publish workflow `uses:` MUST be 40-character SHAs with Node 24 major comments.

## Impact

- `.github/workflows/publish-docker.yml` and `sync-whisper.yml`. Changing the publish workflow starts Docker Images.

## Problem

Tags can be moved; production publish uses Docker Hub secrets.

## Non-goals

- Changing action majors. pinact. Organization policies.

## Hypothesis

If all `uses:` are `owner/repo@<40-hex> # vN`, then no `@v7`/`@v4` remains as the ref, unittest still exits 0, and Docker Images succeeds.

## Expected signal

- Each `uses:` after `@` is 40 hex chars.
- Comments `# v7` or `# v4` remain.
- GHA 🐳 success.

## Research

- https://docs.github.com/en/actions/reference/security/secure-use — pin to full-length commit SHA.
- Dependabot updates `uses: action@<commit> #<tag>` on the same line.

## Chosen / rejected

- Chosen: SHA + `# vN` from `gh api repos/.../commits/<tag>`.
- Rejected: leaving tags (mutable). pinact (extra tool).

## Rollback

Restore `@v7`/`@v4` refs.

## Acceptance

- unittest 8/8 including holdout missing-file JSON 400.
- Chrome: N/A — no UI.
