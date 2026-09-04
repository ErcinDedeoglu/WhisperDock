## Why

`publish-docker.yml` has no `permissions:` block. Actions can still read `github.token` even when the YAML never passes `GITHUB_TOKEN`. The job only checks out the repo and pushes to Docker Hub with `DOCKER_USERNAME`/`DOCKER_TOKEN`; it does not need repository write.

## What Changes

- Add a top-level `permissions: contents: read` block to `.github/workflows/publish-docker.yml`.

## Capabilities

### New Capabilities

### Modified Capabilities

- `docker-image-publish`: the publish workflow MUST declare least-privilege `GITHUB_TOKEN` permissions (`contents: read`).

## Impact

- `.github/workflows/publish-docker.yml`. Changing this file starts Docker Images.

## Problem

Without an explicit `permissions:` key, `GITHUB_TOKEN` follows the repository default, which may still be read/write on older repos. A compromised third-party action in this workflow could write to the repository.

## Non-goals

- Changing `sync-whisper.yml` (already has `contents: write` and `actions: write` because it pushes and dispatches).
- Changing organization/repository default workflow permissions.
- Adding `packages`, `id-token`, or other write scopes.

## Hypothesis

If the publish workflow has top-level `permissions.contents == "read"` and no write scopes, then unspecified scopes are `none`, unittest still exits 0, and Docker Images still succeeds because Hub auth uses Docker secrets, not `GITHUB_TOKEN` write.

## Expected signal

- Parsed YAML has `permissions.contents == "read"`.
- No `contents: write`, `write-all`, `packages: write`, or `id-token: write` in the publish workflow.
- GHA 🐳 success; Hub tag for the new SHA is cli-ok.

## Research

- https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#permissions — unspecified permissions become `none`; `contents: read` is enough to list/checkout.
- https://docs.github.com/en/actions/tutorials/authenticate-with-github_token — actions can access `github.token` even when not passed; grant least required access.
- https://docs.github.com/en/actions/reference/security/secure-use — default `GITHUB_TOKEN` to contents read, raise only per job.

## Chosen / rejected

- Chosen: top-level `permissions: contents: read` (OpenSSF Token-Permissions pattern; Hub uses Docker secrets).
- Rejected: `permissions: {}` (checkout needs contents read). `read-all` (broader than needed). Job-level only (one job; top-level is the audit baseline). Touching `sync-whisper.yml` this slice.

## Rollback

Delete the `permissions:` block.

## Acceptance

- unittest 8/8 including holdout missing-file JSON 400.
- Chrome: N/A — no UI.
