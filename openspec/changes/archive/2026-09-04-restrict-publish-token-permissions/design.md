## Context

`publish-docker.yml` has no `permissions:` key. The job checks out the repo, logs into Docker Hub with `secrets.DOCKER_USERNAME`/`secrets.DOCKER_TOKEN`, and `docker/build-push-action` pushes `dublok/whisperdock`. It does not push to GHCR, create releases, or dispatch other workflows. See proposal.md.

PyYAML parses the workflow key `on` as `True`. Read `permissions` via the parsed mapping, not by grepping `on:`.

## Goals / Non-Goals

**Goals:**
- Explicit least-privilege `GITHUB_TOKEN` for the publish workflow.

**Non-Goals:**
- Changing `sync-whisper.yml`. Repository default workflow permissions.

## Decisions

1. **Top-level `permissions: contents: read`**
   - GitHub: once any permission is set, unspecified scopes are `none` (`metadata` stays read).
   - Checkout needs `contents: read`. Hub auth does not use `GITHUB_TOKEN`.
   - Rejected `permissions: {}` (checkout may fail). Rejected `read-all` (extra scopes). Rejected job-level only (single job; top-level is the OpenSSF baseline).

2. **Do not add `packages` or `id-token`**
   - Image goes to Docker Hub, not GHCR. No OIDC.

## Risks / Trade-offs

- [Changing publish-docker.yml republishes the image] → expected; wait for 🐳 and Hub-verify.
- [If repo default is already contents-read, this is a no-op at runtime] → still worth YAML-explicit audit.

## Migration Plan

Insert the block after `on:` and before `jobs:`. Rollback: delete it.

## Open Questions

None.
