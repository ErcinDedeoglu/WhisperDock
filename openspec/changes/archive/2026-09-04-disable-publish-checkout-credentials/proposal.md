## Why

`actions/checkout` defaults `persist-credentials` to true, so `GITHUB_TOKEN` is configured for later git commands. The publish job only builds and pushes to Docker Hub; it never `git push`. Third-party docker actions after checkout can read the persisted token.

## What Changes

- Set `persist-credentials: false` on the checkout step in `.github/workflows/publish-docker.yml`.

## Capabilities

### New Capabilities

### Modified Capabilities

- `docker-image-publish`: publish checkout MUST disable persisted git credentials.

## Impact

- `.github/workflows/publish-docker.yml`. Changing this file starts Docker Images.

## Problem

Default checkout leaves the job token available to subsequent steps, including third-party docker actions that do not need git auth.

## Non-goals

- Changing `sync-whisper.yml` (both jobs `git push` using persisted checkout credentials).
- Rewriting sync-whisper to pass `GH_TOKEN` on the remote URL.

## Hypothesis

If publish checkout has `persist-credentials: false`, Docker Images still succeeds because Hub auth uses Docker secrets, not git credentials.

## Expected signal

- Checkout step YAML has `persist-credentials: false`.
- GHA 🐳 success; Hub tag for the new SHA is cli-ok.

## Research

- https://github.com/actions/checkout — persist-credentials default true; set false to opt out.
- https://github.com/zizmorcore/zizmor/blob/main/crates/zizmor/src/audit/artipacked.rs — artipacked: token in git config after checkout.

## Chosen / rejected

- Chosen: `persist-credentials: false` on publish checkout only.
- Rejected: same flag on sync-whisper this slice (would break `git push` unless remotes are rewritten).

## Rollback

Remove the `with: persist-credentials: false` block.

## Acceptance

- unittest 8/8 including holdout missing-file JSON 400.
- Chrome: N/A — no UI.
