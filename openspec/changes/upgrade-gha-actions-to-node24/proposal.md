## Why

GitHub Actions run 33847304880 annotated: Node.js 20 actions are deprecated and forced onto Node 24 (`actions/checkout@v4`, `docker/build-push-action@v5.0.0`, `docker/login-action@v3`, `docker/setup-buildx-action@v3`, `docker/setup-qemu-action@v3`). Node 20 is removed from runners on 2026-09-23. Workflows still pin Node-20 action majors.

## What Changes

- Bump JavaScript actions in `publish-docker.yml` and `sync-whisper.yml` to current majors whose `action.yml` declares `using: node24`.
- Require the Docker image workflow's JS actions to be Node 24 versions.

Not **BREAKING** for `/transcribe` or Hub tags. CI still publishes on `main` + `workflow_dispatch`.

## Capabilities

### New Capabilities

- (none)

### Modified Capabilities

- `docker-image-publish`: JavaScript actions in the Docker image workflow MUST use Node 24 runtimes (not Node 20 majors).

## Impact

- `.github/workflows/publish-docker.yml` `uses:` tags.
- `.github/workflows/sync-whisper.yml` `actions/checkout@v4` → `@v7`.
- No Dockerfile, app, or unittest changes.

## Problem

Node-20 action majors emit deprecation annotations and will fail after Node 20 removal (2026-09-23).

## Non-goals

- Pinning actions to commit SHAs.
- `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` or `ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION`.
- Changing publish triggers, platforms, or Hub tags.
- Unused `import socket` or hashed pip lockfile.

## Hypothesis

If workflows use `actions/checkout@v7`, `docker/setup-qemu-action@v4`, `docker/setup-buildx-action@v4`, `docker/login-action@v4`, and `docker/build-push-action@v7`, then the next `publish-docker.yml` run succeeds and its Node 20 deprecation annotation does not list those actions; unittest holdout still 400 JSON.

## Expected signal

- YAML has no `checkout@v4`, `setup-qemu-action@v3`, `setup-buildx-action@v3`, `login-action@v3`, or `build-push-action@v5`.
- GHA job success; annotation absent or not naming those five actions.
- Holdout: missing-file POST still 400 JSON `{"error":"No file part"}`.

## Research

Official pattern: https://github.blog/changelog/2025-09-19-deprecation-of-node-20-on-github-actions-runners/ (users: update workflows to Node 24 action versions; Node 20 removed 2026-09-23)
Why current code is worse: GHA annotation lists five Node-20 `uses:` tags still in the YAML
Chosen approach: bump to latest majors whose `action.yml` is `using: node24` (`checkout@v7.0.1`, `build-push-action@v7.3.0`, `login-action@v4.6.0`, `setup-buildx-action@v4.3.0`, `setup-qemu-action@v4.3.0`)
Rejected alternative: `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` (still warns; does not change `runs.using`)
Proof plan: unittest exit 0; GHA run success without those Node 20 names; Chrome: N/A — no UI

Supporting official sources:
- https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#example-using-versioned-actions
- https://docs.docker.com/build/ci/github-actions/configure-builder/ (`setup-buildx-action@v4`, `checkout@v6`)
- Latest releases: checkout v7.0.1, build-push-action v7.3.0, login-action v4.6.0, setup-buildx-action v4.3.0, setup-qemu-action v4.3.0 — all `using: node24`

## Chosen and rejected approaches

- **Chosen:** Major-tag bumps to Node 24 releases, same style as current `@vN` (except replacing pinned `v5.0.0` with `@v7`).
- **Rejected:** Env-var force/opt-out (workaround; official user action is to update versions).
- **Rejected:** SHA pinning this slice (new supply-chain surface).

## Rollback

Revert the two workflow files.

## Acceptance checks

- `cd src && python3 -m unittest test_app.py -v` exits 0
- Workflows contain the Node 24 majors and not the listed Node 20 tags
- Next `publish-docker.yml` run succeeds; Node 20 annotation does not name those five actions
- Holdout: POST `/transcribe` no file part still 400 JSON `{"error": "No file part"}`
- Chrome: N/A — no UI
