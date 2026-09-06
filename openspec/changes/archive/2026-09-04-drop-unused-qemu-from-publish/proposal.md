## Why

`publish-docker.yml` installs QEMU then builds only `linux/amd64` on `ubuntu-latest` (already amd64). QEMU is for emulating other architectures. The extra action is unused attack surface and extra job time.

## What Changes

- Remove the `docker/setup-qemu-action` step from `.github/workflows/publish-docker.yml`.
- Keep Buildx, login, and `platforms: linux/amd64`.

## Capabilities

### New Capabilities

### Modified Capabilities

- `docker-image-publish`: publish workflow MUST NOT use `setup-qemu-action`; Node 24 pin list no longer includes QEMU.

## Impact

- `.github/workflows/publish-docker.yml`. Changing this file starts Docker Images.

## Problem

QEMU binfmt is registered for a native amd64 build that never needs user-mode emulation.

## Non-goals

- Adding arm64. Removing Buildx. Changing `platforms`.

## Hypothesis

If the QEMU step is removed and `platforms` stays `linux/amd64`, Docker Images still succeeds on ubuntu-latest.

## Expected signal

- Workflow has no `setup-qemu-action`.
- `platforms` is still `linux/amd64`.
- GHA 🐳 success; Hub tag cli-ok.

## Research

- https://github.com/docker/setup-qemu-action — registers QEMU so later steps can run another architecture.
- https://depot.dev/blog/multi-platform-docker-images-in-github-actions — QEMU is for the Arm portion on Intel hosted runners.

## Chosen / rejected

- Chosen: delete the QEMU step; keep Buildx for `build-push-action`.
- Rejected: also dropping Buildx (still required). Adding arm64 (out of scope; johnvansickle was amd64-only).

## Rollback

Restore the SHA-pinned QEMU step before Buildx.

## Acceptance

- unittest 8/8 including holdout missing-file JSON 400.
- Chrome: N/A — no UI.
