## Context

See proposal.md. `publish-docker.yml` uses `on.push.branches: ['**']` and `docker/build-push-action` `push: true`. `sync-whisper.yml` publishes tag branches via `gh workflow run publish-docker.yml --ref`.

## Goals / Non-Goals

**Goals:**

- Stop automatic publish on non-main branches.
- Keep dispatch for main and whisper tag branches.

**Non-Goals:**

- Changing tags, platforms, secrets, or `sync-whisper.yml`.

## Decisions

1. **`on.push.branches: [main]` plus existing `workflow_dispatch`**
   - Why: official include-filter; `'**'` means all branches.
   - Rejected: job `if: github.ref == 'refs/heads/main'` — would skip dispatched tag-branch runs.

2. **Verify with stdlib YAML-ish parse, not a live Actions run**
   - Why: no push until this lands; live GHA is the thing we must not trigger.
   - Rejected: pushing a probe branch.

## Risks / Trade-offs

- [workflow_dispatch only works once the file is on default branch] → File already on `main`; this edit must land on `main` to take effect remotely.
- [Local YAML change does not protect origin until pushed] → After archive, pushing `main` is the intended first safe push.
- [Whisper tag branches lose automatic push-on-push] → Intentional; dispatch remains.

## Migration Plan

- Edit `on:` locally; parse-assert; no Docker Hub call.
- Rollback: restore `'**'`.
- First origin push only after this change is on `main`.

## Open Questions

None.
