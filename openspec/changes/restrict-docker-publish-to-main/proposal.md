## Why

`.github/workflows/publish-docker.yml` triggers on `push.branches: '**'` and always `push: true` to `dublok/whisperdock`. Any branch push publishes a production Docker Hub tag. That is why this loop must not push until the trigger is narrowed.

## What Changes

- Restrict automatic Docker Hub publish to pushes to `main`.
- Keep `workflow_dispatch` so `sync-whisper.yml` can still `gh workflow run publish-docker.yml --ref <tag-branch|main>`.
- Do not add a job-level `if: github.ref == main` (that would block tag-branch dispatch).

Not **BREAKING** for `main` or dispatch. Feature-branch pushes will no longer publish.

## Capabilities

### New Capabilities
- `docker-image-publish`: when the Linux image workflow may push `dublok/whisperdock` tags.

### Modified Capabilities
- (none)

## Impact

- `.github/workflows/publish-docker.yml` `on:` block only.
- `sync-whisper.yml` unchanged.
- No Dockerfile, app, or image rebuild this slice.

## Problem

`'**'` matches all branch names, so accidental or loop pushes publish public images.

## Non-goals

- Changing tag scheme (`:SHA`, `:latest` on `v*`).
- Path filters, environments, or secret rotation.
- Editing `sync-whisper.yml`.

## Hypothesis

If `on.push.branches` is `[main]` and `workflow_dispatch` remains, then the workflow YAML no longer contains `'**'`, automatic publish is main-only, and dispatch for other refs stays possible.

## Expected signal

A stdlib parse of the YAML shows `on.push.branches == ['main']`, `'**'` absent from that list, and `workflow_dispatch` present.

## Research

Official pattern: https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow
Why current code is worse: `'**'` matches all branches (workflow-syntax cheat sheet)
Chosen approach: `workflow_dispatch` + `push.branches: [main]`
Rejected alternative: job `if: github.ref == 'refs/heads/main'` (blocks tag-branch dispatch)
Proof plan: Python parse of `publish-docker.yml` exit 0; Chrome: N/A — no UI

Supporting official sources:
- https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax (`'**'` = all branch names)
- https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows
- https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow (`gh workflow run --ref`)

## Chosen and rejected approaches

- **Chosen:** narrow `on.push.branches` to `main`; keep dispatch.
- **Rejected:** `branches-ignore` of every feature pattern (incomplete).
- **Rejected:** job-level main-only `if`.

## Rollback

Restore `branches: ['**']`. No image rollback.

## Acceptance checks

- Python parse: `on.push.branches == ['main']` and no `'**'` in that list
- `workflow_dispatch` still in `on`
- Chrome: N/A — no UI
