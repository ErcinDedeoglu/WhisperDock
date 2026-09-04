## Why

`publish-docker.yml` runs on every `main` push. Docs-only commits still rebuild and push `dublok/whisperdock` (runs 33852390181 and 33853268414, ~2m45s each). The Docker context is `src/`; OpenSpec/lesson files cannot change the image. That wastes CI and publishes extra Hub tags.

## What Changes

- Add `on.push.paths` so automatic publish runs only when `src/**` or this workflow file changes.
- Keep `workflow_dispatch` unfiltered (sync-whisper still dispatches).

Not **BREAKING** for `src/` or dispatch.

## Capabilities

### New Capabilities

- (none)

### Modified Capabilities

- `docker-image-publish`: automatic push-on-git-push MUST also require a path in `src/**` or `.github/workflows/publish-docker.yml`.

## Impact

- `.github/workflows/publish-docker.yml` `on.push` only.

## Problem

Every main push, including `openspec/**` and `*.md`, publishes a new image.

## Non-goals

- Changing tags, secrets, or `sync-whisper.yml`.
- gunicorn `-w 4`.
- Required-check workarounds (this workflow is not a PR required check).

## Hypothesis

If `on.push.paths` is `src/**` and `.github/workflows/publish-docker.yml`, then a Python parse of the YAML shows those paths, `workflow_dispatch` remains, a src/workflow commit still starts Docker Images, and a later openspec-only commit does not.

## Expected signal

- YAML `on.push.paths` contains `src/**` and the workflow file.
- Apply push: Docker Images run exists.
- Archive/lesson push: no new Docker Images run for that SHA.

## Research

Official pattern: https://docs.github.com/en/actions/writing-workflows/choosing-when-your-workflow-runs/triggering-a-workflow#using-filters-to-target-specific-paths-for-pull-request-or-push-events (`paths` allowlist; cannot combine with `paths-ignore`; `workflow_dispatch` ignores path filters)
Why current code is worse: no `paths`; docs commits 33852390181 and 33853268414 rebuilt Hub
Chosen approach: `paths: ['src/**', '.github/workflows/publish-docker.yml']`
Rejected alternative: `paths-ignore: ['**/*.md']` (openspec yaml would still rebuild)
Proof plan: YAML parse; apply SHA has a run; archive SHA has none; Chrome: N/A — no UI

Supporting: https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#onpushpull_requestpull_request_targetpathspaths-ignore

## Chosen and rejected approaches

- **Chosen:** allowlist `src/**` + this workflow file.
- **Rejected:** denylist markdown only.
- **Rejected:** job-level `if` + dorny/paths-filter (extra action; not a required check).

## Rollback

Remove `paths`.

## Acceptance checks

- Python parse: `on.push.paths` includes `src/**` and `.github/workflows/publish-docker.yml`
- `workflow_dispatch` still in `on`
- unittest holdout still exits 0
- Chrome: N/A — no UI
