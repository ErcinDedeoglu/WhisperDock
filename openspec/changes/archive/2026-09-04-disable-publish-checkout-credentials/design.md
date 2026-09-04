## Context

Publish checkout has no `with:` block. The job never runs `git push`. See proposal.md.

## Goals / Non-Goals

**Goals:**
- Stop persisting `GITHUB_TOKEN` after publish checkout.

**Non-Goals:**
- sync-whisper.yml (needs persisted credentials for `git push`).

## Decisions

1. **Add `with: persist-credentials: false` on the existing SHA-pinned checkout step**
   - Official checkout README: default true; false opts out.
   - Rejected changing sync-whisper this slice.

## Risks / Trade-offs

- [Changing publish-docker.yml republishes the image] → expected; wait for 🐳.
- [A later publish step that needs git auth would fail] → none today; rollback is one key.

## Migration Plan

Insert `with:` under checkout. Rollback: delete it.

## Open Questions

None.
