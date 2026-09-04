## Context

No `concurrency:` today. Publish pushes Hub tags. sync-whisper may dispatch this workflow on a tag ref. See proposal.md.

Prior lesson guessed `cancel-in-progress: true`. Official deploy guidance is not to abort mutating jobs.

## Goals / Non-Goals

**Goals:**
- One in-flight publish per ref; newer runs wait.

**Non-Goals:**
- Cancelling stale publishes. Cross-workflow groups.

## Decisions

1. **Workflow-level `group: ${{ github.workflow }}-${{ github.ref }}`**
   - Official example. Isolates from sync-whisper. Tag dispatch does not block main.
2. **`cancel-in-progress: false`**
   - Hub push is mutating. Queue instead of kill.

## Risks / Trade-offs

- [A queued run waits up to 20m behind an in-flight job] → acceptable vs partial tags.
- [Changing the workflow republishes] → expected; wait for 🐳.

## Migration Plan

Insert after `permissions:`. Rollback: delete the block.

## Open Questions

None.
