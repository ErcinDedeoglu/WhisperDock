## Context

Job has `runs-on: ubuntu-latest` and no timeout. Recent GHA runs: ~2m20s–2m38s. See proposal.md.

## Goals / Non-Goals

**Goals:**
- Cap hung publish jobs well below 360 minutes without flaking healthy builds.

**Non-Goals:**
- sync-whisper.yml. Step timeouts.

## Decisions

1. **`timeout-minutes: 20` on `linux-build-and-push`**
   - Integer; GitHub default 360.
   - 20 ≈ 8× observed duration; cold cmake of whisper.cpp should still fit.
   - Rejected 5 (cache-miss risk) and 360 (no-op).

## Risks / Trade-offs

- [Cold build exceeds 20] → GHA fail; raise timeout, do not drop it.
- [Changing publish-docker.yml republishes] → expected; wait for 🐳.

## Migration Plan

Add the key under the job. Rollback: delete it.

## Open Questions

None.
