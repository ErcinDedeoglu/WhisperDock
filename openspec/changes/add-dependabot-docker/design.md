## Context

`src/Dockerfile` pins python and static-ffmpeg by digest. No Dependabot file exists. See proposal.md.

## Goals / Non-Goals

**Goals:**
- Weekly Docker ecosystem updates for `/src`.

**Non-Goals:**
- Actions or pip. Changing image pins now.

## Decisions

1. **`directory: /src`**
   - GitHub: directory is where the Dockerfile lives, not repo root.
2. **`schedule.interval: weekly`**
   - Official docker example.

## Risks / Trade-offs

- [No PR in this slice] → config presence is the gate; first bump is later.

## Migration Plan

Add the file. Rollback: delete it.

## Open Questions

None.
