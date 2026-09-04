## Context

dependabot.yml already has docker `/src`. Workflows live under `.github/workflows`. See proposal.md.

## Goals / Non-Goals

**Goals:**
- Weekly github-actions updates without dropping docker.

**Non-Goals:**
- SHA-pinning actions. pip.

## Decisions

1. **`directory: /` for github-actions**
   - Official docs: do not set `/.github/workflows`.
2. **Keep docker `/src` unchanged**

## Risks / Trade-offs

- [Two Dependabot scans] → expected; docker entry stays.

## Migration Plan

Append one updates block. Rollback: delete that block.

## Open Questions

None.
