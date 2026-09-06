## Context

Neither job has `timeout-minutes`. Work is git ls-remote, checkout, clone, commit. See proposal.md.

Do not `gh workflow run` this workflow: a successful push mutates `src/whisper` and can dispatch Docker Images.

## Goals / Non-Goals

**Goals:**
- Cap hung sync jobs well below 360 minutes.

**Non-Goals:**
- Running the sync. Changing publish timeout.

## Decisions

1. **`timeout-minutes: 10` on both jobs**
   - Clone+commit, not cmake. 10 ≫ expected duration.
   - Same integer on both jobs.

## Risks / Trade-offs

- [Slow GitHub/git clone exceeds 10] → raise timeout, do not drop it.
- [Push of this file starts Docker Images] → it does not; path filter is `src/**` and publish-docker.yml only.

## Migration Plan

Add the key under each job. Rollback: delete both.

## Open Questions

None.
