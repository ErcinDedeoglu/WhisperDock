# Checkpoint

## Invariants

- **Goal:** Bump GHA JavaScript actions to Node 24 majors so publish-docker no longer annotates Node 20.
- **Acceptance:** YAML uses checkout@v7, qemu@v4, buildx@v4, login@v4, build-push@v7; no old Node 20 tags; unittest 6/6; GHA success without those five names in Node 20 annotation; Chrome N/A.
- **Non-goals:** SHA pins, FORCE_/ALLOW_ env workarounds, trigger/tag/platform changes, unused socket, pip hashes.
- **Constraints:** Work on `main`. Push allowed. Do not force-push. Do not touch secrets.
- **Decisions:** Major tags; build-push v5→v7; also bump sync-whisper checkout.

## Current State

- **Phase:** propose complete; next apply
- **Hypothesis:** Node 24 majors → GHA success and annotation does not list those five actions
- **Expected signal:** YAML pins; GHA green without Node 20 names
- **Rollback:** revert the two workflow files
- **Tasks:** 1.1 pending; 1.2 pending; 2.1 pending
- **Retry count:** 0
- **Confidence:** medium until GHA
- **Next action:** apply 1.1

## Facts

- GHA 33847304880 annotated the five Node 20 tags
- Latest releases all `using: node24`
- Unittest 6/6 green before this slice

## Assumptions

- build-push v7 accepts existing `with:` keys
- checkout v7 still supports `fetch-depth` and `ref`

## Open questions

- None

## Events

- 2026-09-04 selected GHA Node 20 deprecation over unused socket (reliability > polish)
- 2026-09-04 research: GitHub changelog 2025-09-19; latest action.yml node24
