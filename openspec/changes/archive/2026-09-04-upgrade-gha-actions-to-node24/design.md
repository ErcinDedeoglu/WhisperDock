## Context

See proposal.md. `publish-docker.yml` still uses Node 20 majors that GHA run 33847304880 annotated. `sync-whisper.yml` also uses `actions/checkout@v4`. Hosted `ubuntu-latest` already forces Node 24. Inputs on build-push (`context`, `file`, `push`, `platforms`, `tags`) are unchanged.

## Goals / Non-Goals

**Goals:**

- Replace the five annotated `uses:` tags (and the matching checkout in sync-whisper) with Node 24 majors.
- Keep publish triggers, Hub login secrets, and image tags unchanged.

**Non-Goals:**

- SHA pins, env-var workarounds, runner image changes.

## Decisions

1. **Major tags, not patch pins**
   - Why: current files already use `@v4`/`@v3` majors except `build-push-action@v5.0.0`. Latest Node 24 majors: checkout v7, docker qemu/buildx/login v4, build-push v7.
   - Rejected: keep `@v5.0.0` (node20). Rejected: SHA pins this slice.

2. **Jump build-push v5 → v7**
   - Why: v7.3.0 is `using: node24`. v6 added optional build summary; we do not set removed `DOCKER_BUILD_NO_SUMMARY`. Existing `with:` keys remain valid.
   - Rejected: stay on v5 and only set `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` (annotation remains).

3. **Also bump sync-whisper checkout**
   - Why: same Node 20 action, same root cause; one-line; avoids the next scheduled sync job repeating the warning.

## Risks / Trade-offs

- [build-push v7 changes default summary/export] → Holdout: job still pushes `dublok/whisperdock`; if push fails, revert YAML.
- [checkout v7 breaks `fetch-depth: 0` / `ref: main`] → Official checkout still documents those inputs; GHA failure → revert.
- [Docs-only later pushes republish Hub] → Existing main-branch publish behavior; not introduced here.

## Migration Plan

- Edit the two workflow files; unittest holdout; push `main`; wait for `publish-docker.yml`.
- Rollback: restore previous `uses:` tags.

## Open Questions

None.
