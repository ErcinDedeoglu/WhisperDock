## Context

See proposal.md. `src/Dockerfile` lines 5–6 use legacy `ENV key value`. `docker build --check` reports two LegacyKeyValueFormat warnings. Values stay `1`.

## Goals / Non-Goals

**Goals:**

- Convert the two ENV lines to `key=value` so BuildKit stops warning.

**Non-Goals:**

- Other Dockerfile lints, combining ENV, changing the variable names or values.

## Decisions

1. **Keep two ENV instructions, add `=`**
   - Why: official check documents `ENV key=value` as the replacement for `ENV key value`. Same keys/values; smallest diff.
   - Rejected: one combined ENV (unrelated). Rejected: `# check=skip`.

2. **Verify with `docker build --check`, not a full whisper rebuild**
   - Why: the check is the failing gate. Full image rebuild is for GHA after push. Local holdout is unittest plus `--check`.

## Risks / Trade-offs

- [Equals form changes parsing of value] → Values are `1` with no spaces; official examples use `ENV MY_CAT=fluffy`.
- [GHA still warns from cache] → Instruction text change busts the Dockerfile parse; annotations are from the file, not cache.

## Migration Plan

- Edit two lines; run `--check` and unittest; push `main`; wait for GHA.
- Rollback: restore space-separated ENV.

## Open Questions

None.
