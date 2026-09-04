## Context

requirements.txt lives in `src/` with `--hash=sha256`. Dependabot already has docker `/src` and github-actions `/`. See proposal.md.

## Goals / Non-Goals

**Goals:**
- Weekly pip updates for `/src` without dropping docker or actions.

**Non-Goals:**
- Regenerating hashes in this slice.

## Decisions

1. **`package-ecosystem: pip` + `directory: /src`**
   - Same folder as requirements.txt. Two ecosystems may share `/src`.
2. **Keep hashes as-is**
   - Dependabot updates hashes on bump PRs.

## Risks / Trade-offs

- [Historical hash-format bugs] → current updater preserves `--hash=` lines; first scan is the gate.

## Migration Plan

Append one updates block. Rollback: delete it.

## Open Questions

None.
