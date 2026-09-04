## Context

Tags in both workflows. SHAs from `gh api repos/<repo>/commits/<tag>`. See proposal.md.

## Goals / Non-Goals

**Goals:**
- Immutable `uses:` refs; Dependabot can still bump via `# vN`.

**Non-Goals:**
- Changing majors. pinact.

## Decisions

1. **Pins (resolved 2026-09-04)**
   - `actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1` # v7
   - `docker/setup-qemu-action@1f40c72289eff860ee54a304f1438e3cff362e0a` # v4
   - `docker/setup-buildx-action@37fe631027851001ddb9b187196cc803df7f5f0e` # v4
   - `docker/login-action@dbcb813823bdd20940b903addbd779551569679f` # v4
   - `docker/build-push-action@53b7df96c91f9c12dcc8a07bcb9ccacbed38856a` # v7
2. **Same-line `# vN` comments**
   - Required for Dependabot SHA updates, not decoration.

## Risks / Trade-offs

- [publish-docker.yml change republishes the image] → expected; wait for 🐳.

## Migration Plan

Replace `uses:` lines. Rollback: restore tags.

## Open Questions

None.
