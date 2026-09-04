## Context

QEMU step sits before Buildx. `platforms: linux/amd64`. `ubuntu-latest` is amd64. See proposal.md.

Existing spec names QEMU in the Node 24 pin list; that requirement is updated in the same delta.

## Goals / Non-Goals

**Goals:**
- Remove unused QEMU action from native amd64 publish.

**Non-Goals:**
- Removing Buildx. Multi-arch.

## Decisions

1. **Delete only the Setup QEMU step**
   - Official: QEMU is for another architecture. Native amd64 does not need binfmt.
   - Keep Buildx: `build-push-action` expects it.

## Risks / Trade-offs

- [Changing publish-docker.yml republishes the image] → expected; wait for 🐳.
- [A later arm64 platform would need QEMU back] → rollback restores the SHA-pinned step.

## Migration Plan

Delete the QEMU step. Rollback: restore it before Buildx.

## Open Questions

None.
