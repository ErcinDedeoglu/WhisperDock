## Context

See proposal.md. Frozen versions already exist. GHA is linux/amd64; local Docker is linux/arm64. MarkupSafe 3.0.3 publishes distinct cp312 manylinux wheels.

## Goals / Non-Goals

**Goals:**

- Hash-check every frozen package on both Docker platforms we use.

**Non-Goals:**

- pip-tools; hashing Windows/macOS wheels; `--no-binary`.

## Decisions

1. **`--require-hashes --only-binary :all: --no-deps`**
   - Why: official secure-installs pairing of hash mode + no sdists; freeze already lists every dep.
2. **MarkupSafe: two linux cp312 manylinux hashes; others: py3-none-any wheel**
   - Why: pip ORs multiple hashes; those two wheels are what local and GHA actually fetch.
   - Rejected: all 89 MarkupSafe files.

## Risks / Trade-offs

- [PyPI republish same version different hash] → install fails closed; regenerate hashes.
- [New platform wheel needed] → add that `--hash` line.

## Migration Plan

- Rewrite requirements.txt hashes from PyPI JSON; add flags to Dockerfile RUN; rebuild local tag; push.
- Rollback: drop hashes and `--require-hashes`.

## Open Questions

None.
