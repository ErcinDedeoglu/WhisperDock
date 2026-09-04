## Context

See proposal.md. Proven freeze from `whisperdock-local:pinned`. Build context is `./src`. GHA is linux/amd64; local Docker is linux/arm64.

## Goals / Non-Goals

**Goals:**

- Freeze all eight runtime pip packages in `src/requirements.txt`.
- Install with `--no-deps` so nothing extra is resolved.

**Non-Goals:**

- Hash-checking, pip-tools, apt pins.

## Decisions

1. **Freeze without hashes**
   - Why: pip documents `==` pins of transitives as the first repeatable level and notes it works across architectures. MarkupSafe 3.0.3 ships distinct manylinux wheels per arch; a single-arch hash set would fail the other build.
   - Rejected: `--hash` this slice.

2. **`--no-deps -r requirements.txt`**
   - Why: freeze already lists every needed package; `--no-deps` is the documented extra insurance.

3. **COPY requirements.txt before app.py**
   - Why: pip layer should not bust when only `app.py` changes.

## Risks / Trade-offs

- [Freeze omits a hidden dep] → `--no-deps` would fail import; unittest + gunicorn CMD still import Flask.
- [PyPI yank] → build fails closed; revert to inline pins.

## Migration Plan

- Add freeze file; switch Dockerfile RUN; rebuild `whisperdock-local:pinned` pip layer (whisper cache stays).
- Rollback: restore inline pip install; delete requirements.txt.

## Open Questions

None.
