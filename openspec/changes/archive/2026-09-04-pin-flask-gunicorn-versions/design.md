## Context

See proposal.md for motivation. `src/Dockerfile` currently has `RUN pip install Flask gunicorn` after compiling whisper-cli. Build context is `./src`. `whisperdock-local:py312` already resolved Flask 3.1.3 and gunicorn 26.2.0. Changing only the pip `RUN` should reuse cached whisper compile layers.

## Goals / Non-Goals

**Goals:**

- Exact `==` pins for the two direct pip packages on the existing Dockerfile `RUN`.
- Prove rebuild metadata matches those pins without changing `/transcribe`.

**Non-Goals:**

- New requirements file, hash pins, apt pins, base-image digest pins.
- Gunicorn CMD or Flask app changes.

## Decisions

1. **Pin inline on the existing `RUN pip install`**
   - Why: pip documents `==` as pinning; hadolint DL3013 shows `pip install django==1.9` as the correct Dockerfile form. Two packages do not need a new file in the `src/` build context.
   - Rejected: `src/requirements.txt` + `COPY` — extra surface, same direct-dep pin strength.
   - Rejected: hash-checking / `pip freeze` of transitives — pip's next repeatability level; out of this slice.

2. **Pin Flask 3.1.3 and gunicorn 26.2.0**
   - Why: those versions are already installed in `whisperdock-local:py312` (Python 3.12.14) and the host unittest suite is Flask 3.1.x compatible.
   - Rejected: host-venv 3.1.2 / 23.0.0 — would downgrade the proven image stack.
   - Rejected: Flask 3.2.x — upgrade, not a pin of current.

3. **Keep `--timeout 300` gunicorn CMD unchanged**
   - Flask 3.1.x deploying-with-gunicorn docs still use `gunicorn -w 4 'module:app'`. No CMD change unless the pinned gunicorn refuses existing flags.

## Risks / Trade-offs

- [PyPI yank of 3.1.3 or 26.2.0] → Build fails closed; revert the pin line; do not float latest.
- [Transitive deps still float] → Accepted this slice; next experiment can add a hashed lockfile.
- [Cached pip layer ignores the pin] → The `RUN` instruction text changes, so Docker busts that layer; earlier whisper layers stay cached.

## Migration Plan

- Edit one Dockerfile line; rebuild local tag `whisperdock-local:pinned` (not `dublok/*`).
- Rollback: restore `RUN pip install Flask gunicorn`.
- Push to `main` after local verify (publish-docker is main-only).

## Open Questions

None.
