## Context

See proposal.md. CMD already binds 5000. Temps go to `/tmp`. whisper-cli and app.py are root-owned 755/644, readable/executable by others.

## Goals / Non-Goals

**Goals:**

- Default container user is non-root with a stable UID.

**Non-Goals:**

- chown of /app, capability drops, read-only rootfs.

## Decisions

1. **`groupadd -g 10001 app` and `useradd --no-log-init -u 10001 -g app app` then `USER app`**
   - Why: Docker USER recipe plus explicit UID/GID. `--no-log-init` avoids the faillog sparse-file bug. 10001 avoids colliding with typical 1000 host users. Not `-r`: Debian SYS_UID_MAX is 999, so `-r -u 10001` warns.

2. **No chown of /app**
   - Why: runtime only needs read/execute of app.py and whisper-cli and write to `/tmp`.

## Risks / Trade-offs

- [whisper-cli not world-executable] → `test -x` already; rebuild would fail `id` plus gunicorn start.
- [need write under /app] → none today; NamedTemporaryFile uses `/tmp`.

## Migration Plan

- Add user/USER after COPY app.py; rebuild local tag; push.
- Rollback: remove those three instructions.

## Open Questions

None.
