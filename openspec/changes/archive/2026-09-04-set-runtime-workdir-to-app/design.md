## Context

`WORKDIR /app` then `WORKDIR /app/whisper` for make. Last WORKDIR wins at runtime. gunicorn `--chdir /app` loads the module; `pwd` is still `/app/whisper`. `-w 4` RSS was 96–103 MiB — not this slice.

## Goals / Non-Goals

**Goals:** `Config.WorkingDir=/app`.

**Non-Goals:** worker count; `--chdir` removal.

## Decisions

1. **`WORKDIR /app` after `COPY app.py`**
   - Why: official last-WORKDIR-is-runtime-cwd. Absolute paths in app.py stay valid.
2. **Keep gunicorn `--chdir /app`**
   - Why: out of size; still correct if WORKDIR is `/app`.

## Risks / Trade-offs

- [whisper-cli relative writes] → temps already `/tmp`; USER app cannot write `/app` anyway.
