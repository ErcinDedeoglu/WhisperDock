## Context

Runtime is `FROM python:3.12-bookworm` after a multi-stage COPY of whisper-cli. That fat base is `buildpack-deps` and still contains `g++`. Apt pins are Debian 12 versions. See proposal.md.

## Goals / Non-Goals

**Goals:**
- Runtime base without a C++ compiler, still Debian 12 so existing apt pins apply.

**Non-Goals:**
- Changing builder base. Moving to Trixie or Alpine.

## Decisions

1. **Runtime `python:3.12-slim-bookworm`**
   - Official Hub tag; `FROM debian:bookworm-slim`; gcc/g++ purged after CPython build.
   - Rejected `python:3.12-slim`: currently `slim-trixie`, which would break `ffmpeg=7:5.1.9-0+deb12u1`.
   - Rejected Alpine: musl vs glibc `.so` files.

2. **Keep builder `python:3.12-bookworm`**
   - Compile still needs cmake/g++. Slim builder would reinstall them.

## Risks / Trade-offs

- [ffmpeg on slim pulls extra libs] → still smaller than bookworm+g++; measure size.
- [libgomp1 already pinned] → keep the same runtime apt RUN.

## Migration Plan

Change one FROM line. Rebuild `whisperdock-local:pinned`. Rollback: restore `python:3.12-bookworm`.

## Open Questions

None.
