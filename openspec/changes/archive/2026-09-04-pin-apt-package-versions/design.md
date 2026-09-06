## Context

Unpinned apt names. Proven versions from `dpkg-query` on `whisperdock-local:pinned`.

## Decisions

1. **Pin the five direct packages only** to those versions.
2. **Not multi-stage** this slice.

## Risks

- [bookworm removes a pin] → build fails; bump the pin.
