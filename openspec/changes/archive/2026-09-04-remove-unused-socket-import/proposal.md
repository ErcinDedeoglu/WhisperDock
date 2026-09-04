## Why

`src/app.py` imports `socket` and never uses it. `ast` reports it unused. Dead import is polish; no remaining High correctness/reliability findings.

## What Changes

- Remove `import socket` from `src/app.py`.

Not **BREAKING**. `/transcribe` behavior unchanged.

## Capabilities

### New Capabilities

- (none — `skip_specs: true`; unused import is not a service behavior contract)

### Modified Capabilities

- (none)

## Impact

- `src/app.py` first import line only.

## Problem

Unused stdlib import.

## Non-goals

- Other import reorder, Flask/gunicorn changes, hashed lockfile.

## Hypothesis

If `import socket` is removed, `ast` no longer reports `socket` unused, the file has no `import socket`, and unittest still exits 0 including missing-file JSON 400.

## Expected signal

- `src/app.py` has no `import socket`.
- Unittest 6/6 exit 0.
- Holdout: missing-file POST still 400 JSON `{"error":"No file part"}`.

## Research

n/a — version-agnostic unused-import removal (skip rule).

## Chosen and rejected approaches

- **Chosen:** Delete the unused import.
- **Rejected:** Keep it for a future hostname helper (no such code exists).

## Rollback

Restore `import socket`.

## Acceptance checks

- `cd src && python3 -m unittest test_app.py -v` exits 0
- `src/app.py` does not contain `import socket`
- Holdout missing-file still 400 JSON
- Chrome: N/A — no UI
