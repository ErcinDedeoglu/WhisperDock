## Why

README API Usage already shows success and error JSON that match `parse_transcription` and the 500 error body. After those examples it still contains leftover author notes: “Adjust the example response to match the actual output format…” That tells readers the published examples may be wrong.

## What Changes

- Delete the leftover template paragraph and the extra `---` before Development.

Not **BREAKING**. Keep the JSON examples and the “handle both success and error” sentence.

## Capabilities

### New Capabilities

- (none — `skip_specs: true`; README author notes are not a service behavior contract)

### Modified Capabilities

- (none)

## Impact

- `README.md` API Usage only.

## Problem

Published docs include unpublished-author instructions.

## Non-goals

- Rewriting API examples, hashed pip lockfile, other README sections.

## Hypothesis

If the “Adjust the example response…” paragraph is removed, README no longer contains that sentence, the JSON examples remain, and unittest still exits 0.

## Expected signal

- README does not contain `Adjust the example response`.
- README still contains `"error": "Error in transcription"` and `"transcription"`.
- Unittest 6/6 including `test_parse_transcription_readme_segments`.

## Research

n/a — version-agnostic docs deletion (skip rule). Examples already match `src/test_app.py` README_STDOUT.

## Chosen and rejected approaches

- **Chosen:** Delete the leftover notes; keep examples.
- **Rejected:** Rewrite examples (they already match tests).

## Rollback

Restore the deleted paragraph.

## Acceptance checks

- README has no `Adjust the example response`
- README still has the success `transcription` example and error JSON
- `cd src && python3 -m unittest test_app.py -v` exits 0
- Chrome: N/A — no UI
