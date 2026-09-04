## Why

GET `/health` returns 200 `{"status":"ok"}` and the image HEALTHCHECK probes it. README API Usage never mentions `/health`.

## What Changes

- Add a short GET `/health` curl example and JSON body under API Usage.

Not **BREAKING**.

## Capabilities

### New Capabilities

- (none — `skip_specs: true`; README copy is not a service behavior contract)

### Modified Capabilities

- (none)

## Impact

- `README.md` API Usage only.

## Problem

Operators have no documented liveness URL.

## Non-goals

- apt `--no-install-recommends`; changing `/health` behavior.

## Hypothesis

If README documents GET `/health` → `{"status":"ok"}`, then `/health` appears in README, unittest still exits 0, and Docker Images does not run.

## Expected signal

- README contains `GET` `/health` and `"status": "ok"`.

## Research

Official pattern: https://flask.palletsprojects.com/en/stable/quickstart/#http-methods (GET routes)
Why current code is worse: `/health` exists in app.py and HEALTHCHECK; README omits it
Chosen approach: one curl + JSON snippet after `/transcribe`
Rejected alternative: only mention HEALTHCHECK in Dockerfile comments
Proof plan: grep README `/health`; unittest exit 0; Chrome: N/A — no UI

Supporting: transcribe-api spec GET health 200 `{"status":"ok"}`

## Chosen and rejected approaches

- **Chosen:** document GET `/health` next to `/transcribe`.
- **Rejected:** rewrite Getting Started.

## Rollback

Delete the health paragraph.

## Acceptance checks

- README contains `/health` and `"status": "ok"`
- unittest holdout missing-file 400 JSON
- Chrome: N/A — no UI
