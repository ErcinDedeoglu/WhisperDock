# Checkpoint

## Invariants

- Goal: bound POST `/transcribe` to 16 000 000 bytes with JSON 413; no temps; no ffmpeg.
- Acceptance: unittest exit 0; oversized → 413 JSON `error`; missing-file holdout 400 `"No file part"`; Chrome N/A.
- Non-goals: gunicorn/proxy limits, MIME allowlists, push.
- Constraints: do not push (`publish-docker.yml` on `**`). Size S.
- Decision: `MAX_CONTENT_LENGTH = 16 * 1000 * 1000` + `@app.errorhandler(413)` jsonify.

## Current State

- Phase: verify green; next sync then archive
- Hypothesis: config + 413 JSON handler → oversized POST never hits ffmpeg; leaked_tmp=[]
- Expected signal: 413 application/json; existing 5 tests green
- Tasks: 1.1–2.2 done
- Retry count: 0
- Confidence: high (unittest 6/6 + independent holdout)
- Verification: local pass; Chrome N/A; CI n/a no push
- Next action: OpenSpec sync delta into openspec/specs/transcribe-api

## Facts / Assumptions / Open questions

- Facts: Flask 3.1.2; MAX_CONTENT_LENGTH None; unittest 5/5; research URLs in proposal.md
- Assumptions: test_client returns 413 without connection reset
- Open questions: none

## Events

- 2026-09-04 propose: artifacts created; product code unchanged
- 2026-09-04 apply 1.1: MAX_CONTENT_LENGTH=16000000; python assert exit 0
- 2026-09-04 apply 1.2: 413 handler; test-client huge.wav status=413 type=application/json error set
- 2026-09-04 apply 2.1: unittest 6/6 OK including oversized 413 leaked_tmp=[]
- 2026-09-04 apply 2.2: holdout test_missing_file_returns_json_400 ok
- 2026-09-04 verify: unittest 6/6; independent 400+413; validate --strict ok; Chrome N/A
