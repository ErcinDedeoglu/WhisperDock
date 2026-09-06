# Checkpoint

- Goal: HEALTHCHECK + GET `/health` 200 JSON.
- Non-goals: whisper readiness; curl; gunicorn body limit.
- Decision: urllib HEALTHCHECK; `{"status":"ok"}`.
- Evidence: Healthcheck=null; 17 MB gunicorn POST 413 in 10 ms; tiny WAV as app 200.
- Next: validate, apply 1.1.
