# Checkpoint

## Invariants

- **Goal:** Pin Flask and gunicorn in `src/Dockerfile` to exact versions already proven in `whisperdock-local:py312`.
- **Acceptance:** Dockerfile has `Flask==3.1.3` and `gunicorn==26.2.0`; unittest exit 0; local image metadata matches; holdout missing-file 400 JSON; Chrome N/A.
- **Non-goals:** requirements.txt, hashes, apt pins, base digest, Flask 3.2, unused socket, gunicorn CMD, `/transcribe` behavior.
- **Constraints:** Do not tag or push `dublok/*` locally. Origin push to `main` is allowed after local verify. No secrets. No force-push. No merge of a feature branch (work is already on `main`).
- **Decisions:** Inline `==` pins on existing `RUN pip install`. Versions Flask 3.1.3 / gunicorn 26.2.0. No new files.

## Current State

- **Phase:** verify local green; next commit+push then CI
- **Hypothesis:** confirmed locally
- **Expected signal:** pins in Dockerfile; image versions 3.1.3 / 26.2.0; holdout 400 JSON
- **Rollback:** restore `RUN pip install Flask gunicorn`
- **Tasks:** 1.1 done; 2.1 done; 2.2 done
- **Retry count:** 0
- **Confidence:** high local; remote GHA unproven
- **Next action:** commit, push `main`, wait for `publish-docker.yml`

## Facts

- Host Flask 3.1.2 / gunicorn 23.0.0; `whisperdock-local:pinned` Flask 3.1.3 / gunicorn 26.2.0
- Unittest 6/6 exit 0 from `src/`
- Whisper layers cached; pip layer downloaded Flask-3.1.3 and gunicorn-26.2.0
- `publish-docker.yml` `on.push.branches: [main]`
- Chrome N/A

## Assumptions

- Flask 3.1.3 and gunicorn 26.2.0 remain on PyPI
- Transitive pip deps may still float

## Open questions

- None

## Events

- 2026-09-04 preflight: no active change; selected unpinned pip as highest-ranked reliability finding
- 2026-09-04 research: pip repeatable-installs, hadolint DL3013, Flask 3.1 gunicorn docs
- 2026-09-04 proposed `pin-flask-gunicorn-versions`
- 2026-09-04 apply 1.1: Dockerfile pinned
- 2026-09-04 apply 2.1: unittest 6/6
- 2026-09-04 apply 2.2: `whisperdock-local:pinned` Flask 3.1.3 gunicorn 26.2.0
