# Lessons

## 2026-09-04 — return-json-for-transcribe-media-errors

- **Date:** 2026-09-04
- **Change:** return-json-for-transcribe-media-errors
- **Finding:** POST `/transcribe` with invalid media raised uncaught ffmpeg `CalledProcessError`, returning HTML 500 and leaking two `NamedTemporaryFile(delete=False)` paths.
- **Hypothesis:** Catch ffmpeg `CalledProcessError` / `FileNotFoundError` and unlink temps in `finally` → garbage POST is JSON with `error` and `leaked_tmp=[]`.
- **Action:** View-local try/except/finally in `transcribe_audio`; 400 for conversion failure, 500 for missing whisper-cli; stdlib unittest driving shipped Flask view.
- **Evidence:** Discover HTML 500 + 2 leaked temps. After: `GARBAGE_AUDIO status=400 type=application/json leaked_tmp=[] body='{"error":"Error in transcription"}'` on unittest (gate-1, gate-2) and gunicorn in `dublok/whisperdock:latest` with mounted `app.py` (two POSTs). Holdout missing-file still `{"error":"No file part"}`.
- **Outcome:** Hypothesis confirmed. Change archived. No push (`publish-docker.yml` would publish `dublok/whisperdock`).
- **Failure mode:** none this slice. First docker curl without waiting for gunicorn workers: connection reset (qemu/amd64 on arm64).
- **Confidence:** high for invalid-media JSON + cleanup; whisper success path not run here.
- **Applicability:** Any Flask JSON API that shells out with `check=True` and `NamedTemporaryFile(delete=False)` without `finally`.
- **Superseded lesson:** none
- **Pattern-Key:** json-api-uncaught-subprocess-leaks-delete-false-temps
- **Next experiment:** Rank remaining backlog — Python 3.8 EOL image base (reliability), or `MAX_CONTENT_LENGTH` (security). Do not push a branch while `on.push.branches: '**'` publishes production images.

## 2026-09-04 — upgrade-python-image-to-3-12

- **Date:** 2026-09-04
- **Change:** upgrade-python-image-to-3-12
- **Finding:** Published image ran CPython 3.8.20 (`FROM python:3.8`); 3.8 EOL 2024-10-07.
- **Hypothesis:** `FROM python:3.12-bookworm` → local image `sys.version_info[:2]==(3,12)` and unittest still exit 0.
- **Action:** Dockerfile + README 3.12; local tag `whisperdock-local:py312` (not `dublok/*`); unittest 5/5; synced `container-runtime`; archived.
- **Evidence:** `whisperdock-local:py312` python 3.12.14; Flask 3.1.3 / gunicorn 26.2.0; unittest OK; `openspec validate --specs` 2 passed. No push.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-upgrade-python-image-to-3-12`. Local commit f35e97d not pushed.
- **Failure mode:** none. Size-M wait-for-human in checkpoint was overridden by standing loop (local apply, no push).
- **Confidence:** high local 3.12; remote/CI unproven.
- **Applicability:** Docker Python bases still on 3.8; unpinned pip on EOL interpreters.
- **Superseded lesson:** none
- **Pattern-Key:** eol-cpython-docker-base-blocks-current-wheels
- **Next experiment:** Bound upload size via `MAX_CONTENT_LENGTH` in `src/app.py`. Still do not push.

## 2026-09-04 — bound-transcribe-upload-size

- **Date:** 2026-09-04
- **Change:** bound-transcribe-upload-size
- **Finding:** Flask 3.1.2 `MAX_CONTENT_LENGTH` was `None`; POST `/transcribe` saved unbounded bodies then ran ffmpeg.
- **Hypothesis:** `app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000` plus `@app.errorhandler(413)` JSON → oversized POST is 413 `application/json` with `error` and `leaked_tmp=[]` before ffmpeg.
- **Action:** Config + 413 `jsonify` handler in `src/app.py`; unittest oversized `huge.wav`; holdout missing-file 400. Synced `transcribe-api`; archived.
- **Evidence:** Unittest 6/6 exit 0. Independent: missing=400 `No file part`; oversized=413 JSON. `openspec validate --strict` ok. Chrome N/A. Commits 9f66f0f / 1aaa043 / 9f36ea7 not pushed.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-bound-transcribe-upload-size`. No push.
- **Failure mode:** none. Default 413 is HTML; handler is required for the JSON contract. Werkzeug dev server may connection-reset; test_client returns 413.
- **Confidence:** high local test-client; gunicorn/proxy limits unproven.
- **Applicability:** Flask JSON APIs with file uploads and `MAX_CONTENT_LENGTH` left at default `None`.
- **Superseded lesson:** none
- **Pattern-Key:** flask-default-unbounded-upload-html-413
- **Next experiment:** Restrict `.github/workflows/publish-docker.yml` so non-release pushes do not publish `dublok/whisperdock`. Until then, do not push.

## 2026-09-04 — restrict-docker-publish-to-main

- **Date:** 2026-09-04
- **Change:** restrict-docker-publish-to-main
- **Finding:** `publish-docker.yml` used `on.push.branches: '**'` and `push: true`, so any git push published `dublok/whisperdock`.
- **Hypothesis:** `on.push.branches: [main]` plus keep `workflow_dispatch` → YAML has no `'**'`; feature-branch pushes do not start the Hub job; `sync-whisper.yml` dispatch remains.
- **Action:** One-line trigger change. No job-level `if: main` (would block tag-branch dispatch). Synced `docker-image-publish`; archived.
- **Evidence:** Parse `branches==['main']`; unittest 6/6 holdout; `gh workflow run publish-docker.yml` still in sync-whisper; `openspec validate --specs --strict` 3/3. Chrome N/A. Commits 139c03b / 62820c8 / cc70661 not yet pushed.
- **Outcome:** Hypothesis confirmed locally. Archived `openspec/changes/archive/2026-09-04-restrict-docker-publish-to-main`. Origin push is now allowed on `main` (will publish once, as intended).
- **Failure mode:** none. Remote GHA not observed until first `main` push.
- **Confidence:** high local YAML; remote unproven.
- **Applicability:** GitHub Actions deploy workflows that use `'**'` with `push: true` to a production registry.
- **Superseded lesson:** standing “never push” constraint from `'**'` publish.
- **Pattern-Key:** gha-globstar-push-publishes-production-registry
- **Next experiment:** Discover next finding (gunicorn request size vs Flask 16 MB, or pin Flask/gunicorn, or first `main` push + CI evidence).

## 2026-09-04 — fix-readme-docker-build-context

- **Date:** 2026-09-04
- **Change:** fix-readme-docker-build-context
- **Finding:** README `docker build -t whisperdock .` from repo root; Dockerfile and `COPY whisper`/`app.py` live under `src/`. GHA uses `context: ./src`.
- **Hypothesis:** `docker build -t whisperdock src` matches GHA; README no longer contains `docker build -t whisperdock .`.
- **Action:** One-line README edit. `skip_specs`; design skipped. Archived.
- **Evidence:** README_OK grep; unittest 6/6 holdout; validate `--strict` with `skip_specs`. Commit b693dfd / e091fb6.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-fix-readme-docker-build-context`.
- **Failure mode:** none. Rejected `-f src/Dockerfile .` (COPY still repo-root).
- **Confidence:** high for docs path; did not run a full image rebuild this slice.
- **Applicability:** Dockerfiles whose COPY paths assume a subdirectory context documented as repo-root `docker build .`.
- **Superseded lesson:** none
- **Pattern-Key:** dockerfile-subdir-context-vs-readme-dot
- **Next experiment:** Remove unused `import socket` in `src/app.py` (polish) or pin Flask/gunicorn (reliability).

## 2026-09-04 — pin-flask-gunicorn-versions

- **Date:** 2026-09-04
- **Change:** pin-flask-gunicorn-versions
- **Finding:** `RUN pip install Flask gunicorn` unpinned; host Flask 3.1.2 / gunicorn 23.0.0 vs image Flask 3.1.3 / gunicorn 26.2.0.
- **Hypothesis:** `Flask==3.1.3 gunicorn==26.2.0` on that RUN → rebuild metadata matches; unittest still exit 0.
- **Action:** One-line Dockerfile pin. No requirements.txt. Synced `container-runtime`; archived.
- **Evidence:** Unittest 6/6. `whisperdock-local:pinned` and Hub `dublok/whisperdock:102b8d6` both 3.1.3 / 26.2.0. GHA run 33846918383 success. Chrome N/A.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-pin-flask-gunicorn-versions`. Pushed `main`.
- **Failure mode:** none. Hub tag is linux/amd64 only; local arm64 pull needs `--platform linux/amd64`.
- **Confidence:** high local + Hub metadata; whisper success path still not run.
- **Applicability:** Dockerfiles with unpinned `pip install` of runtime WSGI packages.
- **Superseded lesson:** none
- **Pattern-Key:** unpinned-pip-install-floats-rebuilds
- **Next experiment:** Unused `import socket` (polish) or GHA Node 20 deprecation on Docker Actions (reliability) or hashed pip lockfile (transitive pins).

## 2026-09-04 — upgrade-gha-actions-to-node24

- **Date:** 2026-09-04
- **Change:** upgrade-gha-actions-to-node24
- **Finding:** GHA run 33847304880 annotated Node 20 actions: checkout@v4, qemu@v3, buildx@v3, login@v3, build-push@v5.0.0. Node 20 removed 2026-09-23.
- **Hypothesis:** Bump to checkout@v7, qemu/buildx/login@v4, build-push@v7 → next publish-docker succeeds and annotation does not name those five.
- **Action:** Two workflow files. Rejected FORCE_JAVASCRIPT_ACTIONS_TO_NODE24. Synced `docker-image-publish`; archived.
- **Evidence:** Unittest 6/6. GHA 33847832202 success 2m57s. Node 20 annotation gone. New warning: Dockerfile LegacyKeyValueFormat ENV lines 5–6.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-upgrade-gha-actions-to-node24`.
- **Failure mode:** none. build-push v5→v7 did not break existing `with:` keys.
- **Confidence:** high for those five actions; Dockerfile ENV warning is a new finding.
- **Applicability:** Workflows still on checkout@v4 / docker actions @v3/@v5 after the Node 20 runner deprecation.
- **Superseded lesson:** none
- **Pattern-Key:** gha-node20-action-majors-emit-deprecation
- **Next experiment:** Dockerfile `ENV KEY value` → `ENV KEY=value` (LegacyKeyValueFormat) or unused `import socket`.

## 2026-09-04 — fix-dockerfile-env-key-value-format

- **Date:** 2026-09-04
- **Change:** fix-dockerfile-env-key-value-format
- **Finding:** `ENV PYTHONDONTWRITEBYTECODE 1` / `ENV PYTHONUNBUFFERED 1` triggered BuildKit LegacyKeyValueFormat (GHA 33848114073; local `--check` 2 warnings).
- **Hypothesis:** `ENV KEY=1` → `--check` 0 warnings; GHA annotation gone; unittest still 0.
- **Action:** Two-line `=` conversion. Rejected `# check=skip`. Synced `container-runtime`; archived.
- **Evidence:** Baseline `--check` 2 warnings. After: no warnings. Unittest 6/6. GHA 33848503848 success 2m24s with no LegacyKeyValueFormat annotation.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-fix-dockerfile-env-key-value-format`.
- **Failure mode:** none.
- **Confidence:** high local check + GHA.
- **Applicability:** Dockerfiles still using space-separated `ENV key value`.
- **Superseded lesson:** none
- **Pattern-Key:** dockerfile-legacy-env-space-separator
- **Next experiment:** unused `import socket` in `src/app.py` (polish).

## 2026-09-04 — remove-unused-socket-import

- **Date:** 2026-09-04
- **Change:** remove-unused-socket-import
- **Finding:** `src/app.py` imported `socket` and never used it (`ast` unused).
- **Hypothesis:** Delete the import → no `import socket`; unittest still 6/6.
- **Action:** One-line delete. `skip_specs`. Design skipped.
- **Evidence:** Unittest 6/6; `ast` unused `[]`; GHA 33849081771 success. Chrome N/A.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-remove-unused-socket-import`.
- **Failure mode:** none.
- **Confidence:** high.
- **Applicability:** Flask apps with leftover stdlib imports.
- **Superseded lesson:** none
- **Pattern-Key:** unused-stdlib-import-no-behavior
- **Next experiment:** hashed pip lockfile (transitive pins) or README leftover template prose.

## 2026-09-04 — remove-readme-template-prose

- **Date:** 2026-09-04
- **Change:** remove-readme-template-prose
- **Finding:** README still told readers to “Adjust the example response…” after JSON examples that already match `test_parse_transcription_readme_segments`.
- **Hypothesis:** Delete that paragraph → phrase gone; examples remain; unittest 6/6.
- **Action:** Docs delete. `skip_specs`. Design skipped.
- **Evidence:** README grep miss on Adjust; examples still present; unittest 6/6; GHA 33849637846 success.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-remove-readme-template-prose`.
- **Failure mode:** none.
- **Confidence:** high.
- **Applicability:** READMEs that leave scaffold “adjust this example” notes in published docs.
- **Superseded lesson:** none
- **Pattern-Key:** leftover-readme-author-notes-after-examples
- **Next experiment:** hashed pip lockfile (transitive pins).

## 2026-09-04 — pin-python-transitive-deps

- **Date:** 2026-09-04
- **Change:** pin-python-transitive-deps
- **Finding:** Direct Flask/gunicorn pins left transitives floating (Werkzeug, Jinja2, MarkupSafe, blinker, click, itsdangerous).
- **Hypothesis:** `pip freeze` file + `--no-deps -r` → image metadata matches freeze; unittest 6/6.
- **Action:** `src/requirements.txt` from `whisperdock-local:pinned` freeze; Dockerfile COPY then `--no-deps -r`. Rejected hashes (arch wheels).
- **Evidence:** Local and Hub `47d5723` metadata `3.1.3 26.2.0 3.1.8 3.1.6 3.0.3 1.9.0 8.5.0 2.2.0`. Unittest 6/6. GHA 33850394394 success.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-pin-python-transitive-deps`.
- **Failure mode:** none. MarkupSafe selected aarch64 locally and still 3.0.3 on amd64 Hub.
- **Confidence:** high.
- **Applicability:** Docker images that pin only top-level pip packages.
- **Superseded lesson:** none
- **Pattern-Key:** pip-direct-pins-leave-transitives-floating
- **Next experiment:** hash-checking with multi-arch `--hash` lines, or gunicorn request-size vs Flask 16 MB.

## 2026-09-04 — add-pip-require-hashes

- **Date:** 2026-09-04
- **Change:** add-pip-require-hashes
- **Finding:** Frozen `requirements.txt` had no hashes; pip trusted PyPI bytes.
- **Hypothesis:** `--hash` for py3-none-any plus MarkupSafe linux cp312 amd64/arm64, with `--require-hashes --only-binary :all: --no-deps` → local and GHA installs succeed; versions unchanged.
- **Action:** Hashes from PyPI JSON; Dockerfile flags. Rejected hashing all 89 MarkupSafe wheels.
- **Evidence:** Local rebuild hash-ok. Hub `6526193` 3.1.3/26.2.0/3.0.3. GHA 33851184564 success. Unittest 6/6.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-add-pip-require-hashes`.
- **Failure mode:** none this slice.
- **Confidence:** high.
- **Applicability:** Frozen pip files without `--hash` when deploying on more than one linux arch.
- **Superseded lesson:** none
- **Pattern-Key:** pip-freeze-without-hashes-trusts-pypi-bytes
- **Next experiment:** gunicorn request-size vs Flask 16 MB (needs evidence), or non-root USER in Dockerfile.

## 2026-09-04 — run-gunicorn-as-non-root

- **Date:** 2026-09-04
- **Change:** run-gunicorn-as-non-root
- **Finding:** Image ran gunicorn as root (`User=` empty; `id` uid=0).
- **Hypothesis:** `USER app` uid 10001 → inspect User=app; `id` not 0; unittest still 0.
- **Action:** groupadd/useradd 10001 without `-r` (SYS_UID_MAX 999 warned); USER app. No chown /app.
- **Evidence:** Local and Hub `f93ec9f` User=app uid=10001. Unittest 6/6. GHA 33851979087 success.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-run-gunicorn-as-non-root`.
- **Failure mode:** first try used `-r -u 10001` and Debian warned; dropped `-r`.
- **Confidence:** high for default user; whisper success path still not run as app.
- **Applicability:** Docker Flask/gunicorn images that omit USER.
- **Superseded lesson:** none
- **Pattern-Key:** gunicorn-as-root-when-dockerfile-omits-user
- **Next experiment:** gunicorn request-size vs Flask 16 MB (still needs evidence).

## 2026-09-04 — add-container-healthcheck

- **Date:** 2026-09-04
- **Change:** add-container-healthcheck
- **Finding:** Image Healthcheck=null; only POST `/transcribe`. Gunicorn 17 MB POST already JSON 413 in 10 ms (no body-size flag in gunicorn 26.2.0). Tiny WAV as uid 10001 → 200.
- **Hypothesis:** GET `/health` 200 JSON + urllib HEALTHCHECK → inspect Test includes `/health`; container healthy; unittest still 0.
- **Action:** `/health` `{"status":"ok"}`; HEALTHCHECK exec-form python urllib. No curl.
- **Evidence:** unittest 7/7. Local healthy at 7s ExitCode=0. Hub `a223c92` Healthcheck set. GHA 33852948788 success.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-add-container-healthcheck`.
- **Failure mode:** none. Gunicorn body-size was a non-finding (Flask 16 MB already applies).
- **Confidence:** high for liveness; whisper readiness not in HEALTHCHECK by design.
- **Applicability:** Flask/gunicorn images with no HEALTHCHECK and no GET liveness route.
- **Superseded lesson:** none
- **Pattern-Key:** docker-healthcheck-needs-2xx-liveness-route
- **Next experiment:** gunicorn `-w 4` memory vs whisper workers, or skip docs-only Docker rebuilds.
