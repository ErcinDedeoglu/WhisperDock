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

## 2026-09-04 — skip-docs-only-docker-publish

- **Date:** 2026-09-04
- **Change:** skip-docs-only-docker-publish
- **Finding:** Docs-only main pushes rebuilt Hub (33852390181, 33853268414, ~2m45s) though Docker context is `src/`.
- **Hypothesis:** `on.push.paths` src/** + workflow file → apply SHA still runs GHA; later openspec-only SHA does not.
- **Action:** path allowlist; keep workflow_dispatch. Not paths-ignore.
- **Evidence:** YAML parse OK. Apply `3a2d365` run 33853785983 success. Archive SHA: no Docker Images run.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-skip-docs-only-docker-publish`.
- **Failure mode:** none this slice.
- **Confidence:** high for push path filter; tag pushes still ignore paths (GitHub design).
- **Applicability:** expensive on-push image workflows whose build context is a subdirectory.
- **Superseded lesson:** none
- **Pattern-Key:** gha-path-filter-skip-docs-only-image-publish
- **Next experiment:** gunicorn `-w 4` memory vs whisper workers (needs RSS evidence).

## 2026-09-04 — set-runtime-workdir-to-app

- **Date:** 2026-09-04
- **Change:** set-runtime-workdir-to-app
- **Finding:** Last WORKDIR was `/app/whisper`; inspect WorkingDir=/app/whisper. gunicorn `-w 4` idle 96 MiB / 4 concurrent 103 MiB (non-finding).
- **Hypothesis:** `WORKDIR /app` after COPY app.py → inspect WorkingDir=/app; unittest still 0.
- **Action:** one WORKDIR /app. Kept gunicorn `--chdir /app`.
- **Evidence:** Local and Hub `f68a186` WorkingDir=/app. Unittest 7/7. GHA 33854493819 success.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-set-runtime-workdir-to-app`.
- **Failure mode:** none.
- **Confidence:** high for cwd; `--chdir` still redundant.
- **Applicability:** Dockerfiles that WORKDIR into a build subtree and never restore the app root.
- **Superseded lesson:** none
- **Pattern-Key:** dockerfile-last-workdir-is-runtime-cwd
- **Next experiment:** subprocess timeout vs gunicorn 300s, or ffmpeg `-nostdin`.

## 2026-09-04 — ffmpeg-disable-stdin-interaction

- **Date:** 2026-09-04
- **Change:** ffmpeg-disable-stdin-interaction
- **Finding:** ffmpeg `-y -i file` with no `-nostdin`; stdin interaction on by default.
- **Hypothesis:** `-nostdin` before `-i` → unittest 0; argv contains `-nostdin`; no `Press [q] to stop`.
- **Action:** add `-nostdin` to ffmpeg argv.
- **Evidence:** unittest 7/7; q-prompt gone; Hub `1499632` source True. GHA 33854914811 success.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-ffmpeg-disable-stdin-interaction`.
- **Failure mode:** none.
- **Confidence:** high for argv; hang not reproduced under gunicorn.
- **Applicability:** ffmpeg subprocess from a WSGI worker with a file `-i`.
- **Superseded lesson:** none
- **Pattern-Key:** ffmpeg-stdin-interaction-hangs-noninteractive-workers
- **Next experiment:** subprocess `timeout=` vs gunicorn `--timeout 300`.

## 2026-09-04 — bound-subprocess-timeout

- **Date:** 2026-09-04
- **Change:** bound-subprocess-timeout
- **Finding:** ffmpeg/whisper `subprocess.run` had no timeout; gunicorn `--timeout 300` would kill the worker.
- **Hypothesis:** `timeout=240` + `TimeoutExpired` JSON 500 → unittest 8/8 including timeout path; holdout 400.
- **Action:** SUBPROCESS_TIMEOUT=240 on both runs; catch TimeoutExpired as 500.
- **Evidence:** unittest 8/8. Hub `c1d6bfc` SUBPROCESS_TIMEOUT=240. GHA 33855403261 success.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-bound-subprocess-timeout`.
- **Failure mode:** none.
- **Confidence:** high for mocked timeout; live 240s hang not run.
- **Applicability:** Flask+gunicorn apps that spawn unbounded ffmpeg/whisper children.
- **Superseded lesson:** none
- **Pattern-Key:** subprocess-timeout-before-gunicorn-worker-kill
- **Next experiment:** README still says WAV 16kHz-only though ffmpeg converts any ffmpeg-readable input.

## 2026-09-04 — fix-readme-wav-only-claim

- **Date:** 2026-09-04
- **Change:** fix-readme-wav-only-claim
- **Finding:** README required WAV 16 kHz; ffmpeg already converts with `-ar 16000 -ac 1`.
- **Hypothesis:** replace that sentence → old phrase gone; unittest still 0; no Docker Images run.
- **Action:** one README sentence; skip_specs.
- **Evidence:** unittest 8/8. `296d22e` had no Docker Images run.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-fix-readme-wav-only-claim`.
- **Failure mode:** none.
- **Confidence:** high for docs; curl example still uses `.wav`.
- **Applicability:** READMEs that document a client constraint the server already lifts.
- **Superseded lesson:** none
- **Pattern-Key:** readme-client-format-vs-server-ffmpeg-convert
- **Next experiment:** document GET `/health` in README, or apt `--no-install-recommends`.

## 2026-09-04 — document-health-endpoint

- **Date:** 2026-09-04
- **Change:** document-health-endpoint
- **Finding:** GET `/health` existed; README API Usage omitted it.
- **Hypothesis:** document curl + `{"status":"ok"}` → README has `/health`; unittest 0; no Docker Images run.
- **Action:** README snippet after the 16 MB sentence; skip_specs.
- **Evidence:** unittest 8/8. `b4f1788` had no Docker Images run.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-document-health-endpoint`.
- **Failure mode:** none.
- **Confidence:** high for docs.
- **Applicability:** APIs that add liveness routes without updating the README.
- **Superseded lesson:** none
- **Pattern-Key:** health-route-undocumented-in-readme
- **Next experiment:** apt `--no-install-recommends` (combine with apt-lists cleanup in one RUN).

## 2026-09-04 — apt-no-install-recommends

- **Date:** 2026-09-04
- **Change:** apt-no-install-recommends
- **Finding:** apt install lacked `--no-install-recommends`; lists removed in a later RUN.
- **Hypothesis:** one RUN with the flag + `rm lists` → ffmpeg still works; unittest 0.
- **Action:** merged apt RUNs; sorted packages.
- **Evidence:** local and Hub `2b054f2` ffmpeg 5.1.9. Unittest 8/8. GHA 33856283056 success.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-apt-no-install-recommends`.
- **Failure mode:** none; ffmpeg still pulled many codecs as dependencies.
- **Confidence:** high for Dockerfile shape; image size delta not measured.
- **Applicability:** Debian Dockerfiles that split apt install and list cleanup.
- **Superseded lesson:** none
- **Pattern-Key:** apt-install-recommends-and-lists-in-separate-layer
- **Next experiment:** drop build-essential after compile (multi-stage), or pin apt package versions.

## 2026-09-04 — pin-apt-package-versions

- **Date:** 2026-09-04
- **Change:** pin-apt-package-versions
- **Finding:** apt packages unpinned after --no-install-recommends.
- **Hypothesis:** pin five proven versions → dpkg-query match; unittest 0.
- **Action:** pin build-essential cmake ffmpeg git libsndfile1.
- **Evidence:** local and Hub `a12d507` versions match. Unittest 8/8. GHA 33856751873 success.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-pin-apt-package-versions`.
- **Failure mode:** none.
- **Confidence:** high for those five; transitives still unpinned.
- **Applicability:** Debian Dockerfiles that name packages without `=version`.
- **Superseded lesson:** none
- **Pattern-Key:** apt-get-unpinned-direct-packages
- **Next experiment:** multi-stage runtime without build-essential (M; ldd whisper-cli first).

## 2026-09-04 — multi-stage-drop-build-tools

- **Date:** 2026-09-04
- **Change:** multi-stage-drop-build-tools
- **Finding:** 2.34 GB image shipped cmake, compilers, and whisper.cpp sources. ldd needs four .so + libgomp1.
- **Hypothesis:** runtime COPY whisper-cli+libs+model → cmake absent; cli works; size < 2.34 GB.
- **Action:** builder + runtime bookworm; COPY five binaries/libs + model; runtime apt ffmpeg/libsndfile1/libgomp1.
- **Evidence:** local 2.21 GB; cmake_st=1; cli-ok. Hub `0b47593` same. Unittest 8/8. GHA 33857454093 success.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-multi-stage-drop-build-tools`.
- **Failure mode:** python:3.12-bookworm still ships g++/git; `which g++` is not a valid absence check.
- **Confidence:** high for cmake/source drop; modest size win.
- **Applicability:** images that compile C++ then ship the compiler.
- **Superseded lesson:** none
- **Pattern-Key:** multi-stage-copy-cli-not-compiler
- **Next experiment:** python:3.12-slim runtime to drop base g++ (need ffmpeg on slim).

## 2026-09-04 — slim-runtime-drop-compiler

- **Date:** 2026-09-04
- **Change:** slim-runtime-drop-compiler
- **Finding:** python:3.12-bookworm runtime still shipped g++ 4:12.2.0-3 via buildpack-deps.
- **Hypothesis:** runtime FROM python:3.12-slim-bookworm → g++ absent; cli works; size < 2.21 GB; unittest 0.
- **Action:** runtime FROM python:3.12-slim-bookworm; builder stays bookworm.
- **Evidence:** local 1.06 GB; g_st=1; cli-ok; `(3, 12)`. Hub `0bdb3d7` same. Unittest 8/8. GHA 33858243431 success.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-slim-runtime-drop-compiler`.
- **Failure mode:** `python:3.12-slim` currently tracks Trixie and would break Debian 12 apt pins.
- **Confidence:** high.
- **Applicability:** multi-stage Python images whose fat bookworm runtime still contains compilers.
- **Superseded lesson:** none
- **Pattern-Key:** slim-bookworm-runtime-not-floating-slim
- **Next experiment:** ffmpeg still pulls mesa/llvm on slim (~1 GB); try a static ffmpeg or drop libavdevice.

## 2026-09-04 — static-ffmpeg-drop-mesa

- **Date:** 2026-09-04
- **Change:** static-ffmpeg-drop-mesa
- **Finding:** Debian ffmpeg on slim pulled libllvm15 (107 MB) and Mesa via libavdevice59. Image 1.06 GB.
- **Hypothesis:** COPY mwader/static-ffmpeg:9.0.1 + apt libgomp1 only → no libllvm15; ffmpeg works; size < 1.06 GB.
- **Action:** Runtime COPY `--from=mwader/static-ffmpeg:9.0.1 /ffmpeg /usr/local/bin/ffmpeg`; drop debian ffmpeg/libsndfile1.
- **Evidence:** local 673 MB; llvm_st=1; ffmpeg 9.0.1; conv-ok; cli-ok. Hub `0d6126d` same. Unittest 8/8. GHA 33858813688 success.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-static-ffmpeg-drop-mesa`.
- **Failure mode:** johnvansickle is amd64-only; would break GHA arm64.
- **Confidence:** high.
- **Applicability:** Debian images that apt-install ffmpeg only to resample audio.
- **Superseded lesson:** none
- **Pattern-Key:** static-ffmpeg-copy-from-not-apt
- **Next experiment:** drop unused debian ffmpeg from builder (build-time only; no runtime size).

## 2026-09-04 — drop-builder-ffmpeg

- **Date:** 2026-09-04
- **Change:** drop-builder-ffmpeg
- **Finding:** Builder still apt-installed Debian ffmpeg and libsndfile1 though whisper-cli compiles with WHISPER_COMMON_FFMPEG off.
- **Hypothesis:** three-package builder apt → whisper-cli still builds; no builder ffmpeg=.
- **Action:** Drop ffmpeg and libsndfile1 from builder apt; keep git.
- **Evidence:** dockerfile-ok; cli-ok; ffmpeg 9.0.1. Hub `301d0be` same. Unittest 8/8. GHA 33859305448 success.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-drop-builder-ffmpeg`.
- **Failure mode:** none. cmake still finds git; ggml commit stays unknown without .git.
- **Confidence:** high.
- **Applicability:** multi-stage images that leftover runtime packages in the compile stage.
- **Superseded lesson:** none
- **Pattern-Key:** builder-apt-only-compile-deps
- **Next experiment:** pin FROM digests for python and static-ffmpeg (reproducibility).

## 2026-09-04 — pin-base-image-digests

- **Date:** 2026-09-04
- **Change:** pin-base-image-digests
- **Finding:** python bookworm, slim-bookworm, and static-ffmpeg were floating tags.
- **Hypothesis:** tag@sha256 index pins → Dockerfile has three digests; image builds multi-arch.
- **Action:** Pin OCI index digests from `docker buildx imagetools inspect`.
- **Evidence:** pins-ok; cli-ok. Hub `c43d09b` cli-ok ffmpeg python 3.12. Unittest 8/8. GHA 33859792599 success.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-pin-base-image-digests`.
- **Failure mode:** pinning a platform digest (not the index) would break GHA amd64+arm64.
- **Confidence:** high.
- **Applicability:** multi-arch Dockerfiles that still use mutable tags.
- **Superseded lesson:** none
- **Pattern-Key:** pin-index-digest-keep-tag
- **Next experiment:** add a `.dockerignore` under `src/` if the build context still ships tests/docs.

## 2026-09-04 — add-src-dockerignore

- **Date:** 2026-09-04
- **Change:** add-src-dockerignore
- **Finding:** `docker build` context is `src/` with no root `.dockerignore`; nested `whisper/.dockerignore` is unused. Context included `test_app.py` and `__pycache__`.
- **Hypothesis:** `src/.dockerignore` listing those paths → probe COPY test_app.py fails; whisper-cli still builds.
- **Action:** Add ignore file with `test_app.py`, `__pycache__`, `*.pyc`. skip_specs.
- **Evidence:** probe_st=1 CopyIgnoredFile. cli-ok. Hub `6e219ff` cli-ok no-test. Unittest 8/8. GHA 33860343232 success.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-add-src-dockerignore`.
- **Failure mode:** ignoring whisper/tests or examples would break cmake add_subdirectory.
- **Confidence:** high for ignore of unittest; tiny context win (~28 KB).
- **Applicability:** subdirectory Docker contexts whose nested .dockerignore is not the context root.
- **Superseded lesson:** none
- **Pattern-Key:** dockerignore-at-context-root
- **Next experiment:** Dependabot docker ecosystem for digest bumps, or document HEALTHCHECK in README if missing.

## 2026-09-04 — add-dependabot-docker

- **Date:** 2026-09-04
- **Change:** add-dependabot-docker
- **Finding:** Digest-pinned Dockerfile had no Dependabot config. README already documents `/health`.
- **Hypothesis:** docker `/src` weekly YAML → keys present; Docker Images does not start.
- **Action:** `.github/dependabot.yml` version 2, docker, `/src`, weekly.
- **Evidence:** dependabot-ok. Unittest 8/8. No 🐳 run for `fa71bd8`. Dependabot Updates 33860769501 success.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-add-dependabot-docker`.
- **Failure mode:** `directory: /` would miss `src/Dockerfile`.
- **Confidence:** high for config; first image-bump PR not observed.
- **Applicability:** digest-pinned Dockerfiles without Dependabot.
- **Superseded lesson:** none
- **Pattern-Key:** dependabot-docker-directory-is-dockerfile-dir
- **Next experiment:** Dependabot github-actions ecosystem for workflow action pins.

## 2026-09-04 — add-dependabot-github-actions

- **Date:** 2026-09-04
- **Change:** add-dependabot-github-actions
- **Finding:** Dependabot watched docker `/src` only; workflows still use unpinned major tags.
- **Hypothesis:** github-actions `/` weekly → keys present; Docker Images does not start.
- **Action:** Second updates entry; keep docker `/src`.
- **Evidence:** dependabot-ok. Unittest 8/8. No 🐳 for `beb2836`. Dependabot Updates 33860964301 success (`github_actions in /.`).
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-add-dependabot-github-actions`.
- **Failure mode:** `directory: /.github/workflows` is wrong per GitHub docs.
- **Confidence:** high for config.
- **Applicability:** repos with Dependabot docker but not actions.
- **Superseded lesson:** none
- **Pattern-Key:** dependabot-github-actions-directory-slash
- **Next experiment:** Dependabot pip for hashed requirements.txt, or SHA-pin GHA actions.

## 2026-09-04 — add-dependabot-pip

- **Date:** 2026-09-04
- **Change:** add-dependabot-pip
- **Finding:** Hashed `src/requirements.txt` had no Dependabot pip ecosystem.
- **Hypothesis:** pip `/src` weekly → keys present; Docker Images does not start.
- **Action:** Third updates entry; keep docker `/src` and github-actions `/`.
- **Evidence:** dependabot-ok. Unittest 8/8. No 🐳 for `dc711ff`. Dependabot Updates 33861245411 success (`pip in /src`).
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-add-dependabot-pip`.
- **Failure mode:** `directory: /` would miss `src/requirements.txt`.
- **Confidence:** high for config; hash-rewriting on bump PRs unobserved.
- **Applicability:** hashed requirements.txt not covered by Dependabot.
- **Superseded lesson:** none
- **Pattern-Key:** dependabot-pip-directory-is-requirements-dir
- **Next experiment:** SHA-pin GitHub Actions (`uses: org/action@<sha> # vN`).

## 2026-09-04 — pin-gha-action-shas

- **Date:** 2026-09-04
- **Change:** pin-gha-action-shas
- **Finding:** Workflows used mutable `@v7`/`@v4` tags with Docker Hub secrets.
- **Hypothesis:** SHA + `# vN` → 40-hex uses; Docker Images succeeds.
- **Action:** Pin 7 uses lines from `gh api repos/.../commits/<tag>`.
- **Evidence:** pins-ok 7. Unittest 8/8. GHA 33861558847 success. Hub `3d1d460` cli-ok.
- **Outcome:** Hypothesis confirmed. Archived `openspec/changes/archive/2026-09-04-pin-gha-action-shas`.
- **Failure mode:** omitting `# vN` would break Dependabot SHA bumps.
- **Confidence:** high.
- **Applicability:** workflows that still reference actions by mutable tags.
- **Superseded lesson:** none
- **Pattern-Key:** gha-uses-full-sha-plus-version-comment
- **Next experiment:** least-privilege `permissions:` on publish-docker.yml (`contents: read`).
