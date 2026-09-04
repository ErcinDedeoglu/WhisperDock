## 1. Pin digests

- [x] 1.1 Add `@sha256:` index digests to the three image refs in `src/Dockerfile`, rebuild `whisperdock-local:pinned` (not `dublok/*`), and verify the Dockerfile contains those three digest strings and `test -x` whisper-cli succeeds

## 2. Holdout

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
