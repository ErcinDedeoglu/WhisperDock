## 1. Ignore file

- [x] 1.1 Add `src/.dockerignore` excluding `test_app.py` `__pycache__` `*.pyc`, verify a probe `COPY test_app.py` build fails, rebuild `whisperdock-local:pinned` (not `dublok/*`), and verify `test -x` whisper-cli succeeds

## 2. Holdout

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
