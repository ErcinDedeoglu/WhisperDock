## 1. Runtime base

- [x] 1.1 Change runtime `FROM` in `src/Dockerfile` to `python:3.12-slim-bookworm`, rebuild `whisperdock-local:pinned` (not `dublok/*`), and verify `dpkg-query -W g++` fails, `test -x` whisper-cli succeeds, python reports `(3, 12)`, and image size is below 2.21 GB

## 2. Holdout

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
