## 1. Multi-stage

- [x] 1.1 Split `src/Dockerfile` into builder + runtime as in design.md, rebuild `whisperdock-local:pinned` (not `dublok/*`), and verify `dpkg-query -W cmake` fails, `test -x` whisper-cli succeeds, and image size is below 2.34 GB

## 2. Verify

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
