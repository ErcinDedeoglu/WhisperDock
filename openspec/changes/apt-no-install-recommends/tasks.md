## 1. Apt RUN

- [x] 1.1 Merge apt update/install/list cleanup into one `RUN` with `--no-install-recommends` in `src/Dockerfile`, rebuild `whisperdock-local:pinned` (not `dublok/*`), and verify `ffmpeg -version` works

## 2. Verify

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
