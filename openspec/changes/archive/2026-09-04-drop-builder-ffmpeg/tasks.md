## 1. Builder apt

- [x] 1.1 Remove `ffmpeg` and `libsndfile1` from the builder apt RUN in `src/Dockerfile`, rebuild `whisperdock-local:pinned` (not `dublok/*`), and verify the builder install line has no `ffmpeg=` and `test -x` whisper-cli succeeds

## 2. Holdout

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
