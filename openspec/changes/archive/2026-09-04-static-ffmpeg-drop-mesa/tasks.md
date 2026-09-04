## 1. Runtime ffmpeg

- [x] 1.1 Replace runtime Debian ffmpeg/libsndfile1 with `COPY --from=mwader/static-ffmpeg:9.0.1 /ffmpeg /usr/local/bin/ffmpeg` and apt `libgomp1` only, rebuild `whisperdock-local:pinned` (not `dublok/*`), and verify `dpkg-query -W libllvm15` fails, `command -v ffmpeg` succeeds, and image size is below 1.06 GB

## 2. Holdout

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
