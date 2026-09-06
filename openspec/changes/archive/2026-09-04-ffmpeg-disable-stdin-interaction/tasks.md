## 1. ffmpeg argv

- [x] 1.1 Add `-nostdin` to the ffmpeg argument list in `src/app.py` (before `-i`) and verify the list contains `-nostdin`

## 2. Verify

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
