## 1. Timeouts

- [x] 1.1 Add `timeout=240` to both `subprocess.run` calls in `src/app.py`, catch `subprocess.TimeoutExpired` as JSON 500 `Error in transcription`, and add a unittest that patches `subprocess.run` to raise `TimeoutExpired` and asserts 500 JSON with no leaked temps

## 2. Verify

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
