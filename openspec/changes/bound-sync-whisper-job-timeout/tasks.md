## 1. Timeout

- [x] 1.1 Set `timeout-minutes: 10` on `sync-by-tag` and `sync-latest-commit` in `.github/workflows/sync-whisper.yml`, then verify both parsed jobs have `timeout-minutes == 10`

## 2. Holdout

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
