## 1. Timeout

- [x] 1.1 Set `timeout-minutes: 20` on the `linux-build-and-push` job in `.github/workflows/publish-docker.yml`, then verify the parsed job mapping has `timeout-minutes == 20`

## 2. Holdout

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
