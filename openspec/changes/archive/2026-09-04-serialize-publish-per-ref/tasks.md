## 1. Concurrency

- [x] 1.1 Add top-level `concurrency` with `group: ${{ github.workflow }}-${{ github.ref }}` and `cancel-in-progress: false` to `.github/workflows/publish-docker.yml`, then verify the parsed mapping

## 2. Holdout

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
