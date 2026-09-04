## 1. Bump actions

- [x] 1.1 Update `.github/workflows/publish-docker.yml` `uses:` to `actions/checkout@v7`, `docker/setup-qemu-action@v4`, `docker/setup-buildx-action@v4`, `docker/login-action@v4`, `docker/build-push-action@v7` and verify the five Node 20 tags are absent
- [x] 1.2 Update `.github/workflows/sync-whisper.yml` `actions/checkout@v4` to `actions/checkout@v7` and verify no `checkout@v4` remains in `.github/workflows/`

## 2. Verify

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py -v` and verify exit 0 including holdout missing-file JSON 400
