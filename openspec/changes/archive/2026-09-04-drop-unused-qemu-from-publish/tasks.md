## 1. QEMU

- [x] 1.1 Remove the Setup QEMU step from `.github/workflows/publish-docker.yml`, then verify the file has no `docker/setup-qemu-action` and `platforms` is still `linux/amd64`

## 2. Holdout

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
