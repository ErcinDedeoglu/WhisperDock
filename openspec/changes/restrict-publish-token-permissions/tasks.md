## 1. Permissions

- [x] 1.1 Add top-level `permissions: contents: read` to `.github/workflows/publish-docker.yml` after `on:` and before `jobs:`, then verify the parsed mapping has `contents == "read"` and the file has no `contents: write`, `write-all`, `packages: write`, or `id-token: write`

## 2. Holdout

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
