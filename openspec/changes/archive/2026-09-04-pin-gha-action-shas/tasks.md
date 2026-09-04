## 1. Pin

- [x] 1.1 Pin all `uses:` in `publish-docker.yml` and `sync-whisper.yml` to the design.md SHAs with `# vN` comments, then verify each ref after `@` is 40 hex chars and Node 20 tags remain absent

## 2. Holdout

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
