## 1. Restrict trigger

- [x] 1.1 Change `on.push.branches` in `.github/workflows/publish-docker.yml` from `'**'` to `main`, keep `workflow_dispatch`, and verify a Python parse of that file reports `branches == ['main']`, `'**' not in branches`, and `'workflow_dispatch' in on`

## 2. Holdout

- [x] 2.1 Confirm `.github/workflows/sync-whisper.yml` still contains `gh workflow run publish-docker.yml` and that `publish-docker.yml` still has `push: true` on the build-push step
