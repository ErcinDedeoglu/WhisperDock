## 1. Path filter

- [x] 1.1 Add `on.push.paths` for `src/**` and `.github/workflows/publish-docker.yml` in `publish-docker.yml`, keep `workflow_dispatch`, and verify a Python parse of that YAML shows those paths and no `paths-ignore`

## 2. Verify

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
