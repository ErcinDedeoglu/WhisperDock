# Verify

- Date: 2026-09-04
- Gate: YAML parse `on.push.branches == ['main']`, no `'**'`, `workflow_dispatch` present, `push: true` remains
- Holdout: `cd src && python3 -m unittest test_app.py -v` exit 0 (6 tests)
- Independent: `sync-whisper.yml` still `gh workflow run publish-docker.yml`
- Chrome: N/A — no UI
- OpenSpec: `openspec validate restrict-docker-publish-to-main --strict` exit 0
- CI/CD: n/a — no origin push until this is on `main` and archived
- Environment: n/a — no preview
