# Verify

## Gates

- `cd src && python3 -m unittest test_app.py -v` — exit 0, 6/6
- Holdout missing-file: 400 JSON `{"error":"No file part"}`
- Holdout publish triggers: `on.push.branches: [main]` and `workflow_dispatch` unchanged
- `openspec validate upgrade-gha-actions-to-node24 --strict` — valid
- Chrome: N/A — no UI

## Workflows

- publish-docker.yml: checkout@v7, qemu@v4, buildx@v4, login@v4, build-push@v7
- sync-whisper.yml: checkout@v7 (both jobs)
- No Node 20 tags remaining under `.github/workflows/`

## Remote

Pending origin push and GHA annotation check.
