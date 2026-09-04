# Verify

## Gates

- Baseline: `docker build --check` — 2 LegacyKeyValueFormat warnings (lines 5–6)
- After: `docker build --check -f src/Dockerfile src` — Check complete, no warnings found
- `cd src && python3 -m unittest test_app.py` — exit 0, 6/6
- Holdout missing-file: 400 JSON `{"error":"No file part"}`
- Chrome: N/A — no UI

## Remote

Pending origin push and GHA annotation check.
