# Verify

## Gates

- `cd src && python3 -m unittest test_app.py` — exit 0, 6/6
- Holdout missing-file: 400 JSON `{"error":"No file part"}`
- `ast` unused imports: `[]`
- `import socket` absent from `src/app.py`
- Chrome: N/A — no UI
- Specs: skip_specs — no delta

## Remote

Pending origin push and GHA.
