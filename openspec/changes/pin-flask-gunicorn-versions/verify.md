# Verify

## Gates

- `cd src && python3 -m unittest test_app.py -v` — exit 0, 6/6
- Holdout missing-file: 400 JSON `{"error":"No file part"}` (unittest)
- `openspec validate pin-flask-gunicorn-versions --strict` — valid
- Chrome: N/A — no UI

## Image

- Tag: `whisperdock-local:pinned` (not `dublok/*`)
- `importlib.metadata.version('flask')` → `3.1.3`
- `importlib.metadata.version('gunicorn')` → `26.2.0`
- Whisper compile layers cached; pip layer rebuilt and collected `Flask==3.1.3` `gunicorn==26.2.0`

## Dockerfile

- `RUN pip install Flask==3.1.3 gunicorn==26.2.0`
- No unpinned `pip install Flask gunicorn`

## Independent check

Unittest (host Flask 3.1.2) plus image metadata (pinned 3.1.3/26.2.0) agree the API holdout still passes and the image is not floating latest.

## Remote

Pending origin push of `main` and GitHub Actions `publish-docker.yml`.
