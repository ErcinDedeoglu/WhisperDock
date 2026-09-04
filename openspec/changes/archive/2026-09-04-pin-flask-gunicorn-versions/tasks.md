## 1. Pin pip packages

- [x] 1.1 Change `src/Dockerfile` `RUN pip install Flask gunicorn` to `RUN pip install Flask==3.1.3 gunicorn==26.2.0` and verify the file contains both `==` pins and does not contain unpinned `pip install Flask gunicorn`

## 2. Verify

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py -v` and verify exit 0 including holdout missing-file JSON 400
- [x] 2.2 Rebuild local tag `whisperdock-local:pinned` (not `dublok/*`) from `src/` and verify `importlib.metadata.version` reports Flask 3.1.3 and gunicorn 26.2.0
