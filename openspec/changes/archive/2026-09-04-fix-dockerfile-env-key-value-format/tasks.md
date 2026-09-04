## 1. ENV syntax

- [x] 1.1 Change `src/Dockerfile` `ENV PYTHONDONTWRITEBYTECODE 1` and `ENV PYTHONUNBUFFERED 1` to `ENV PYTHONDONTWRITEBYTECODE=1` and `ENV PYTHONUNBUFFERED=1` and verify `docker build --check -f src/Dockerfile src` reports no LegacyKeyValueFormat

## 2. Verify

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py -v` and verify exit 0 including holdout missing-file JSON 400
