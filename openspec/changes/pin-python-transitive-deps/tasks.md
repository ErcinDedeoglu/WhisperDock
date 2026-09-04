## 1. Freeze file

- [x] 1.1 Add `src/requirements.txt` with the eight `==` pins from `whisperdock-local:pinned` freeze and verify the file lists Flask==3.1.3, gunicorn==26.2.0, Werkzeug==3.1.8, Jinja2==3.1.6, MarkupSafe==3.0.3, blinker==1.9.0, click==8.5.0, itsdangerous==2.2.0
- [x] 1.2 Change `src/Dockerfile` to `COPY requirements.txt /app/requirements.txt` before `COPY app.py`, replace inline pip with `RUN pip install --no-cache-dir --no-deps -r /app/requirements.txt`, and verify the Dockerfile has no inline `pip install Flask==`

## 2. Verify

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py -v` and verify exit 0 including holdout missing-file JSON 400
- [x] 2.2 Rebuild local tag `whisperdock-local:pinned` (not `dublok/*`) and verify `importlib.metadata` reports Flask 3.1.3 gunicorn 26.2.0 Werkzeug 3.1.8 Jinja2 3.1.6 MarkupSafe 3.0.3 blinker 1.9.0 click 8.5.0 itsdangerous 2.2.0
