## 1. Hashes

- [x] 1.1 Rewrite `src/requirements.txt` with sha256 hashes (py3-none-any wheels; MarkupSafe cp312 manylinux aarch64 and x86_64) and verify every package line contains `--hash=sha256:`
- [x] 1.2 Add `--require-hashes --only-binary :all:` to the Dockerfile pip install and verify those flags are present

## 2. Verify

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py -v` and verify exit 0 including holdout missing-file JSON 400
- [x] 2.2 Rebuild `whisperdock-local:pinned` (not `dublok/*`) and verify `importlib.metadata` still reports Flask 3.1.3 gunicorn 26.2.0 Werkzeug 3.1.8 Jinja2 3.1.6 MarkupSafe 3.0.3 blinker 1.9.0 click 8.5.0 itsdangerous 2.2.0
