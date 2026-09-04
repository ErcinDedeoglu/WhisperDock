## 1. Health route

- [x] 1.1 Add GET `/health` in `src/app.py` returning JSON `{"status": "ok"}` with 200, and a unittest that asserts status 200, JSON content type, and that body
- [x] 1.2 Add Dockerfile `HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3` exec-form `python -c` urllib GET `http://127.0.0.1:5000/health`, rebuild `whisperdock-local:pinned` (not `dublok/*`), verify inspect Healthcheck is set and a started container reaches `healthy`

## 2. Verify

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
