## 1. Workdir

- [x] 1.1 Add `WORKDIR /app` after `COPY app.py` in `src/Dockerfile`, rebuild `whisperdock-local:pinned` (not `dublok/*`), and verify `docker inspect` WorkingDir is `/app`

## 2. Verify

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
