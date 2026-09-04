## 1. Non-root user

- [x] 1.1 After `COPY app.py`, add `app` group/user uid/gid 10001 with `--no-log-init` and `USER app` in `src/Dockerfile`, then rebuild `whisperdock-local:pinned` (not `dublok/*`) and verify `docker inspect` User is `app` and `docker run --entrypoint id -u` is not 0

## 2. Verify

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py -v` and verify exit 0 including holdout missing-file JSON 400
