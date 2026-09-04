## 1. Image base

- [x] 1.1 Change `src/Dockerfile` `FROM python:3.8` to `FROM python:3.12-bookworm` and verify `rg -n '^FROM ' src/Dockerfile` prints `python:3.12-bookworm`
- [x] 1.2 Update README development prerequisite from Python 3.8 to Python 3.12 and verify `rg -n 'Python 3\.' README.md` no longer lists 3.8

## 2. Verify runtime

- [x] 2.1 Rebuild a local image (do not tag `dublok/whisperdock`) from `src/Dockerfile` and verify `docker run --rm --entrypoint python <local-tag> -c "import sys; assert sys.version_info[:2] == (3, 12)"` exits 0
- [x] 2.2 Run `cd src && python3 -m unittest test_app.py -v` and verify exit 0, including holdout missing-file JSON 400
