## 1. Pins

- [x] 1.1 Pin the five apt packages in `src/Dockerfile` to the proven versions, rebuild `whisperdock-local:pinned` (not `dublok/*`), and verify `dpkg-query` matches

## 2. Verify

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
