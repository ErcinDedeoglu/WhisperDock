## 1. Config

- [x] 1.1 Add a github-actions Dependabot entry with `directory: /` and weekly schedule, then verify docker `/src` remains and github-actions `/` is present

## 2. Holdout

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
