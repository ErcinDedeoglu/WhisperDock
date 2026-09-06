## 1. Config

- [x] 1.1 Add a pip Dependabot entry with `directory: /src` and weekly schedule, then verify docker, github-actions, and pip entries are all present

## 2. Holdout

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
