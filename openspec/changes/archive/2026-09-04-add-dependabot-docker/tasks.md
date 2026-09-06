## 1. Config

- [x] 1.1 Add `.github/dependabot.yml` with docker ecosystem `directory: /src` and weekly schedule, then verify those keys are present

## 2. Holdout

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
