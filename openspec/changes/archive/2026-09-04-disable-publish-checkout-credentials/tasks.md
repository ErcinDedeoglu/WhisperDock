## 1. Checkout

- [x] 1.1 Set `persist-credentials: false` on the checkout step in `.github/workflows/publish-docker.yml`, then verify the parsed checkout `with.persist-credentials` is false

## 2. Holdout

- [x] 2.1 Run `cd src && python3 -m unittest test_app.py` and verify exit 0 including holdout missing-file JSON 400
