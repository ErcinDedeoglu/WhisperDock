# Verify

- Date: 2026-09-04
- Gate: README contains `docker build -t whisperdock src`; does not contain `docker build -t whisperdock .`
- Holdout: `cd src && python3 -m unittest test_app.py` exit 0 (6 tests)
- Chrome: N/A — no UI
- OpenSpec: `openspec validate fix-readme-docker-build-context --strict` exit 0 (`skip_specs`)
- Sync: n/a — no delta specs
- CI/CD: n/a this turn (push after archive is allowed on main)
- Environment: n/a
