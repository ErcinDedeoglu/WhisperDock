# Verify

## Gates

- unittest 6/6; holdout missing-file 400 JSON
- `whisperdock-local:pinned` metadata match freeze
- pip install used `--require-hashes`; MarkupSafe aarch64 wheel accepted
- Chrome: N/A — no UI

## Remote

- Push: `main` `6526193` → `origin/main`
- CI/CD: GitHub Actions `🐳 Docker Images` run 33851184564 success (2m56s) — amd64 hash-check passed
- Environment: `dublok/whisperdock:6526193` Flask 3.1.3 gunicorn 26.2.0 MarkupSafe 3.0.3
