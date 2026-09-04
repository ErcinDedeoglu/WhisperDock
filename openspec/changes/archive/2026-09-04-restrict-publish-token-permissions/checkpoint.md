# Checkpoint: restrict-publish-token-permissions

## Goal
Add top-level `permissions: contents: read` to `publish-docker.yml`.

## Acceptance
- Parsed YAML `permissions.contents == "read"`
- No write-all / contents: write / packages: write / id-token: write
- unittest 8/8 including holdout missing-file JSON 400
- Chrome: N/A
- GHA 🐳 success; Hub tag cli-ok

## Non-goals
sync-whisper.yml; org/repo default workflow permissions; extra write scopes

## Constraints
- Origin push main allowed
- Changing publish-docker.yml starts Docker Images
- Do not tag/push dublok/* locally
- Unittest from src/
- PyYAML parses `on` as True

## Active phase
apply

## Hypothesis
Top-level contents: read → unspecified scopes none; Hub still publishes via Docker secrets.

## Expected signal
YAML permissions.contents == read; GHA success; Hub cli-ok

## Tasks
- 1.1 pending
- 2.1 pending

## Last action
Wrote OpenSpec artifacts; validated strict.

## Next action
Insert permissions block; verify YAML; run unittest.
