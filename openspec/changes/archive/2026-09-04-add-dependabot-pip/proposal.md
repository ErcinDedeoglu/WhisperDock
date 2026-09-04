## Why

`src/requirements.txt` is hashed and frozen. Dependabot watches docker and github-actions only, so Flask/gunicorn hashes never bump automatically.

## What Changes

- Add `package-ecosystem: pip`, `directory: /src`, weekly.

## Capabilities

### New Capabilities

### Modified Capabilities

- `container-runtime`: Dependabot SHALL watch pip manifests in `/src`.

## Impact

- `.github/dependabot.yml` only. Docker Images should not run.

## Problem

Hashed Python deps have no automated bump path.

## Non-goals

- SHA-pinning Actions. Changing requirement versions now.

## Hypothesis

If Dependabot pip scans `/src` weekly, then the YAML contains `package-ecosystem: pip` and `directory: /src`, unittest still exits 0, and `🐳 Docker Images` does not start.

## Expected signal

- Those keys present (pip entry distinct from docker `/src`).
- Unittest 8/8.
- Dependabot Updates `pip in /src` starts; no Docker Images.

## Research

- https://docs.github.com/code-security/dependabot/dependabot-version-updates/configuration-options-for-the-dependabot.yml-file — pip ecosystem; directory is the folder with requirements.txt.
- Dependabot-core RequirementReplacer preserves `--hash=sha256:` lines.

## Chosen / rejected

- Chosen: pip `/src` weekly alongside docker `/src`.
- Rejected: directory `/` (requirements.txt is under src/). pip-compile `.in` files (none exist).

## Rollback

Remove the pip updates block.

## Acceptance

- unittest 8/8 including holdout missing-file JSON 400.
- Chrome: N/A — no UI.
