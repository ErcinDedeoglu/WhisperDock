## ADDED Requirements

### Requirement: Dependabot watches GitHub Actions
The repository SHALL declare Dependabot version updates for the GitHub Actions ecosystem at `/` so workflow `uses:` majors can be bumped.

#### Scenario: dependabot.yml enables github-actions at repo root
- **WHEN** `.github/dependabot.yml` is read
- **THEN** it contains `package-ecosystem: github-actions` and a `directory: /` entry for that ecosystem
