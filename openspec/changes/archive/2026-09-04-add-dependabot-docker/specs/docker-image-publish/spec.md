## ADDED Requirements

### Requirement: Dependabot watches the service Dockerfile
The repository SHALL declare Dependabot version updates for the Docker ecosystem in `/src` so pinned base image digests can be bumped.

#### Scenario: dependabot.yml enables docker in src
- **WHEN** `.github/dependabot.yml` is read
- **THEN** it contains `package-ecosystem: docker` and `directory: /src`
