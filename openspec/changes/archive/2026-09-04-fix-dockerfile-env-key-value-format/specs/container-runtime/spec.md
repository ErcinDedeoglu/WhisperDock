## ADDED Requirements

### Requirement: Dockerfile ENV uses key=value form
The service Dockerfile SHALL set environment variables with `ENV key=value`. It MUST NOT use the legacy space-separated `ENV key value` form.

#### Scenario: PYTHONDONTWRITEBYTECODE and PYTHONUNBUFFERED use equals
- **WHEN** the service Dockerfile ENV instructions are read
- **THEN** they include `ENV PYTHONDONTWRITEBYTECODE=1` and `ENV PYTHONUNBUFFERED=1`

#### Scenario: Legacy space-separated ENV is absent
- **WHEN** the service Dockerfile is searched for `ENV PYTHONDONTWRITEBYTECODE 1` or `ENV PYTHONUNBUFFERED 1`
- **THEN** those space-separated forms are not present
