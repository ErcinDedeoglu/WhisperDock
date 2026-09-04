# container-runtime Specification

## Purpose

Defines the CPython runtime the transcription service container MUST ship, so rebuilds stay on a supported interpreter rather than an end-of-life Python line.

## Requirements

### Requirement: Service image runs supported CPython 3.12
The service container SHALL run CPython 3.12.x. The interpreter MUST NOT be Python 3.8 or any other end-of-life CPython line.

#### Scenario: Container python reports 3.12
- **WHEN** the built service image is started with `python -c "import sys; print(sys.version_info[:2])"`
- **THEN** the printed pair is `(3, 12)`

#### Scenario: End-of-life 3.8 is not the image python
- **WHEN** the built service image python is inspected
- **THEN** `sys.version_info[:2]` is not `(3, 8)`

### Requirement: Service image pins Flask and gunicorn
The service container SHALL install Flask 3.1.3 and gunicorn 26.2.0. The image build MUST NOT install those packages without an exact version pin.

#### Scenario: Image reports pinned Flask and gunicorn
- **WHEN** the built service image is started with `python -c "from importlib.metadata import version; print(version('flask')); print(version('gunicorn'))"`
- **THEN** the printed versions are `3.1.3` and `26.2.0`

#### Scenario: Dockerfile pip install is version-pinned
- **WHEN** the service Dockerfile pip install instruction for Flask and gunicorn is read
- **THEN** it specifies `Flask==3.1.3` and `gunicorn==26.2.0` and does not install those two packages without `==`

### Requirement: Dockerfile ENV uses key=value form
The service Dockerfile SHALL set environment variables with `ENV key=value`. It MUST NOT use the legacy space-separated `ENV key value` form.

#### Scenario: PYTHONDONTWRITEBYTECODE and PYTHONUNBUFFERED use equals
- **WHEN** the service Dockerfile ENV instructions are read
- **THEN** they include `ENV PYTHONDONTWRITEBYTECODE=1` and `ENV PYTHONUNBUFFERED=1`

#### Scenario: Legacy space-separated ENV is absent
- **WHEN** the service Dockerfile is searched for `ENV PYTHONDONTWRITEBYTECODE 1` or `ENV PYTHONUNBUFFERED 1`
- **THEN** those space-separated forms are not present
