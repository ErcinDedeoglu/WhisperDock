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
The service container SHALL install Flask 3.1.3 and gunicorn 26.2.0 from a frozen requirements file. The image build MUST NOT resolve those packages as unpinned latest.

#### Scenario: Image reports pinned Flask and gunicorn
- **WHEN** the built service image is started with `python -c "from importlib.metadata import version; print(version('flask')); print(version('gunicorn'))"`
- **THEN** the printed versions are `3.1.3` and `26.2.0`

#### Scenario: Dockerfile pip install is version-pinned
- **WHEN** the service Dockerfile pip install instruction is read
- **THEN** it installs from a requirements file with `-r` that pins `Flask==3.1.3` and `gunicorn==26.2.0`

### Requirement: Service image pins Flask gunicorn and transitives
The service container SHALL install Flask 3.1.3, gunicorn 26.2.0, and the transitive set blinker 1.9.0, click 8.5.0, itsdangerous 2.2.0, Jinja2 3.1.6, MarkupSafe 3.0.3, Werkzeug 3.1.8. The image build MUST install those packages from a frozen requirements file rather than an unpinned transitive resolve.

#### Scenario: Image reports frozen direct and transitive versions
- **WHEN** the built service image is started with `python -c "from importlib.metadata import version; print(version('flask'), version('gunicorn'), version('werkzeug'), version('jinja2'), version('markupsafe'), version('blinker'), version('click'), version('itsdangerous'))"`
- **THEN** the printed versions are `3.1.3 26.2.0 3.1.8 3.1.6 3.0.3 1.9.0 8.5.0 2.2.0`

#### Scenario: Dockerfile installs from a frozen requirements file
- **WHEN** the service Dockerfile pip install instruction is read
- **THEN** it installs with `-r` a requirements file and does not contain an inline `pip install Flask==`

### Requirement: Dockerfile ENV uses key=value form
The service Dockerfile SHALL set environment variables with `ENV key=value`. It MUST NOT use the legacy space-separated `ENV key value` form.

#### Scenario: PYTHONDONTWRITEBYTECODE and PYTHONUNBUFFERED use equals
- **WHEN** the service Dockerfile ENV instructions are read
- **THEN** they include `ENV PYTHONDONTWRITEBYTECODE=1` and `ENV PYTHONUNBUFFERED=1`

#### Scenario: Legacy space-separated ENV is absent
- **WHEN** the service Dockerfile is searched for `ENV PYTHONDONTWRITEBYTECODE 1` or `ENV PYTHONUNBUFFERED 1`
- **THEN** those space-separated forms are not present

### Requirement: Pip install verifies package hashes
The service image build SHALL install Python packages in pip hash-checking mode. Every frozen requirement MUST include at least one sha256 hash. The install MUST pass `--require-hashes`.

#### Scenario: Requirements file lists hashes
- **WHEN** `src/requirements.txt` is read
- **THEN** every package line includes `--hash=sha256:`

#### Scenario: Dockerfile forces hash-checking
- **WHEN** the service Dockerfile pip install instruction is read
- **THEN** it includes `--require-hashes`
