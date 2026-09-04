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

### Requirement: Service process does not run as root
The service container SHALL run gunicorn as a non-root user. The image MUST declare `USER` to that account.

#### Scenario: Image default user is not root
- **WHEN** the built service image is inspected for `Config.User`
- **THEN** the value is a non-root account name (not empty and not `root`)

#### Scenario: Container process uid is not 0
- **WHEN** the built service image is started with `id -u`
- **THEN** the printed uid is not `0`

### Requirement: Image declares an HTTP health check
The service image SHALL include a `HEALTHCHECK` that GETs `/health` on the bind address using the image Python interpreter. The check MUST exit 0 when that route returns HTTP 200.

#### Scenario: Image Healthcheck is set
- **WHEN** the built service image is inspected for `Config.Healthcheck`
- **THEN** the value is not null and the Test command includes `/health`

#### Scenario: Started container becomes healthy
- **WHEN** the built service image is run and the start period elapses
- **THEN** `State.Health.Status` is `healthy`

### Requirement: Runtime working directory is the app root
The service image SHALL set `WORKDIR` to `/app` after the whisper build steps so the container default working directory is the Flask app root, not the whisper.cpp tree.

#### Scenario: Image working directory is /app
- **WHEN** the built service image is inspected for `Config.WorkingDir`
- **THEN** the value is `/app`

### Requirement: Apt install avoids recommends and leftover lists
The service Dockerfile SHALL install Debian packages with `apt-get install -y --no-install-recommends` and SHALL delete `/var/lib/apt/lists/*` in the same `RUN` as `apt-get update`.

#### Scenario: Install uses no-install-recommends
- **WHEN** the service Dockerfile apt-get install instruction is read
- **THEN** it includes `--no-install-recommends`

#### Scenario: Apt lists are removed in the install RUN
- **WHEN** the service Dockerfile is read
- **THEN** `rm -rf /var/lib/apt/lists/*` appears in the same RUN as `apt-get update` and there is no later standalone `apt-get clean` RUN

### Requirement: Apt packages are version-pinned
The service Dockerfile SHALL install `build-essential`, `cmake`, `ffmpeg`, `git`, and `libsndfile1` with explicit `package=version` pins.

#### Scenario: Dockerfile pins the five apt packages
- **WHEN** the service Dockerfile apt-get install instruction is read
- **THEN** it contains `build-essential=12.9`, `cmake=3.25.1-1`, `ffmpeg=7:5.1.9-0+deb12u1`, `git=1:2.39.5-0+deb12u3`, and `libsndfile1=1.2.0-1+deb12u1`

### Requirement: Runtime image omits C++ build tools
The published service image SHALL NOT install `cmake`. It MUST still provide `/app/whisper/build/bin/whisper-cli`.

#### Scenario: cmake is not installed
- **WHEN** the built service image is started with `dpkg-query -W cmake`
- **THEN** the command exits non-zero

#### Scenario: whisper-cli remains
- **WHEN** the built service image is started with `test -x /app/whisper/build/bin/whisper-cli`
- **THEN** the command exits 0
