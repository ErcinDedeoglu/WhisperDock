## ADDED Requirements

### Requirement: Service image pins Flask gunicorn and transitives
The service container SHALL install Flask 3.1.3, gunicorn 26.2.0, and the transitive set blinker 1.9.0, click 8.5.0, itsdangerous 2.2.0, Jinja2 3.1.6, MarkupSafe 3.0.3, Werkzeug 3.1.8. The image build MUST install those packages from a frozen requirements file rather than an unpinned transitive resolve.

#### Scenario: Image reports frozen direct and transitive versions
- **WHEN** the built service image is started with `python -c "from importlib.metadata import version; print(version('flask'), version('gunicorn'), version('werkzeug'), version('jinja2'), version('markupsafe'), version('blinker'), version('click'), version('itsdangerous'))"`
- **THEN** the printed versions are `3.1.3 26.2.0 3.1.8 3.1.6 3.0.3 1.9.0 8.5.0 2.2.0`

#### Scenario: Dockerfile installs from a frozen requirements file
- **WHEN** the service Dockerfile pip install instruction is read
- **THEN** it installs with `-r` a requirements file and does not contain an inline `pip install Flask==`
