## ADDED Requirements

### Requirement: Service image pins Flask and gunicorn
The service container SHALL install Flask 3.1.3 and gunicorn 26.2.0. The image build MUST NOT install those packages without an exact version pin.

#### Scenario: Image reports pinned Flask and gunicorn
- **WHEN** the built service image is started with `python -c "from importlib.metadata import version; print(version('flask')); print(version('gunicorn'))"`
- **THEN** the printed versions are `3.1.3` and `26.2.0`

#### Scenario: Dockerfile pip install is version-pinned
- **WHEN** the service Dockerfile pip install instruction for Flask and gunicorn is read
- **THEN** it specifies `Flask==3.1.3` and `gunicorn==26.2.0` and does not install those two packages without `==`
