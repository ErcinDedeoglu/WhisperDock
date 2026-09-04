## ADDED Requirements

### Requirement: Runtime working directory is the app root
The service image SHALL set `WORKDIR` to `/app` after the whisper build steps so the container default working directory is the Flask app root, not the whisper.cpp tree.

#### Scenario: Image working directory is /app
- **WHEN** the built service image is inspected for `Config.WorkingDir`
- **THEN** the value is `/app`
