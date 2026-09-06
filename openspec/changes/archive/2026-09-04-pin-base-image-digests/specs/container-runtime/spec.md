## ADDED Requirements

### Requirement: Base images are pinned by digest
The service Dockerfile SHALL pin `python:3.12-bookworm`, `python:3.12-slim-bookworm`, and `mwader/static-ffmpeg:9.0.1` with `@sha256:` index digests.

#### Scenario: Dockerfile pins python bookworm by digest
- **WHEN** the service Dockerfile is read
- **THEN** it contains `python:3.12-bookworm@sha256:`

#### Scenario: Dockerfile pins python slim-bookworm by digest
- **WHEN** the service Dockerfile is read
- **THEN** it contains `python:3.12-slim-bookworm@sha256:`

#### Scenario: Dockerfile pins static-ffmpeg by digest
- **WHEN** the service Dockerfile is read
- **THEN** it contains `mwader/static-ffmpeg:9.0.1@sha256:`
