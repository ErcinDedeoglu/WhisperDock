## Purpose

Defines the CPython runtime the transcription service container MUST ship, so rebuilds stay on a supported interpreter rather than an end-of-life Python line.

## ADDED Requirements

### Requirement: Service image runs supported CPython 3.12
The service container SHALL run CPython 3.12.x. The interpreter MUST NOT be Python 3.8 or any other end-of-life CPython line.

#### Scenario: Container python reports 3.12
- **WHEN** the built service image is started with `python -c "import sys; print(sys.version_info[:2])"`
- **THEN** the printed pair is `(3, 12)`

#### Scenario: End-of-life 3.8 is not the image python
- **WHEN** the built service image python is inspected
- **THEN** `sys.version_info[:2]` is not `(3, 8)`
