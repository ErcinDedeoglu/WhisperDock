## ADDED Requirements

### Requirement: Service process does not run as root
The service container SHALL run gunicorn as a non-root user. The image MUST declare `USER` to that account.

#### Scenario: Image default user is not root
- **WHEN** the built service image is inspected for `Config.User`
- **THEN** the value is a non-root account name (not empty and not `root`)

#### Scenario: Container process uid is not 0
- **WHEN** the built service image is started with `id -u`
- **THEN** the printed uid is not `0`
