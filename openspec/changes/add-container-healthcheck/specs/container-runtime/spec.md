## ADDED Requirements

### Requirement: Image declares an HTTP health check
The service image SHALL include a `HEALTHCHECK` that GETs `/health` on the bind address using the image Python interpreter. The check MUST exit 0 when that route returns HTTP 200.

#### Scenario: Image Healthcheck is set
- **WHEN** the built service image is inspected for `Config.Healthcheck`
- **THEN** the value is not null and the Test command includes `/health`

#### Scenario: Started container becomes healthy
- **WHEN** the built service image is run and the start period elapses
- **THEN** `State.Health.Status` is `healthy`
