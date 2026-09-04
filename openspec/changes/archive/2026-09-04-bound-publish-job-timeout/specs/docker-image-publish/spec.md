## ADDED Requirements

### Requirement: Publish job has a bounded timeout
The Docker image workflow's Linux build-and-push job SHALL set `timeout-minutes` to `20`. It MUST NOT rely on the platform default of 360 minutes.

#### Scenario: Job timeout is 20 minutes
- **WHEN** the published workflow file's `linux-build-and-push` job mapping is read
- **THEN** `timeout-minutes` is `20`
