## ADDED Requirements

### Requirement: Publish checkout does not persist credentials
The Docker image workflow checkout step SHALL set `persist-credentials: false`. It MUST NOT leave the default persisted `GITHUB_TOKEN` in git config for later steps.

#### Scenario: persist-credentials is false
- **WHEN** the published workflow file's checkout step `with` mapping is read
- **THEN** `persist-credentials` is `false`
