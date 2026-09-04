## ADDED Requirements

### Requirement: Publish workflow GITHUB_TOKEN is contents-read
The Docker image workflow SHALL declare a top-level `permissions` mapping with `contents: read`. It MUST NOT grant `contents: write`, `write-all`, `packages: write`, or `id-token: write`.

#### Scenario: Top-level contents read is present
- **WHEN** the published workflow file's top-level `permissions` mapping is read
- **THEN** `contents` is `read`

#### Scenario: Write scopes are absent
- **WHEN** the published workflow file is searched for token write grants
- **THEN** it does not contain `contents: write`, `write-all`, `packages: write`, or `id-token: write`
