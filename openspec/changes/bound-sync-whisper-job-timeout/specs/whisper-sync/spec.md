## Purpose

Defines how the whisper.cpp sync workflow is allowed to run so hung clones cannot consume the GitHub-hosted six-hour job limit.

## ADDED Requirements

### Requirement: Sync jobs have a bounded timeout
Both jobs in the whisper.cpp sync workflow SHALL set `timeout-minutes` to `10`. They MUST NOT rely on the platform default of 360 minutes.

#### Scenario: sync-by-tag timeout is 10 minutes
- **WHEN** the sync workflow file's `sync-by-tag` job mapping is read
- **THEN** `timeout-minutes` is `10`

#### Scenario: sync-latest-commit timeout is 10 minutes
- **WHEN** the sync workflow file's `sync-latest-commit` job mapping is read
- **THEN** `timeout-minutes` is `10`
