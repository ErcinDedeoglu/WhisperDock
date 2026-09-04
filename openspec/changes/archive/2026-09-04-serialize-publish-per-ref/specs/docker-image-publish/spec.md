## ADDED Requirements

### Requirement: Publish runs are serialized per ref
The Docker image workflow SHALL declare a top-level `concurrency` mapping whose `group` includes `github.workflow` and `github.ref`. `cancel-in-progress` MUST be `false` so an in-flight Hub push is not aborted.

#### Scenario: Concurrency group is per workflow and ref
- **WHEN** the published workflow file's top-level `concurrency` mapping is read
- **THEN** `group` contains `github.workflow` and `github.ref`

#### Scenario: In-progress publish is not cancelled
- **WHEN** the published workflow file's top-level `concurrency` mapping is read
- **THEN** `cancel-in-progress` is `false`
