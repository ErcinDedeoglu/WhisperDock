# docker-image-publish Specification

## Purpose

Defines when the Linux Docker image workflow is allowed to push `dublok/whisperdock` tags, so feature-branch commits cannot publish production images.

## Requirements

### Requirement: Automatic publish only on main
The Docker image workflow SHALL start an automatic push-on-git-push job only for the `main` branch. It MUST NOT use an all-branches push filter such as `**`.

#### Scenario: Push filter is main only
- **WHEN** the published workflow file's `on.push.branches` list is read
- **THEN** the list is exactly `main` and does not contain `**`

### Requirement: Manual dispatch remains available
The Docker image workflow SHALL still declare `workflow_dispatch` so a caller can run it on a chosen ref without a matching git-push event.

#### Scenario: workflow_dispatch trigger is present
- **WHEN** the published workflow file's `on` mapping is read
- **THEN** `workflow_dispatch` is present
