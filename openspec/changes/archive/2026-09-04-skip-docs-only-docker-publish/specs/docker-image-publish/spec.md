## ADDED Requirements

### Requirement: Automatic publish only when image sources change
The Docker image workflow SHALL start an automatic push-on-git-push job only when the push changes `src/**` or `.github/workflows/publish-docker.yml`. It MUST NOT use `paths-ignore` on the same push event.

#### Scenario: Push path filter is src and the workflow file
- **WHEN** the published workflow file's `on.push.paths` list is read
- **THEN** the list contains `src/**` and `.github/workflows/publish-docker.yml`

#### Scenario: paths-ignore is absent on push
- **WHEN** the published workflow file's `on.push` mapping is read
- **THEN** it does not contain `paths-ignore`
