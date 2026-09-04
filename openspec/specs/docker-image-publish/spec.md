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

### Requirement: Publish workflow JavaScript actions use Node 24
The Docker image workflow SHALL pin JavaScript actions to full-length commit SHAs with a same-line version comment naming the Node 24 majors (`# v7` for checkout and build-push, `# v4` for buildx and login). It MUST NOT use `actions/checkout@v4`, `docker/setup-qemu-action@v3`, `docker/setup-buildx-action@v3`, `docker/login-action@v3`, or `docker/build-push-action@v5`.

#### Scenario: Workflow uses Node 24 action majors
- **WHEN** the Docker image workflow file's `uses:` lines are read
- **THEN** checkout has comment `# v7`, Buildx `# v4`, login `# v4`, and build-push `# v7`

#### Scenario: Node 20 majors are absent
- **WHEN** the Docker image workflow file is searched for the previously used Node 20 tags
- **THEN** it does not contain `actions/checkout@v4`, `docker/setup-qemu-action@v3`, `docker/setup-buildx-action@v3`, `docker/login-action@v3`, or `docker/build-push-action@v5`

#### Scenario: Publish workflow actions are SHA-pinned
- **WHEN** the Docker image workflow file's `uses:` lines are read
- **THEN** each `uses:` ref after `@` is a 40-character hexadecimal SHA`

### Requirement: Automatic publish only when image sources change
The Docker image workflow SHALL start an automatic push-on-git-push job only when the push changes `src/**` or `.github/workflows/publish-docker.yml`. It MUST NOT use `paths-ignore` on the same push event.

#### Scenario: Push path filter is src and the workflow file
- **WHEN** the published workflow file's `on.push.paths` list is read
- **THEN** the list contains `src/**` and `.github/workflows/publish-docker.yml`

#### Scenario: paths-ignore is absent on push
- **WHEN** the published workflow file's `on.push` mapping is read
- **THEN** it does not contain `paths-ignore`

### Requirement: Dependabot watches the service Dockerfile
The repository SHALL declare Dependabot version updates for the Docker ecosystem in `/src` so pinned base image digests can be bumped.

#### Scenario: dependabot.yml enables docker in src
- **WHEN** `.github/dependabot.yml` is read
- **THEN** it contains `package-ecosystem: docker` and `directory: /src`

### Requirement: Dependabot watches GitHub Actions
The repository SHALL declare Dependabot version updates for the GitHub Actions ecosystem at `/` so workflow `uses:` majors can be bumped.

#### Scenario: dependabot.yml enables github-actions at repo root
- **WHEN** `.github/dependabot.yml` is read
- **THEN** it contains `package-ecosystem: github-actions` and a `directory: /` entry for that ecosystem

### Requirement: Publish workflow GITHUB_TOKEN is contents-read
The Docker image workflow SHALL declare a top-level `permissions` mapping with `contents: read`. It MUST NOT grant `contents: write`, `write-all`, `packages: write`, or `id-token: write`.

#### Scenario: Top-level contents read is present
- **WHEN** the published workflow file's top-level `permissions` mapping is read
- **THEN** `contents` is `read`

#### Scenario: Write scopes are absent
- **WHEN** the published workflow file is searched for token write grants
- **THEN** it does not contain `contents: write`, `write-all`, `packages: write`, or `id-token: write`

### Requirement: Publish checkout does not persist credentials
The Docker image workflow checkout step SHALL set `persist-credentials: false`. It MUST NOT leave the default persisted `GITHUB_TOKEN` in git config for later steps.

#### Scenario: persist-credentials is false
- **WHEN** the published workflow file's checkout step `with` mapping is read
- **THEN** `persist-credentials` is `false`

### Requirement: Publish workflow does not install QEMU
The Docker image workflow SHALL NOT use `docker/setup-qemu-action`. It MUST keep `platforms` as `linux/amd64` only.

#### Scenario: setup-qemu-action is absent
- **WHEN** the published workflow file is searched for QEMU setup
- **THEN** it does not contain `docker/setup-qemu-action`

#### Scenario: Publish platform remains amd64
- **WHEN** the published workflow file's build-push `platforms` value is read
- **THEN** it is exactly `linux/amd64`

### Requirement: Publish job has a bounded timeout
The Docker image workflow's Linux build-and-push job SHALL set `timeout-minutes` to `20`. It MUST NOT rely on the platform default of 360 minutes.

#### Scenario: Job timeout is 20 minutes
- **WHEN** the published workflow file's `linux-build-and-push` job mapping is read
- **THEN** `timeout-minutes` is `20`
