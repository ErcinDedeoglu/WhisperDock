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
The Docker image workflow SHALL pin JavaScript actions to full-length commit SHAs with a same-line version comment naming the Node 24 majors (`# v7` for checkout and build-push, `# v4` for qemu, buildx, and login). It MUST NOT use `actions/checkout@v4`, `docker/setup-qemu-action@v3`, `docker/setup-buildx-action@v3`, `docker/login-action@v3`, or `docker/build-push-action@v5`.

#### Scenario: Workflow uses Node 24 action majors
- **WHEN** the Docker image workflow file's `uses:` lines are read
- **THEN** checkout has comment `# v7`, QEMU `# v4`, Buildx `# v4`, login `# v4`, and build-push `# v7`

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
