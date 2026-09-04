## MODIFIED Requirements

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
- **THEN** each `uses:` ref after `@` is a 40-character hexadecimal SHA
