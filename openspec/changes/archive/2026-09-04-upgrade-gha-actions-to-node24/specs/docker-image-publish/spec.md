## ADDED Requirements

### Requirement: Publish workflow JavaScript actions use Node 24
The Docker image workflow SHALL use JavaScript actions that declare a Node 24 runtime. It MUST NOT use `actions/checkout@v4`, `docker/setup-qemu-action@v3`, `docker/setup-buildx-action@v3`, `docker/login-action@v3`, or `docker/build-push-action@v5`.

#### Scenario: Workflow uses Node 24 action majors
- **WHEN** the Docker image workflow file's `uses:` tags are read
- **THEN** checkout is `actions/checkout@v7`, QEMU is `docker/setup-qemu-action@v4`, Buildx is `docker/setup-buildx-action@v4`, login is `docker/login-action@v4`, and build-push is `docker/build-push-action@v7`

#### Scenario: Node 20 majors are absent
- **WHEN** the Docker image workflow file is searched for the previously used Node 20 tags
- **THEN** it does not contain `actions/checkout@v4`, `docker/setup-qemu-action@v3`, `docker/setup-buildx-action@v3`, `docker/login-action@v3`, or `docker/build-push-action@v5`
