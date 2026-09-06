## ADDED Requirements

### Requirement: Pip install verifies package hashes
The service image build SHALL install Python packages in pip hash-checking mode. Every frozen requirement MUST include at least one sha256 hash. The install MUST pass `--require-hashes`.

#### Scenario: Requirements file lists hashes
- **WHEN** `src/requirements.txt` is read
- **THEN** every package line includes `--hash=sha256:`

#### Scenario: Dockerfile forces hash-checking
- **WHEN** the service Dockerfile pip install instruction is read
- **THEN** it includes `--require-hashes`
