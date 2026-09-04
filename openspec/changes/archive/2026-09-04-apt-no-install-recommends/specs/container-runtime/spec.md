## ADDED Requirements

### Requirement: Apt install avoids recommends and leftover lists
The service Dockerfile SHALL install Debian packages with `apt-get install -y --no-install-recommends` and SHALL delete `/var/lib/apt/lists/*` in the same `RUN` as `apt-get update`.

#### Scenario: Install uses no-install-recommends
- **WHEN** the service Dockerfile apt-get install instruction is read
- **THEN** it includes `--no-install-recommends`

#### Scenario: Apt lists are removed in the install RUN
- **WHEN** the service Dockerfile is read
- **THEN** `rm -rf /var/lib/apt/lists/*` appears in the same RUN as `apt-get update` and there is no later standalone `apt-get clean` RUN
