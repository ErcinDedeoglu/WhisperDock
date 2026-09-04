## ADDED Requirements

### Requirement: Runtime image omits C++ build tools
The published service image SHALL NOT install `cmake`. It MUST still provide `/app/whisper/build/bin/whisper-cli`.

#### Scenario: cmake is not installed
- **WHEN** the built service image is started with `dpkg-query -W cmake`
- **THEN** the command exits non-zero

#### Scenario: whisper-cli remains
- **WHEN** the built service image is started with `test -x /app/whisper/build/bin/whisper-cli`
- **THEN** the command exits 0
