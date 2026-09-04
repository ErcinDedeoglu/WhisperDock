## ADDED Requirements

### Requirement: Runtime image omits g++
The published service image SHALL NOT install `g++`. It MUST still provide `/app/whisper/build/bin/whisper-cli` and CPython 3.12.

#### Scenario: g++ is not installed
- **WHEN** the built service image is started with `dpkg-query -W g++`
- **THEN** the command exits non-zero

#### Scenario: whisper-cli remains after slim runtime
- **WHEN** the built service image is started with `test -x /app/whisper/build/bin/whisper-cli`
- **THEN** the command exits 0
