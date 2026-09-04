## ADDED Requirements

### Requirement: Runtime image omits Mesa LLVM from Debian ffmpeg
The published service image SHALL NOT install `libllvm15`. It MUST provide an `ffmpeg` executable on PATH.

#### Scenario: libllvm15 is not installed
- **WHEN** the built service image is started with `dpkg-query -W libllvm15`
- **THEN** the command exits non-zero

#### Scenario: ffmpeg remains on PATH
- **WHEN** the built service image is started with `command -v ffmpeg`
- **THEN** the command exits 0
