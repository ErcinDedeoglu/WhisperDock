## REMOVED Requirements

### Requirement: Apt packages are version-pinned
**Reason**: Builder no longer installs Debian ffmpeg or libsndfile1; those pins described a five-package install that is obsolete after static runtime ffmpeg.
**Migration**: Use builder pins for `build-essential`, `cmake`, and `git` only.

## ADDED Requirements

### Requirement: Builder apt packages are version-pinned
The service Dockerfile builder stage SHALL install `build-essential`, `cmake`, and `git` with explicit `package=version` pins. It MUST NOT install Debian `ffmpeg` or `libsndfile1`.

#### Scenario: Dockerfile pins the three builder apt packages
- **WHEN** the service Dockerfile builder apt-get install instruction is read
- **THEN** it contains `build-essential=12.9`, `cmake=3.25.1-1`, and `git=1:2.39.5-0+deb12u3`

#### Scenario: Builder does not install Debian ffmpeg
- **WHEN** the service Dockerfile builder apt-get install instruction is read
- **THEN** it does not contain `ffmpeg=`
