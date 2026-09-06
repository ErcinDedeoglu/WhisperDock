## ADDED Requirements

### Requirement: Apt packages are version-pinned
The service Dockerfile SHALL install `build-essential`, `cmake`, `ffmpeg`, `git`, and `libsndfile1` with explicit `package=version` pins.

#### Scenario: Dockerfile pins the five apt packages
- **WHEN** the service Dockerfile apt-get install instruction is read
- **THEN** it contains `build-essential=12.9`, `cmake=3.25.1-1`, `ffmpeg=7:5.1.9-0+deb12u1`, `git=1:2.39.5-0+deb12u3`, and `libsndfile1=1.2.0-1+deb12u1`
