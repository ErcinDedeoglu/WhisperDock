## ADDED Requirements

### Requirement: ffmpeg conversion does not read stdin
When converting uploaded media, the service SHALL invoke ffmpeg with `-nostdin` so the process does not wait for interactive stdin.

#### Scenario: Conversion argv disables stdin
- **WHEN** the transcribe conversion command list is read
- **THEN** it contains `-nostdin`
