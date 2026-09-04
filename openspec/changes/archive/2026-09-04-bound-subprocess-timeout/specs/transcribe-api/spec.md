## ADDED Requirements

### Requirement: Hung conversion or transcription is a JSON server error
When ffmpeg or whisper-cli exceeds 240 seconds, the service SHALL respond with HTTP 500 and an `application/json` body whose `error` field is `Error in transcription`. The subprocess invocations MUST pass `timeout=240`.

#### Scenario: Conversion subprocess times out
- **WHEN** ffmpeg raises `TimeoutExpired`
- **THEN** the response status is 500, the content type is JSON, and the body is `{"error": "Error in transcription"}`
