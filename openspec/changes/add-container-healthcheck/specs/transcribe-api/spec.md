## ADDED Requirements

### Requirement: Health probe is a JSON success
GET `/health` SHALL respond with HTTP 200 and an `application/json` body whose `status` field is `ok`. The handler MUST NOT run ffmpeg or whisper-cli.

#### Scenario: GET health
- **WHEN** a client GETs `/health`
- **THEN** the response status is 200, the content type is JSON, and the body is `{"status": "ok"}`
