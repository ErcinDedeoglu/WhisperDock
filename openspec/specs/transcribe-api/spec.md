# transcribe-api Specification

## Purpose

Defines the HTTP `/transcribe` contract for this speech-to-text service: JSON success and error bodies, client vs server failure status, and that uploaded media is not left on disk after the request.

## Requirements

### Requirement: Missing upload is a JSON client error
When the request has no file part or an empty filename, the service SHALL respond with HTTP 400 and an `application/json` body whose `error` field identifies the missing upload. The response MUST NOT be HTML.

#### Scenario: No file part
- **WHEN** a client POSTs `/transcribe` with no `file` part
- **THEN** the response status is 400, the content type is JSON, and the body is `{"error": "No file part"}`

#### Scenario: Empty filename
- **WHEN** a client POSTs `/transcribe` with a `file` part whose filename is empty
- **THEN** the response status is 400, the content type is JSON, and the body is `{"error": "No selected file"}`

### Requirement: Oversized upload is a JSON client error
When the request body exceeds 16 000 000 bytes, the service SHALL respond with HTTP 413 and an `application/json` body that includes a non-empty `error` field. The response MUST NOT be HTML. The service MUST NOT persist new temporary media files for that request and MUST NOT start media conversion.

#### Scenario: Body larger than the upload limit
- **WHEN** a client POSTs `/transcribe` with a `file` part whose request body is larger than 16 000 000 bytes
- **THEN** the response status is 413, the content type is JSON, the JSON object contains an `error` field, and the process temp directory contains no new `tmp*` files created for that request

### Requirement: Unconvertible media is a JSON client error
When the uploaded bytes cannot be converted to 16 kHz mono WAV, the service SHALL respond with HTTP 400 and an `application/json` body that includes an `error` field. The response MUST NOT be HTML.

#### Scenario: Garbage bytes uploaded as wav
- **WHEN** a client POSTs `/transcribe` with a `file` named `bad.wav` whose contents are not valid media
- **THEN** the response status is 400, the content type is JSON, and the JSON object contains an `error` field

### Requirement: Transcription subprocess failure is a JSON server error
When media conversion succeeds but the transcription process cannot be started or exits non-zero, the service SHALL respond with HTTP 500 and an `application/json` body whose `error` field is `Error in transcription`. The response MUST NOT be HTML.

#### Scenario: Transcription binary missing
- **WHEN** conversion succeeds and the transcription executable is not present at the configured path
- **THEN** the response status is 500, the content type is JSON, and the body is `{"error": "Error in transcription"}`

### Requirement: Successful transcription is JSON segments
When conversion and transcription both succeed, the service SHALL respond with HTTP 200 and an `application/json` body whose `transcription` field is a list of objects with `start_time`, `end_time`, and `text`.

#### Scenario: Timestamped whisper-cli stdout
- **WHEN** transcription stdout contains lines of the form `[HH:MM:SS.mmm --> HH:MM:SS.mmm] text`
- **THEN** each line becomes one object in `transcription` with those times and stripped text

### Requirement: Temporary media files are removed after the request
The service SHALL delete the uploaded file and the converted WAV path after the request finishes, including when conversion or transcription fails.

#### Scenario: Invalid media leaves no temp files
- **WHEN** a client POSTs unconvertible bytes as `bad.wav`
- **THEN** after the response is returned, the process temp directory contains no new `tmp*` files created for that request

### Requirement: Health probe is a JSON success
GET `/health` SHALL respond with HTTP 200 and an `application/json` body whose `status` field is `ok`. The handler MUST NOT run ffmpeg or whisper-cli.

#### Scenario: GET health
- **WHEN** a client GETs `/health`
- **THEN** the response status is 200, the content type is JSON, and the body is `{"status": "ok"}`
