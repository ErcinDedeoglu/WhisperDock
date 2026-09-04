## ADDED Requirements

### Requirement: Oversized upload is a JSON client error
When the request body exceeds 16 000 000 bytes, the service SHALL respond with HTTP 413 and an `application/json` body that includes a non-empty `error` field. The response MUST NOT be HTML. The service MUST NOT persist new temporary media files for that request and MUST NOT start media conversion.

#### Scenario: Body larger than the upload limit
- **WHEN** a client POSTs `/transcribe` with a `file` part whose request body is larger than 16 000 000 bytes
- **THEN** the response status is 413, the content type is JSON, the JSON object contains an `error` field, and the process temp directory contains no new `tmp*` files created for that request
