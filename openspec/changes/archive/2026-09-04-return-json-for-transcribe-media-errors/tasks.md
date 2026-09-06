## 1. Tests that fail on current behavior

- [x] 1.1 Add `src/test_app.py` using stdlib `unittest` and Flask `test_client` against shipped `app` (no mocks of `transcribe_audio` or `parse_transcription`). Cover missing file (400 JSON `No file part`), empty filename (400 JSON `No selected file`), garbage `bad.wav` (400 JSON with `error`, content-type not HTML, zero new `tmp*` files in `tempfile.gettempdir()`), missing whisper-cli after a valid tiny WAV (500 JSON `Error in transcription` when the binary is absent), and `parse_transcription` on the README two-segment dump. Verify: `python3 -m unittest discover -s src -p 'test_*.py' -v` is red on garbage-audio (HTML 500 / leaked temps) before the handler change.

## 2. Handler fix

- [x] 2.1 In `src/app.py` `transcribe_audio`, catch ffmpeg `CalledProcessError` as HTTP 400 JSON `error`, catch missing whisper-cli `FileNotFoundError` and non-zero whisper as HTTP 500 JSON `{"error": "Error in transcription"}`, and unlink both temp paths in `finally`. Do not change parse regex, whisper flags, or Dockerfile. Verify: the same unittest command exits 0 and garbage-audio asserts JSON + empty leak list.

## 3. Verify

- [x] 3.1 Re-run `python3 -m unittest discover -s src -p 'test_*.py' -v` (exit 0). Confirm holdout missing-file still 400 JSON. Launch Flask test-client garbage POST twice and confirm both runs show JSON `error` and `leaked_tmp=[]`. Chrome: N/A — no UI.
