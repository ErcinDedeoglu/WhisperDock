import io
import os
import shutil
import tempfile
import unittest
import wave

from app import app, parse_transcription

WHISPER_CLI = "/app/whisper/build/bin/whisper-cli"
README_STDOUT = (
    "[00:00:00.000 --> 00:00:03.000]  Welcome to our speech-to-text service.\n"
    "[00:00:03.500 --> 00:00:05.000]  This is a sample transcription.\n"
)


def _silence_wav_bytes(frames=1600):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


class TranscribeApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self._tempdir = tempfile.mkdtemp(prefix="whisperdock-test-")
        self._prev_tempdir = tempfile.tempdir
        tempfile.tempdir = self._tempdir

    def tearDown(self):
        tempfile.tempdir = self._prev_tempdir
        shutil.rmtree(self._tempdir, ignore_errors=True)

    def test_missing_file_returns_json_400(self):
        response = self.client.post("/transcribe")
        self.assertEqual(response.status_code, 400)
        self.assertIn("application/json", response.content_type)
        self.assertEqual(response.get_json(), {"error": "No file part"})

    def test_empty_filename_returns_json_400(self):
        response = self.client.post(
            "/transcribe",
            data={"file": (io.BytesIO(b"x"), "")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("application/json", response.content_type)
        self.assertEqual(response.get_json(), {"error": "No selected file"})

    def test_garbage_audio_returns_json_error_and_leaves_no_temp_files(self):
        before = set(os.listdir(self._tempdir))
        response = self.client.post(
            "/transcribe",
            data={"file": (io.BytesIO(b"not-audio"), "bad.wav")},
        )
        after = set(os.listdir(self._tempdir))
        leaked = sorted(after - before)
        body = response.get_data(as_text=True)
        print(
            "GARBAGE_AUDIO status=%s type=%s leaked_tmp=%s body=%r"
            % (response.status_code, response.content_type, leaked, body)
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("application/json", response.content_type)
        self.assertNotIn("text/html", response.content_type)
        payload = response.get_json()
        self.assertIsInstance(payload, dict)
        self.assertIn("error", payload)
        self.assertTrue(payload["error"])
        self.assertEqual(leaked, [])

    def test_oversized_upload_returns_json_413_and_leaves_no_temp_files(self):
        limit = app.config["MAX_CONTENT_LENGTH"]
        self.assertEqual(limit, 16 * 1000 * 1000)
        before = set(os.listdir(self._tempdir))
        response = self.client.post(
            "/transcribe",
            data={"file": (io.BytesIO(b"x" * (limit + 1)), "huge.wav")},
        )
        after = set(os.listdir(self._tempdir))
        leaked = sorted(after - before)
        self.assertEqual(response.status_code, 413)
        self.assertIn("application/json", response.content_type)
        self.assertNotIn("text/html", response.content_type)
        payload = response.get_json()
        self.assertIsInstance(payload, dict)
        self.assertIn("error", payload)
        self.assertTrue(payload["error"])
        self.assertEqual(leaked, [])

    def test_missing_whisper_cli_returns_json_500(self):
        if os.path.exists(WHISPER_CLI):
            self.skipTest("whisper-cli is present at %s" % WHISPER_CLI)
        response = self.client.post(
            "/transcribe",
            data={"file": (io.BytesIO(_silence_wav_bytes()), "silence.wav")},
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn("application/json", response.content_type)
        self.assertNotIn("text/html", response.content_type)
        self.assertEqual(response.get_json(), {"error": "Error in transcription"})

    def test_parse_transcription_readme_segments(self):
        parsed = parse_transcription(README_STDOUT)
        self.assertEqual(
            parsed,
            [
                {
                    "start_time": "00:00:00.000",
                    "end_time": "00:00:03.000",
                    "text": "Welcome to our speech-to-text service.",
                },
                {
                    "start_time": "00:00:03.500",
                    "end_time": "00:00:05.000",
                    "text": "This is a sample transcription.",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
