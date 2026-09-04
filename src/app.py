from flask import Flask, request, jsonify
import subprocess
import os
import tempfile
import re

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000


@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify(error="Request entity too large"), 413


def _unlink_quietly(path):
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


@app.route('/health', methods=['GET'])
def health():
    return jsonify(status="ok")


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    temp_path = None
    converted_path = None
    try:
        temp = tempfile.NamedTemporaryFile(delete=False)
        file.save(temp.name)
        temp.close()
        temp_path = temp.name

        converted_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        converted_temp.close()
        converted_path = converted_temp.name

        try:
            subprocess.run([
                "ffmpeg", "-y", "-nostdin", "-i", temp_path,
                "-ar", "16000",
                "-ac", "1",
                "-vn",
                converted_path
            ], check=True)
        except FileNotFoundError:
            app.logger.error("ffmpeg executable was not found")
            return jsonify(error="Error in transcription"), 500
        except subprocess.CalledProcessError as exc:
            app.logger.error("ffmpeg failed with return code %s", exc.returncode)
            return jsonify(error="Error in transcription"), 400

        try:
            result = subprocess.run([
                "/app/whisper/build/bin/whisper-cli",
                "-f", converted_path,
                "--model", "/app/whisper/models/ggml-base.en.bin",
                "--no-gpu",
                "--no-prints",
            ], capture_output=True, text=True)
        except FileNotFoundError:
            app.logger.error("whisper-cli executable was not found")
            return jsonify(error="Error in transcription"), 500

        app.logger.info(f"Return code: {result.returncode}")
        if result.returncode != 0:
            app.logger.error(f"Error output: {result.stderr}")
            return jsonify(error="Error in transcription"), 500

        return jsonify(transcription=parse_transcription(result.stdout))
    finally:
        _unlink_quietly(temp_path)
        _unlink_quietly(converted_path)

def parse_transcription(transcription):
    pattern = re.compile(r'\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\](.*?)\n', re.DOTALL)
    matches = pattern.findall(transcription)
    parsed_transcription = []
    for start_time, end_time, text in matches:
        text = text.strip()
        entry = {
            "start_time": start_time,
            "end_time": end_time,
            "text": text
        }
        parsed_transcription.append(entry)
    return parsed_transcription

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
