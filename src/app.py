import socket
from flask import Flask, request, jsonify
import subprocess
import os
import tempfile
import re

app = Flask(__name__)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    # Save the uploaded file to a temporary file
    temp = tempfile.NamedTemporaryFile(delete=False)
    file.save(temp.name)
    temp.close()

    # Convert the uploaded media to 16kHz mono audio WAV file using ffmpeg
    converted_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    subprocess.run([
        "ffmpeg", "-y", "-i", temp.name,
        "-ar", "16000",
        "-ac", "1",
        "-vn",
        converted_temp.name
    ], check=True)
    os.remove(temp.name)  # Remove the original file as it's no longer needed

    # Call the main executable
    result = subprocess.run(["/app/whisper/main", "-f", converted_temp.name, "--model", "/app/whisper/models/ggml-base.en.bin"], capture_output=True, text=True)

    # Log the return code and stderr
    app.logger.info(f"Return code: {result.returncode}")
    if result.returncode != 0:
        app.logger.error(f"Error output: {result.stderr}")

    transcription = result.stdout if result.returncode == 0 else "Error in transcription"

    # Remove the temporary converted file
    os.remove(converted_temp.name)

    if result.returncode == 0:
        transcription = parse_transcription(result.stdout)
        return jsonify(transcription=transcription)
    else:
        return jsonify(error="Error in transcription"), 50

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
