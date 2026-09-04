# Use a base image with both C++ and Python
FROM python:3.12-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install C++ build tools and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential=12.9 \
    cmake=3.25.1-1 \
    ffmpeg=7:5.1.9-0+deb12u1 \
    git=1:2.39.5-0+deb12u3 \
    libsndfile1=1.2.0-1+deb12u1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to the root of the app
WORKDIR /app

# Copy the entire whisper directory
COPY whisper /app/whisper

# Set the working directory to the whisper directory
WORKDIR /app/whisper

# Download the model by running the script
RUN bash ./models/download-ggml-model.sh base.en

# GGML_NATIVE=ON compiles for the builder CPU (AVX-512 on GitHub runners),
# which SIGILLs on qemu and on older amd64 hosts. Keep a portable baseline.
RUN make CMAKE_ARGS="-DGGML_NATIVE=OFF"
RUN test -x /app/whisper/build/bin/whisper-cli

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --no-deps --require-hashes --only-binary :all: -r /app/requirements.txt

COPY app.py /app/

WORKDIR /app

RUN groupadd -g 10001 app && useradd --no-log-init -u 10001 -g app app
USER app

# Expose the port the app runs on
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health')"]

# Start the Flask application with Gunicorn, adjust the worker number and timeout as needed
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "300", "--chdir", "/app", "app:app"]
