FROM python:3.12-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential=12.9 \
    cmake=3.25.1-1 \
    ffmpeg=7:5.1.9-0+deb12u1 \
    git=1:2.39.5-0+deb12u3 \
    libsndfile1=1.2.0-1+deb12u1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY whisper /app/whisper
WORKDIR /app/whisper

RUN bash ./models/download-ggml-model.sh base.en

# GGML_NATIVE=ON compiles for the builder CPU (AVX-512 on GitHub runners),
# which SIGILLs on qemu and on older amd64 hosts. Keep a portable baseline.
RUN make CMAKE_ARGS="-DGGML_NATIVE=OFF"
RUN test -x /app/whisper/build/bin/whisper-cli

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg=7:5.1.9-0+deb12u1 \
    libgomp1=12.2.0-14+deb12u1 \
    libsndfile1=1.2.0-1+deb12u1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/whisper/build/bin/whisper-cli /app/whisper/build/bin/whisper-cli
COPY --from=builder /app/whisper/build/bin/libwhisper.so.1 /app/whisper/build/bin/libwhisper.so.1
COPY --from=builder /app/whisper/build/bin/libggml.so.0 /app/whisper/build/bin/libggml.so.0
COPY --from=builder /app/whisper/build/bin/libggml-base.so.0 /app/whisper/build/bin/libggml-base.so.0
COPY --from=builder /app/whisper/build/bin/libggml-cpu.so.0 /app/whisper/build/bin/libggml-cpu.so.0
COPY --from=builder /app/whisper/models/ggml-base.en.bin /app/whisper/models/ggml-base.en.bin

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --no-deps --require-hashes --only-binary :all: -r /app/requirements.txt

COPY app.py /app/

RUN groupadd -g 10001 app && useradd --no-log-init -u 10001 -g app app
USER app

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health')"]

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "300", "--chdir", "/app", "app:app"]
