FROM python:3.12-bookworm@sha256:581429e3df12d76e6af4be5ab7d0e7fc2013eb57dc23d2de691411c8efdbb970 AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential=12.9 \
    cmake=3.25.1-1 \
    git=1:2.39.5-0+deb12u3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY whisper /app/whisper
WORKDIR /app/whisper

RUN bash ./models/download-ggml-model.sh base.en

# GGML_NATIVE=ON compiles for the builder CPU (AVX-512 on GitHub runners),
# which SIGILLs on qemu and on older amd64 hosts. Keep a portable baseline.
RUN make CMAKE_ARGS="-DGGML_NATIVE=OFF"
RUN test -x /app/whisper/build/bin/whisper-cli

FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1=12.2.0-14+deb12u1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=mwader/static-ffmpeg:9.0.1@sha256:54e55b0cb8f672870fc38ceb2e6c411855cb3b39c505f5f3b2505ee01ed5f2b7 /ffmpeg /usr/local/bin/ffmpeg

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
