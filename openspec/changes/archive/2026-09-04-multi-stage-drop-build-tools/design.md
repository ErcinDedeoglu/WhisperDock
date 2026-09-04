## Context

ldd whisper-cli needs libs in `/app/whisper/build/bin` plus libgomp1. Image 2.34 GB.

## Decisions

1. **FROM … AS builder** through `test -x whisper-cli`.
2. **Runtime** python:3.12-bookworm; apt ffmpeg + libsndfile1 + libgomp1 (pinned); COPY whisper-cli, libwhisper.so.1, libggml.so.0, libggml-base.so.0, libggml-cpu.so.0, ggml-base.en.bin.

## Risks

- [missing .so] → whisper-cli fails to start; add the file.
