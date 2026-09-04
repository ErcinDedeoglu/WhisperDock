## Context

Runtime apt currently installs Debian `ffmpeg=7:5.1.9-0+deb12u1`, which Depends on `libavdevice59` → Mesa/LLVM. App argv is `ffmpeg -y -nostdin -i … -ar 16000 -ac 1 -vn`. whisper-cli needs `libgomp1` only. See proposal.md.

## Goals / Non-Goals

**Goals:**
- Runtime ffmpeg without Debian video stack.

**Non-Goals:**
- Changing builder apt. Shipping ffprobe.

## Decisions

1. **`COPY --from=mwader/static-ffmpeg:9.0.1 /ffmpeg /usr/local/bin/ffmpeg`**
   - Official README pin; multi-arch; static; `ffmpeg` stays on PATH.
   - Rejected johnvansickle (amd64-only). Rejected compiling ffmpeg.

2. **Runtime apt only `libgomp1=12.2.0-14+deb12u1`**
   - Drop debian ffmpeg and libsndfile1 (whisper-cli ldd does not need libsndfile).

## Risks / Trade-offs

- [Third-party binary] → pin 9.0.1; rollback is one FROM/COPY revert.
- [GHA multi-arch] → image is amd64+arm64 since 5.0.1-3.

## Migration Plan

Edit runtime stage. Rebuild `whisperdock-local:pinned`. Rollback: restore debian ffmpeg apt.

## Open Questions

None.
