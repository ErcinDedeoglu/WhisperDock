## Context

Two RUNs: install then `apt-get clean && rm lists`. Missing `--no-install-recommends`.

## Decisions

1. **One RUN** with `--no-install-recommends` and `rm -rf /var/lib/apt/lists/*`.
2. **Keep package set** (build-essential cmake git libsndfile1 ffmpeg). Sort alphabetically.

## Risks

- [Missing recommended lib for ffmpeg] → rebuild fails; add the package explicitly.
