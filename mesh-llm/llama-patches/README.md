# ⚠️ TEMPORARY — mesh hook C++ patches

**This directory is temporary.** It exists only on the `micn/virtual-llm`
branch so we can iterate on C++ and Rust changes in one repo / one PR.

## What's here

The 7 files that implement mesh hooks in llama-server. These overlay onto
the `upstream-latest` branch of `michaelneale/llama.cpp`.

## How it works

`just build` (via `scripts/build-mac.sh`) runs `sync.sh` after pulling
`upstream-latest`, copying these files into `llama.cpp/` before cmake runs.

## Workflow

1. Edit C++ in `llama.cpp/` directly, rebuild with `just build`
2. Save changes back: `./mesh-llm/llama-patches/sync.sh --save`
3. `git add mesh-llm/llama-patches/ && git commit`

## When done

1. Push the C++ changes to `mesh-hooks` branch on `michaelneale/llama.cpp`
2. Delete this directory
3. Remove the sync block from `scripts/build-mac.sh`
4. Switch `LLAMA_BRANCH` to `mesh-hooks` (or merge into `upstream-latest`)
