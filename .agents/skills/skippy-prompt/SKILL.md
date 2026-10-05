---
name: skippy-prompt
description: Use this skill when running or debugging interactive Skippy prompts against staged serving, including lab sync, native builds, stage startup, the HTTP prompt REPL, and process lifecycle.
metadata:
  short-description: Run prompt-owned staged workflows
---

# skippy-prompt

Use this skill for prompt-owned staged workflows. The skill is the launcher:
Codex orchestrates sync, host-native builds, stage config generation, process
startup, observation, prompt driving, and teardown.

## Ownership Rules

- The machine where the user asks to launch prompt is always `stage-0`.
- Remote hosts are `stage-1..N` in the order provided by the user.
- Bring down any running `mesh-llm` serving on the chosen nodes before starting
  prompt-owned stage servers.
- Do not bring back standalone `kv-server` or `ngram-pool`.
- Use `$HOME/tmp` for run roots, source syncs, logs, and bundles. Avoid `/tmp`
  unless the user explicitly asks for it.
- Public OpenAI compatibility belongs in `skippy-inference-api`. The
  interactive client uses stage-0's OpenAI endpoint; raw protocol and cache
  checks belong in `skippy-correctness`.
- Start stages with `skippy serve --config <stage.json> --stage-transport binary`.
  Use `--worker-only` on downstream stages. Stage 0 exposes the public API by
  default; attach with `skippy prompt --endpoint` or add `--prompt` to stage 0.

## Launch Workflow

1. Confirm repo state, branch, commit, model ref/path, hosts, desired layer
   ranges, context size, and prompt mode.
2. Stop existing mesh/runtime processes on every selected host:
   `mesh-llm stop` first, then verify with `ps`; use `pkill -f` only if the
   scoped stop path fails.
3. Rsync the current source tree to each remote host under
   `$HOME/tmp/mesh-llm-prompt-src/<branch-or-sha>/`, excluding build outputs and
   caches (`target/`, `.git/`, `.deps/llama-build/`, UI `node_modules/`).
4. Detect each host:
   `uname -s`, `uname -m`, GPU inventory, compiler/runtime availability, and
   existing llama build cache.
5. Choose the best backend per host:
   - macOS: Metal.
   - Linux NVIDIA with CUDA toolchain: CUDA. Use this for `white.local` unless
     CUDA is genuinely unavailable.
   - Linux AMD with ROCm toolchain: ROCm.
   - Vulkan-capable Linux without CUDA/ROCm: Vulkan.
   - CPU only as a last resort or explicit user request.
6. Build on each host with repo-native `just` targets. Use `just build` on
   macOS and `just release-runtime-build <backend>` on Linux when UI rebuild
   is unnecessary. Do not hand-roll cargo/cmake build sequences.
7. Materialize or locate model/package inputs on the launcher. If the source
   model only exists locally, rsync package/materialized stage inputs to remote
   hosts.
8. Start final stage first, then upstream stages, ending with local `stage-0`.
   Use foreground TTY SSH for first repro/debug runs and tee logs under
   `$HOME/tmp/skippy-prompt-runs/<run-id>/`.
9. Wait for readiness of every stage, then attach `skippy prompt --endpoint`
   from the launcher to the stage-0 OpenAI endpoint.
10. Keep process handles or SSH sessions observable. Do not report success until
    stage servers are running and a prompt request has been attempted or the user
    explicitly only asked for startup.

## Host Detection Commands

Use these as probes, adapting for the host OS:

```bash
uname -s
uname -m
command -v nvidia-smi && nvidia-smi -L
command -v nvcc && nvcc --version
command -v rocminfo && rocminfo
command -v vulkaninfo && vulkaninfo --summary
system_profiler SPDisplaysDataType
```

Backend selection is evidence-based. If a preferred backend fails, capture the
failure and either fix the toolchain or clearly say why the fallback is being
used.

## Commands

Before using source-repo prompt commands, verify the crate exists here:

```bash
cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name' | sort
```

Expected prompt-owned binaries are:

```text
skippy
skippy-correctness
skippy-package-builder
metrics-server
```

For remote long-running stages, use the `remote-observable-process` skill:
allocate a TTY, use an interactive login shell, tee logs, and keep the session
open while proving the topology.
