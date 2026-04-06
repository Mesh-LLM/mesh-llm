# Usage Guide

This page keeps the longer operational reference out of the top-level README.

## Installation details

Install the latest release bundle:

```bash
curl -fsSL https://raw.githubusercontent.com/michaelneale/mesh-llm/main/install.sh | bash
```

The installer probes your machine, recommends a flavor, and asks what to install.

For a non-interactive install, set the flavor explicitly:

```bash
curl -fsSL https://raw.githubusercontent.com/michaelneale/mesh-llm/main/install.sh | MESH_LLM_INSTALL_FLAVOR=vulkan bash
```

Release bundles install flavor-specific llama.cpp binaries:

- macOS: `rpc-server-metal`, `llama-server-metal`
- Linux CPU: `rpc-server-cpu`, `llama-server-cpu`
- Linux CUDA: `rpc-server-cuda`, `llama-server-cuda`
- Linux ROCm: `rpc-server-rocm`, `llama-server-rocm`
- Linux Vulkan: `rpc-server-vulkan`, `llama-server-vulkan`

If you keep more than one flavor in the same `bin` directory, choose one explicitly:

```bash
mesh-llm serve --llama-flavor vulkan --model Qwen2.5-32B
```

Source builds must use `just`:

```bash
git clone https://github.com/michaelneale/mesh-llm
cd mesh-llm
just build
```

Requirements:

- `just`
- `cmake`
- Rust toolchain
- Node.js 24 + npm

Backend-specific notes:

- NVIDIA builds require `nvcc`
- AMD builds require ROCm/HIP
- Vulkan builds require the Vulkan development files and `glslc`
- CPU-only and Jetson/Tegra are also supported

For full build details, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Common commands

```bash
mesh-llm serve --auto
mesh-llm serve --model Qwen2.5-32B
mesh-llm serve --join <token>
mesh-llm client --auto
mesh-llm gpus
mesh-llm discover
```

If you run `mesh-llm` with no arguments, it prints `--help` and exits. It does not start the console or bind ports until you choose a mode.

## Background service

To install Mesh LLM as a per-user background service:

```bash
curl -fsSL https://raw.githubusercontent.com/michaelneale/mesh-llm/main/install.sh | bash -s -- --service
```

To seed the service with a custom startup command on first install:

```bash
curl -fsSL https://raw.githubusercontent.com/michaelneale/mesh-llm/main/install.sh | bash -s -- --service --service-args 'serve --model Qwen2.5-3B'
```

Service installs are user-scoped:

- macOS installs a `launchd` agent at `~/Library/LaunchAgents/com.mesh-llm.mesh-llm.plist`
- Linux installs a `systemd --user` unit at `~/.config/systemd/user/mesh-llm.service`
- Shared environment config lives in `~/.config/mesh-llm/service.env`

Platform behavior:

- macOS reads startup args from `~/.config/mesh-llm/service.args`
- Linux writes the `mesh-llm` argv directly into `ExecStart=`

Optional shared environment file example:

```text
MESH_LLM_NO_SELF_UPDATE=1
```

If you edit the Linux unit manually:

```bash
systemctl --user daemon-reload
systemctl --user restart mesh-llm.service
```

If you want the service to survive reboot before login:

```bash
sudo loginctl enable-linger "$USER"
```

## Model catalog

List or fetch models from the built-in catalog:

```bash
mesh-llm download
mesh-llm download 32b
mesh-llm download 72b --draft
```

Draft pairings for speculative decoding:

| Model | Size | Draft | Draft size |
|---|---|---|---|
| Qwen2.5 (3B/7B/14B/32B/72B) | 2-47GB | Qwen2.5-0.5B | 491MB |
| Qwen3-32B | 20GB | Qwen3-0.6B | 397MB |
| Llama-3.3-70B | 43GB | Llama-3.2-1B | 760MB |
| Gemma-3-27B | 17GB | Gemma-3-1B | 780MB |

## Specifying models

`mesh-llm serve --model` accepts several formats. Hugging Face-backed models are cached in the standard Hugging Face cache on first use.

```bash
mesh-llm serve --model Qwen3-8B
mesh-llm serve --model Qwen3-8B-Q4_K_M
mesh-llm serve --model https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
mesh-llm serve --model bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf
mesh-llm serve --gguf ~/my-models/custom-model.gguf
mesh-llm serve --gguf ~/my-models/qwen3.5-4b.gguf --mmproj ~/my-models/mmproj-BF16.gguf
```

Useful model commands:

```bash
mesh-llm models recommended
mesh-llm models installed
mesh-llm models search qwen 8b
mesh-llm models search --catalog qwen
mesh-llm models show Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf
mesh-llm models download Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf
mesh-llm models migrate
mesh-llm models migrate --apply
mesh-llm models updates --check
mesh-llm models updates --all
mesh-llm models updates Qwen/Qwen3-8B-GGUF
```

## Model storage

- Hugging Face repo snapshots are the canonical managed model store.
- Flat `~/.models/` storage is no longer scanned for managed models.
- If you still have legacy files there, use `mesh-llm models migrate --apply`.
- Arbitrary local GGUF files still work through `mesh-llm serve --gguf`.
- MoE split artifacts are cached under `~/.cache/mesh-llm/splits/`.

## Inspect local GPUs

```bash
mesh-llm gpus
```

This prints the local GPU inventory with stable IDs, backend device names, VRAM, unified-memory status, and cached bandwidth if a benchmark fingerprint is already present.

## Local runtime control

Stage one supports local-only hot load and unload on a running node.

```bash
mesh-llm load Llama-3.2-1B-Instruct-Q4_K_M
mesh-llm unload Llama-3.2-1B-Instruct-Q4_K_M
mesh-llm status
```

Management API endpoints:

```bash
curl localhost:3131/api/runtime
curl localhost:3131/api/runtime/processes
curl -X POST localhost:3131/api/runtime/models \
  -H 'Content-Type: application/json' \
  -d '{"model":"Llama-3.2-1B-Instruct-Q4_K_M"}'
curl -X DELETE localhost:3131/api/runtime/models/Llama-3.2-1B-Instruct-Q4_K_M
```

This stage is intentionally node-local. Mesh-wide rebalancing and distributed load/unload come later.
