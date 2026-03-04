# Contributing

This file covers local build and development workflows for this repository.

## Prerequisites

- macOS Apple Silicon for the default local workflow
- `just`
- `cmake`
- Rust toolchain (`cargo`)
- Node.js + npm (for UI development)

## Build from source

Build everything (llama.cpp fork, mesh binary, and UI production build):

```bash
just build
```

Create a portable bundle:

```bash
just bundle
```

## UI development workflow

Use this two-terminal flow for UI development.

Terminal A (run `mesh-llm` yourself):

```bash
mesh-llm --port 9337 --console 3131
```

If `mesh-llm` is not on your `PATH`:

```bash
./mesh-llm/target/release/mesh-llm --port 9337 --console 3131
```

Terminal B (run Vite with HMR):

```bash
just ui-dev
```

Open:

```text
http://127.0.0.1:5173
```

`ui-dev` defaults:

- Serves on `127.0.0.1:5173`
- Proxies `/api/*` to `http://127.0.0.1:3131`

Overrides:

```bash
# Different backend API origin for /api proxy
just ui-dev http://127.0.0.1:4141

# Different Vite dev port
just ui-dev http://127.0.0.1:3131 5174
```

## Useful commands

```bash
just stop             # stop mesh/rpc/llama processes
just test             # quick test against :9337
just --list           # list all recipes
```
