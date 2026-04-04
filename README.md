# Mesh LLM

![Mesh LLM logo](docs/mesh-llm-logo.svg)

![Mesh LLM](mesh.png)

Mesh LLM lets you pool spare GPU capacity across machines and expose the result as one OpenAI-compatible API.

If a model fits on one machine, it runs there. If it does not, Mesh LLM automatically spreads the work across the mesh:

- Dense models use pipeline parallelism.
- MoE models use expert sharding with zero cross-node inference traffic.
- Every node gets the same local API at `http://localhost:9337/v1`.

## Why people use it

- Run models larger than a single machine can hold.
- Turn a few uneven boxes into one shared inference pool.
- Give agents a local OpenAI-compatible endpoint instead of wiring each tool by hand.
- Keep the setup simple: start one node, add more later.

## Quick start

Install the latest release:

```bash
curl -fsSL https://raw.githubusercontent.com/michaelneale/mesh-llm/main/install.sh | bash
```

Then start a node:

```bash
mesh-llm --auto
```

That command:

- picks a suitable bundled backend for your machine
- downloads a model if needed
- joins the best public mesh
- exposes an OpenAI-compatible API at `http://localhost:9337/v1`
- starts the web console at `http://localhost:3131`

Check what is available:

```bash
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

Send a request:

```bash
curl http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"GLM-4.7-Flash-Q4_K_M","messages":[{"role":"user","content":"hello"}]}'
```

## Common workflows

### 1. Try the public mesh

```bash
mesh-llm --auto
```

This is the easiest way to see the system working end to end.

### 2. Start a private mesh

```bash
mesh-llm --model Qwen2.5-32B
```

This starts serving a model, opens the local API and console, and prints an invite token for other machines.

If you want the mesh to be discoverable via `--auto`, publish it:

```bash
mesh-llm --model Qwen2.5-32B --publish
```

### 3. Add another machine

```bash
mesh-llm --join <token>
```

Use `--client` if the machine should join without serving a model:

```bash
mesh-llm --client --join <token>
```

### 4. Create a named mesh for a group

```bash
mesh-llm --auto --model GLM-4.7-Flash-Q4_K_M --mesh-name "poker-night"
```

Everyone runs the same command. The first node creates the mesh, the rest discover and join it automatically.

### 5. Serve more than one model

```bash
mesh-llm --model Qwen2.5-32B --model GLM-4.7-Flash
```

Requests are routed by the `model` field:

```bash
curl localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"GLM-4.7-Flash-Q4_K_M","messages":[{"role":"user","content":"hello"}]}'
```

## How it works

Mesh LLM keeps the user-facing surface simple: talk to `localhost:9337`, pick a model, and let the mesh decide how to serve it.

- If a model fits on one machine, it runs there with no network overhead.
- If a dense model does not fit, layers are split across low-latency peers.
- If an MoE model does not fit, experts are split across nodes and requests are hash-routed for cache locality.
- Different nodes can serve different models at the same time.

Each node also exposes a management API and web console on port `3131`.

## Install notes

The installer currently targets macOS and Linux release bundles. Windows coming soon.

To force a specific bundled flavor during install:

```bash
curl -fsSL https://raw.githubusercontent.com/michaelneale/mesh-llm/main/install.sh | MESH_LLM_INSTALL_FLAVOR=vulkan bash
```

Installed release bundles use flavor-specific llama.cpp binaries:

- macOS: `metal`
- Linux: `cpu`, `cuda`, `rocm`, `vulkan`

To update a bundle install to the latest release:

```bash
mesh-llm update
```

If you build from source, always use `just`:

```bash
git clone https://github.com/michaelneale/mesh-llm
cd mesh-llm
just build
```

Requirements and backend-specific build notes are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Web console

When a node is running, open:

```text
http://localhost:3131
```

The console shows live topology, VRAM usage, loaded models, and built-in chat. It is backed by `/api/status` and `/api/events`.

You can also try the hosted demo:

**[mesh-llm-console.fly.dev](https://mesh-llm-console.fly.dev/)**

## More docs

- [docs/USAGE.md](docs/USAGE.md) for service installs, model commands, storage, and runtime control
- [docs/AGENTS.md](docs/AGENTS.md) for Goose, Claude Code, pi, OpenCode, curl, and blackboard usage
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for benchmark numbers and context
- [CONTRIBUTING.md](CONTRIBUTING.md) for local development and build workflows
- [PLUGINS.md](PLUGINS.md) for the plugin system and blackboard internals
- [mesh-llm/README.md](mesh-llm/README.md) for Rust crate structure
- [ROADMAP.md](ROADMAP.md) for future work

## Community

Join the [#mesh-llm channel on the Goose Discord](https://discord.gg/goose-oss) for discussion and support.
