# Skippy

Skippy is the standalone model-serving product in this workspace. It owns model
management, native runtimes, local inference, layer-package execution,
distributed stage serving, and an OpenAI-compatible frontend.

Skippy can run independently through its `skippy` CLI. The sibling
[Mesh product](../mesh/README.md) composes the same serving implementation with
peer discovery, routing, transport, plugins, SDKs, and a web console.

## Build and run

Run repository commands from the workspace root:

```bash
just skippy-cli-build
./target/debug/skippy --help
```

`skippy serve` selects a verified local native runtime or installs a compatible
release runtime when needed. A typical local invocation is:

```bash
./target/debug/skippy serve --model /path/to/model.gguf
```

See the [CLI guide](crates/skippy-cli/README.md) for runtime management, model
downloads, single-model serving, and explicit split-worker commands. See
[CONTRIBUTING.md](../CONTRIBUTING.md) for the workspace development workflow.

## Source layout

| Path | Ownership |
|---|---|
| [`crates/`](crates/) | CLI, model tooling, runtime, serving, protocol, topology, caching, metrics, and benchmarks |
| [`docs/`](docs/) | Operator guides, compatibility status, designs, experiments, and runbooks |
| [`evals/`](evals/) | Skippy performance and correctness evaluation workloads |
| [`scripts/`](scripts/) | Skippy-owned build, packaging, certification, and benchmark automation |
| [`llama_cpp/`](llama_cpp/) | Pinned llama.cpp source metadata and the durable Skippy ABI patch queue |

The main entry point is [`crates/skippy-cli`](crates/skippy-cli/). Runtime
execution lives in [`crates/skippy-runtime`](crates/skippy-runtime/), serving
coordination in [`crates/skippy-serving`](crates/skippy-serving/), and the
shared HTTP surface in
[`crates/skippy-inference-api`](crates/skippy-inference-api/).

## Key guides

- [Architecture and API boundaries](docs/ARCHITECTURE.md)
- [Configuration reference](docs/CONFIGURATION.md)
- [Split serving](docs/SKIPPY_SPLITS.md)
- [Layer package repositories](docs/LAYER_PACKAGE_REPOS.md)
- [Supported model families](docs/FAMILY_STATUS.md)
- [Non-chat models](docs/NON_CHAT_MODELS.md)
- [Data flow](docs/DATA_FLOW.md)
- [Topology planner](docs/TOPOLOGY_PLANNER.md)

Skippy is intentionally independent of Mesh. Both products share the root Cargo
workspace, lockfile, build infrastructure, and native working caches, while
product-owned code, documentation, scripts, and evaluations stay in this tree.
