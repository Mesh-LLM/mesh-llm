# Mesh LLM

Mesh is the distributed product in this workspace. It adds peer discovery,
admission, encrypted transport, routing, plugins, management APIs, SDKs, and the
web console around Skippy's model-serving runtime.

Every Mesh node exposes the same OpenAI-compatible API. Requests can run on the
local machine, route to another node, or use a Skippy stage split when a model
does not fit on one machine.

For installation and a first request, start with the
[workspace README](../README.md). The [documentation hub](docs/README.md) links
the operator, architecture, API, plugin, and SDK guides.

## Build and run

Run repository commands from the workspace root:

```bash
just build
./target/debug/mesh-llm --help
./target/debug/mesh-llm serve --auto
```

`just build` produces the backend-neutral Mesh host and an adjacent packaged
native runtime for local development. Use `just release-build` for serious
testing or deployment. See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full
development workflow.

## Source layout

| Path | Ownership |
|---|---|
| [`crates/`](crates/) | Mesh binaries, runtimes, networking, APIs, plugins, UI, and shared libraries |
| [`docs/`](docs/) | Authored Mesh guides, designs, specifications, and plans |
| [`sdk/`](sdk/) | Node.js, Swift, Kotlin, and other SDK packaging |
| [`deploy/`](deploy/) | Service and platform deployment assets |
| [`evals/`](evals/) | Mesh-level evaluations and benchmark scenarios |
| [`scripts/`](scripts/) | Mesh-owned build, test, release, and maintenance scripts |
| [`website/`](website/) | Public website source; generated output is written to root `docs/` |

The shipped `mesh-llm` binary is assembled in
[`crates/mesh-llm`](crates/mesh-llm/). Most runtime behavior lives in
[`crates/mesh-llm-host-runtime`](crates/mesh-llm-host-runtime/); the binary crate
keeps only CLI dispatch and runtime handoff wiring.

## Key guides

- [Using and operating Mesh](docs/USAGE.md)
- [Public and private meshes](docs/MESHES.md)
- [CLI reference](docs/CLI.md)
- [Architecture](docs/design/DESIGN.md)
- [Plugin development](docs/plugins/README.md)
- [SDK guide](docs/SDK.md)

Mesh depends on the sibling [Skippy product](../skippy/README.md) for model
management, native inference, split execution, and OpenAI serving. Skippy does
not depend on Mesh crates or plugin hosts.
