# Mesh LLM and Exo

Use this comparison to understand how Mesh LLM and
[Exo](https://github.com/exo-explore/exo) differ across architecture, runtime,
networking, API compatibility, hardware support, and packaging models.

Both projects pool resources across machines to run models larger than a single
device can handle, but they diverge significantly in architecture, hardware
focus, and operational model.

## Summary

- **Exo** connects local devices into an AI cluster with automatic discovery,
  MLX/MLX-distributed backends, an event-sourcing architecture with Master/Worker/Runner
  systems, RDMA over Thunderbolt 5, and multiple API compatibility layers.
  Strongest on Apple Silicon where MLX and Thunderbolt RDMA provide a cohesive stack.
- **Mesh LLM** pools machines into private or published inference meshes with
  embedded Skippy/llama.cpp staged execution, QUIC/iroh peer-to-peer networking,
  gossip-based state dissemination, and package-backed GGUF layer splits for large models.
  Spans CUDA, ROCm, Vulkan, Metal, and CPU with a single binary.

Exo is a strong fit for local multi-device Apple Silicon clusters, especially
with Thunderbolt 5 RDMA and tensor parallelism. Mesh LLM is a strong fit for
operator-controlled distributed serving across heterogeneous hardware, private
and public mesh workflows, agent integrations, and large GGUF models that use
package-backed layer splits.

## Feature comparison

| Topic | Exo | Mesh LLM |
|---|---|---|
| Language & stack | Python, Svelte, Swift, Rust | Rust (host), TypeScript/React (web console) |
| Inference backend | MLX, MLX distributed (Apple Silicon) | Embedded Skippy/llama.cpp stage runtime |
| Model splitting | Tensor parallelism (all-reduce collectives), pipeline parallelism (layer-sequential), topology-aware auto placement with realtime device graph | Contiguous layer-range splits assigned by coordinator; each range loaded from a package-backed GGUF stage artifact |
| Architecture pattern | Event sourcing with Master (single writer), Worker, Runner (isolated process), API, Election systems; 5 topic channels | Decentralized mesh with gossip protocol, QUIC/iroh peer-to-peer connections, heartbeat-based peer state, Nostr published discovery |
| Hardware emphasis | Apple Silicon/macOS Tier 1 (M3 Ultra, M4 Pro, M5); Linux CPU Tier 3; Linux GPU (CUDA, Vulkan) under development | Release flavors for macOS, Linux, Windows, CUDA, ROCm, Vulkan, Metal, and CPU; all backends supported in the same binary |
| RDMA support | RDMA over Thunderbolt 5, day-0; macOS 26.2+; 99% latency reduction between TB5-connected devices | Not applicable (no RDMA transport) |
| Networking | libp2p, mDNS, bootstrap peers, `EXO_LIBP2P_NAMESPACE` for cluster isolation | QUIC/iroh mesh paths, Nostr discovery for published meshes, managed relays by default, private invite tokens |
| API compatibility | OpenAI Chat Completions, Claude Messages, OpenAI Responses, Ollama API | OpenAI-compatible `/v1/models`, chat completions, completions, and Responses through `openai-frontend` |
| Dashboard & ports | Local API and dashboard at `http://localhost:52415`; macOS app available; built-in image gen UI | Web console and management API at `http://localhost:3131`; inference API at `http://localhost:9337/v1` |
| Model artifact model | Full MLX models from HuggingFace; custom model cards with `trust_remote_code` toggle | Layer package repositories with `model-package.json` manifest, GGUF artifacts, validation, certification, HF Jobs publishing |
| Image generation | Native FLUX.1-dev and FLUX.1-Kontext-dev support; streaming partial images; OpenAI-compatible `/v1/images/generations` | Flash-MoE SSD plugin support; no native image generation |
| Agent integrations | Generic OpenAI-compatible client support | Built-in launchers for Goose, Claude Code, OpenCode, Pi; blackboard plugin; MCP server at `:3131/mcp` |
| Mixture-of-Agents | Not available | Experimental `model: "mesh"` fan-out across every model in the mesh with code-level arbitration |
| Instance lifecycle | Explicit create/delete instance model with placement preview, placement compute, and SSE-based await | Automatic model serving on the local node or routing to a peer that has the requested model |
| Operational model | Local cluster coordination across nearby devices with zero-config auto-discovery | Operator-controlled distributed serving across private or published meshes; owner-control plane with explicit endpoint bootstrap |
| Image models | `EXO_ENABLE_IMAGE_MODELS=true` enables FLUX-based image generation and editing | Not a built-in capability; plugin-extensible |
| Offline mode | `EXO_OFFLINE=true` runs from local models only | Full offline support via local catalog; no internet required for local models |
| License | Apache-2.0 | Apache-2.0 |

## Architecture

### Exo: Event sourcing with Master ordering

Exo uses an **event-sourcing architecture** with Erlang-style message passing
through a custom channel library. Five systems communicate over five topic
channels:

**Systems:**
- **Master** - single writer that executes placement, orders events, and
  maintains the DiskEventLog for crash recovery
- **Worker** - schedules work on a node, gathers system information
- **Runner** - executes inference jobs in an isolated process for fault tolerance
- **API** - Python web server with adapter pattern for multiple API formats
- **Election** - distributed master election for unstable networking conditions

**Topics:**
- **Commands** - API and Worker instruct the Master (placement, catchup)
- **Local Events** - all nodes write events; Master reads and orders them
- **Global Events** - Master writes ordered events; all nodes fold into `State`
- **Election Messages** - nodes negotiate master before cluster establishment
- **Connection Messages** - mDNS-discovered hardware connections

Events are past-tense side-effect records. Commands are imperative instructions.
The `apply()` function is a pure state-transition function. Nodes use
`OrderedBuffer` and NACK recovery to reconstruct identical state replicas.

**Key design principles** from Exo's RULES.md: referential transparency,
no hidden state, exhaustive type discipline with Pydantic, and eliminating
runtime error-handling overhead through type-level guarantees.

### Mesh LLM: Decentralized mesh with gossip

Mesh LLM uses a **decentralized mesh** with no single master or event log.
Nodes discover each other through:
- **Nostr discovery** for published/public meshes
- **Invite tokens** for private meshes
- **QUIC/iroh** for peer-to-peer transport

State is disseminated through **gossip protocol**: heartbeats carry peer state
updates, capability advertisements (vision, audio, multimodal, reasoning,
tool_use, moe), and mesh membership information. Each node independently
maintains its view of the mesh.

The **owner-control plane** uses an additive `mesh-llm-control/1` lane with
explicit endpoint bootstrap, separate from the public mesh plane for
mixed-version compatibility.

For model serving:
- A single-machine-fit model runs locally
- The mesh automatically routes requests by the `model` field to the peer serving it
- Skippy stage splits coordinate layer-range serving across multiple peers

### Key architectural differences

| Aspect | Exo | Mesh LLM |
|---|---|---|
| Coordination | Master as single writer | Decentralized, no single point |
| State model | Event-sourced, globally ordered | Gossip-based, eventually consistent |
| Fault tolerance | Master re-election, DiskEventLog replay | Mesh auto-heals through peer membership |
| Process model | Runner in isolated process | Embedded runtime in single process |
| API layer | Adapter pattern per API format | Unified openai-frontend crate |
| Type discipline | Python/Pydantic with strict types | Rust type system |

## Model splitting

### Exo

Exo supports three sharding strategies:

1. **Pipeline parallelism** - layers split sequentially across nodes
2. **Tensor parallelism** - weights partitioned within layers using all-reduce
   collectives; requires `supports_tensor=True` in model card and
   `hidden_size` divisible by node count
3. **CFG parallelism** - for image models, runs positive/negative prompt branches
   in parallel

Placement is computed through a **placement engine** that:
1. Identifies all cycles (paths) in the device topology graph
2. Filters by available memory vs model `storage_size`
3. Scores candidates with `_cycle_download_score` favoring cached models
4. Selects the smallest cycle to minimize communication overhead

Communication backends:
- **MLX Ring** - TCP/IP ring topology for standard networking
- **MLX Jaccl** - RDMA/Thunderbolt for high-performance connections

### Mesh LLM

Mesh LLM uses **Skippy stage splits**: contiguous layer ranges where each
range is a package-backed GGUF stage artifact.

The topology planner:
1. Inspects model architecture and available GPUs/memory across the mesh
2. Assigns contiguous layer ranges to each peer
3. Starts downstream stages first (last layer range first)
4. Waits for stage readiness
5. Publishes the stage-0 route for inference traffic

Stage artifacts live in **layer package repositories** with:
- `model-package.json` manifest
- GGUF fragments (one per stage range)
- Validation and certification metadata
- HF Jobs publishing workflow for automated building

### Splitting philosophy difference

Exo splits **within layers** (tensor parallelism) or between layer groups
(pipeline parallelism). The tensor approach requires all-reduce collectives
between every device on every layer step, which benefits from RDMA bandwidth.

Mesh LLM splits only **between layer groups** (contiguous ranges), avoiding
inter-device communication during inference. Each device decodes independently
once it receives its activation frame from the prior stage. This trades higher
per-device memory for zero inter-device communication during decode.

## Hardware and platform support

### Exo

| Tier | Platforms | Status |
|---|---|---|
| Tier 1 | Apple Silicon macOS (M3 Ultra, M4 Pro, M5 Macs) | Tested and maintained |
| Tier 2 | (unlisted) | Checked occasionally |
| Tier 3 | Linux CPU | Minimal support, should run |
| Planned | Linux CUDA (Nvidia DGX Spark), Linux Vulkan | Under development |
| Longer term | Windows CUDA, Windows CPU | Future consideration |

RDMA over Thunderbolt 5: macOS 26.2+ only. Requires specific hardware
(M4 Pro+, TB5 cables). Recovery mode boot required to enable.

### Mesh LLM

| Platform | Status |
|---|---|
| macOS (Metal) | Tier 1 (release bundles) |
| Linux (CPU) | Tier 1 (release bundles) |
| Linux (CUDA) | Tier 1 (release bundles for x86_64, aarch64, Blackwell) |
| Linux (ROCm) | Tier 1 (release bundles) |
| Linux (Vulkan) | Tier 1 (release bundles) |
| Windows (CPU) | Tier 1 (release bundles) |
| Windows (CUDA) | Tier 1 (release bundles) |
| Windows (ROCm) | Tier 1 (release bundles) |
| Windows (Vulkan) | Tier 1 (release bundles) |

## API surface

### Exo

Exo provides four API compatibility layers through an adapter pattern:

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/messages` | Claude Messages API |
| `POST /v1/responses` | OpenAI Responses API |
| `POST /ollama/api/chat`, `/generate` | Ollama API |
| `POST /v1/images/generations`, `/edits` | Image generation/editing |
| `POST /bench/chat/completions`, `/bench/images/*` | Benchmarking variants |

Exo also extends standard OpenAI parameters with `enable_thinking` for
thinking-capable models (DeepSeek V3.1, Qwen3, GLM-4.7), `top_k`, and
logprobs support.

Notable Exo API flows:
- **Instance lifecycle**: `POST /instance` -> `GET /instance/await?model_id=...` (SSE) -> inference -> `DELETE /instance/{id}`
- **Placement preview**: `GET /instance/previews?model_id=...` before creating an instance
- **Model search**: `GET /models/search?query=...` against HuggingFace

### Mesh LLM

Mesh LLM provides OpenAI-compatible endpoints through the `openai-frontend` crate:

| Endpoint | Purpose |
|---|---|
| `GET /v1/models` | List available models on the mesh |
| `POST /v1/chat/completions` | Chat completions (streaming + non-streaming) |
| `POST /v1/completions` | Text completions |
| `POST /v1/responses` | OpenAI Responses API |
| `POST /v1/cancel` | Cancel in-flight requests |

Additional management endpoints on the console port (`:3131`):
| Endpoint | Purpose |
|---|---|
| `GET /api/status` | Node/mesh status |
| `GET /api/events` | Event stream |
| `GET /api/discover` | Mesh discovery info |
| `GET /mcp` | MCP tool server (blackboard, etc.) |

Mesh LLM's API supports:
- Tool calling and structured outputs (JSON mode)
- Streaming with SSE
- MoA (`model: "mesh"`) fan-out across all models
- Model routing by peer availability

## Model sources and artifact management

### Exo

- Models are downloaded from HuggingFace as full MLX models
- `mlx-community` namespace is the primary source
- Custom models from any HuggingFace repo via `POST /models/add`
- Search HuggingFace directly via API
- Offline directory support with `EXO_MODELS_READ_ONLY_DIRS`
- Multiple writable model directories checked in order for free space
- Models must fit entirely in device memory (with optional splitting)

### Mesh LLM

- Models served as GGUF files locally or as layer packages
- Public model catalog at meshllm.cloud/catalog
- Catalog-backed model references (`hf://meshllm/<repo>@<rev>`)
- Layer package repositories with formal manifest and artifact layout
- Certification workflow for model families
- GGUF fragments sized for specific GPU memory budgets
- Local GGUF files work directly without packaging

## Community and ecosystem

| Aspect | Exo | Mesh LLM |
|---|---|---|
| Primary language | Python | Rust |
| Desktop app | macOS app (DMG), Homebrew cask | No desktop app |
| Agent launchers | Manual OpenAI client config | Built-in: goose, claude, opencode, pi |
| Plugin system | No formal plugin system | Plugin SDK, FFI plugins, MCP tools |
| Image generation | Built-in (FLUX) | Via plugins |
| MCP tools | None | Blackboard post/search/feed, plugin tools |
| CI/CD | GitHub Actions | GitHub Actions, release automation, xtask |
| Deployment | Nix flake, source, macOS app, uv | Release bundles per platform, just, Homebrew |
| Documentation | README + docs/ + DeepWiki | README + docs/ + website/ |

## Exo sources

Research for this comparison checked:

- Repository: <https://github.com/exo-explore/exo>
- README: <https://github.com/exo-explore/exo/blob/main/README.md>
- Architecture: <https://github.com/exo-explore/exo/blob/1e51dc89/docs/architecture.md>
- API: <https://github.com/exo-explore/exo/blob/main/docs/api.md>
- Platforms: <https://github.com/exo-explore/exo/blob/main/PLATFORMS.md>
- Rules: <https://github.com/exo-explore/exo/blob/main/RULES.md>
- TODO: <https://github.com/exo-explore/exo/blob/main/TODO.md>
- DeepWiki: Event sourcing, model placement, MLX backend
- Latest checked release: `v1.0.71`, published 2026-04-23.

## Mesh LLM docs for comparison

- [MESHES.md](MESHES.md) for private/public mesh behavior.
- [SKIPPY_SPLITS.md](SKIPPY_SPLITS.md) for package-backed stage splits.
- [LAYER_PACKAGE_REPOS.md](LAYER_PACKAGE_REPOS.md) for package publishing.
- [AGENTS.md](AGENTS.md) for agent/client integrations.
- [README.md](../README.md) for feature overview.
