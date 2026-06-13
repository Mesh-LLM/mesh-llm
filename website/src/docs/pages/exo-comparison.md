---
title: Exo comparison
---

# Mesh and exo

Mesh and [exo](https://github.com/exo-explore/exo) both explore running frontier AI across
more than one machine, but they approach the problem from different angles.

Exo connects local devices into an AI cluster using automatic discovery, MLX inference
backends, event-sourcing architecture, and multiple API compatibility layers. It is
particularly well-suited to Apple Silicon environments where Thunderbolt RDMA and
MLX provide a cohesive hardware-software stack.

Mesh pools machines into private or published meshes with embedded
llama.cpp/Skippy staged execution, QUIC/iroh networking, package-backed GGUF
layer splits, and one OpenAI-compatible endpoint served through a mesh of peers.

## Quick comparison

| Area | Exo | Mesh |
|---|---|---|
| **Language** | Python, Svelte, Swift | Rust, TypeScript/React (web console) |
| **Inference backend** | MLX, MLX distributed | Embedded Skippy/llama.cpp stage runtime |
| **Model splitting** | Tensor parallelism, pipeline parallelism, topology-aware auto placement | Contiguous layer-range splits with package-backed stage artifacts |
| **Hardware focus** | Apple Silicon/macOS (Tier 1), Linux CPU (Tier 3) | macOS, Linux, Windows, CUDA, ROCm, Vulkan, Metal, CPU |
| **Networking** | libp2p, mDNS, bootstrap peers, namespace isolation | QUIC/iroh, Nostr published discovery, managed relays, invite tokens |
| **Architecture** | Event sourcing with Master/Worker/Runner/Election systems | Decentralized mesh with gossip protocol |
| **API compatibility** | OpenAI Chat Completions, Claude Messages, OpenAI Responses, Ollama | OpenAI-compatible /v1/models, chat completions, completions, Responses |
| **Dashboard** | Built-in dashboard at localhost:52415, macOS app | Web console at localhost:3131, inference API at localhost:9337/v1 |
| **Model artifacts** | Full MLX models from HuggingFace | Layer package repos with GGUF fragments, manifests, validation |
| **Image generation** | FLUX models, img2img editing | Via plugin architecture |
| **License** | Apache-2.0 | Apache-2.0 |

## When to use each

- **Use Mesh when you want:**
  - One OpenAI-compatible endpoint across a mesh of machines, no matter what hardware each runs
  - Cross-platform support spanning CUDA, ROCm, Vulkan, Metal, and CPU
  - A public or private mesh with catalog-backed model references and operator controls
  - Layer packages for very large GGUF models that won't fit on a single device
  - Agent and tool integrations with built-in launchers for Goose, Claude Code, OpenCode, and Pi
  - Mixture-of-Agents routing that fans out a single prompt across every model in the mesh
- **Use exo when you want:**
  - An Apple Silicon-native cluster with automatic device discovery and zero-config setup
  - RDMA over Thunderbolt 5 for high-bandwidth, low-latency multi-device inference
  - Tensor parallelism and pipeline parallelism across devices
  - Multiple API compatibility layers including Claude Messages and Ollama
  - A macOS desktop app and a local dashboard
  - Image generation with FLUX models
  - An event-sourcing architecture with a Master/Worker/Runner pattern

## Architecture differences

Exo uses **event sourcing** with five systems (Master, Worker, Runner, API, Election)
communicating through five topic channels. The Master acts as a single writer that
orders events and broadcasts a globally consistent state. This design makes data
flow explicit and enables deterministic state everywhere at the cost of a
single-point writer bottleneck.

Mesh uses a **decentralized mesh** with QUIC/iroh peer-to-peer connections and
gossip protocol for state dissemination. There is no single master; nodes coordinate
through peer membership, heartbeats, and capability advertisement. Published meshes
use Nostr for public discovery while private meshes stay invite-token based.

## Splitting philosophy

Exo can split a model by **tensor parallelism** (partitioning individual layer
weights across devices with all-reduce collectives) or **pipeline parallelism**
(splitting layers sequentially). Placement is computed automatically based on a
real-time topology that considers device memory and network bandwidth between
each link. Both pipeline and tensor splits use MLX distributed for communication.

Mesh splits by assigning **contiguous layer ranges** to each peer, with each
range loaded from a package-backed stage artifact (a GGUF fragment). A coordinator
plans the topology, starts downstream stages first, waits for readiness, then
publishes the stage-0 route. Stage artifacts are distributed through package
repositories with manifests, validation, and certification.

## Hardware and platform support

Exo has a narrow but deep hardware focus: Apple Silicon macOS is Tier 1, with
careful support for M3 Ultra, M4 Pro, and M5 Macs. Linux CPU is Tier 3 (should run
without crashing). Linux GPU support (CUDA, Vulkan) is under development with no
published timeline. The RDMA over Thunderbolt 5 feature is macOS 26.2+ only and
requires specific hardware (M4 Pro Mac Mini, M4 Max Mac Studio, M3 Ultra Mac
Studio, M4 Max MacBook Pro).

Mesh ships release flavors for macOS, Linux, Windows, CUDA, ROCm, Vulkan, Metal,
and CPU. Every release includes multiple platform bundles. The embedded llama.cpp
stage runtime runs on any supported backend, and the mesh protocol works across
heterogeneous hardware - a CUDA node can serve stage artifacts to a Metal node
in the same mesh.

## Community and ecosystem

Exo has a significantly larger community (45k+ stars, active Discord). It offers
a macOS desktop app, supports image generation natively, and has broader API
compatibility (Claude, Ollama formats). The project is Python-based, making it
more accessible for developers who want to contribute or extend it.

Mesh is newer (1.1k stars) but has invested in a formal layer-package ecosystem,
agent integrations (Goose, Claude Code, OpenCode, Pi launchers), a public
model catalog, and a plugin system (Flash-MoE, telemetry, blackboard). Its Rust
foundation gives it performance characteristics suited for production serving.

## Summary

Exo is the right choice when your environment is Apple Silicon, you want automatic
local discovery, and you need the lowest-latency multi-device inference through
RDMA. Mesh is the right choice when you need cross-platform serving, private or
published meshes with operator controls, agent integrations, and package-backed
layer splits for very large models.
