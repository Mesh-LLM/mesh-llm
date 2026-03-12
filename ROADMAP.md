# Roadmap

High-level directions for mesh-llm. Not promises — just things we're thinking about.

## Smart model router ✅ (Phase 1)

Implemented. Heuristic classifier detects Code/Reasoning/Chat/Creative/ToolCall with Quick/Moderate/Deep complexity. Task-dominant scoring ensures the right model handles each request. Tool capability is a hard filter. Multi-model per node with auto packs by VRAM tier.

Next: static speed estimates in model profiles, response quality checks (retry on garbage), complexity-aware token budgets. See [mesh-llm/docs/ROUTER_V2.md](mesh-llm/docs/ROUTER_V2.md) for the full phased plan.

## Production relay infrastructure

Currently mesh-llm uses iroh's default public relays for NAT traversal. We have a self-hosted iroh-relay on Fly.io (`relay/`) but it's not the default yet. Dedicated relays in key regions would improve connectivity.

## Agent launcher

`mesh-llm run` as a one-command way to launch AI agents talking to the mesh:

```bash
mesh-llm run goose          # launch goose session with mesh backend
mesh-llm run pi             # launch pi with --provider mesh
mesh-llm run opencode       # opencode pointed at mesh API
```

We already print launch commands when the mesh is ready and show them in the web console. There's also a native Goose provider (`micn/mesh-provider-v2` branch on `block/goose`) with emulated tool calling.

## Single binary distribution

Currently ships as a 3-binary bundle (`mesh-llm` + `llama-server` + `rpc-server`). Could compile llama.cpp directly into the Rust binary via [llama-cpp-2](https://crates.io/crates/llama-cpp-2) — one binary, no bundle.

## MoE expert sharding ✅

Implemented. Auto-detects MoE, computes overlapping expert assignments, splits locally, session-sticky routing. Zero cross-node traffic. See [MoE_PLAN.md](MoE_PLAN.md).

Remaining: optimized rankings for unknown models, scale testing on Mixtral 8×22B / Qwen3-235B.

## Demand-based rebalancing

Partially done. Unified demand map via gossip, standby nodes promote to serve. Next: large-VRAM hosts auto-upgrade models when demand warrants it.

## Resilience

Done: Nostr re-discovery (v0.26.1), llama-server watchdog (v0.27.0), multi-host load balancing (v0.27.0). Next: tensor split recovery when a peer dies.
