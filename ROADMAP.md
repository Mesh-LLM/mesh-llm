# Roadmap

High-level directions for mesh-llm. Not promises — just things we're thinking about, or have thought about.

## Smart model router ✅

Implemented. Heuristic classifier detects Code/Reasoning/Chat/Creative/ToolCall with Quick/Moderate/Deep complexity. Task-dominant scoring ensures the right model handles each request. Tool capability is a hard filter. Multi-model per node with auto packs by VRAM tier. Auto-fallback ladders walk to the next-best model when the top pick's peers are unhealthy.

## Mixture of Agents (MoA) ✅

Implemented as the `mesh` virtual model. Fan-out across multiple worker models on the mesh, reducer synthesizes the result. Streaming output, tool-call passthrough, opinionated no-think default, configurable first-answer grace. See [docs/design/MOA_GATEWAY.md](docs/design/MOA_GATEWAY.md).

This could do with ongoing development and benchmarking to improve. 

## Mobile chat app (exemplar)

A native mobile app that joins a mesh by scanning a QR code. Client-only — no GPU, no model serving. Just a beautiful chat interface backed by the mesh's GPU pool.

- Scan QR code → join mesh → chat with any model the mesh serves
- Uses iroh relay for connectivity (works through NAT, cellular, WiFi)
- OpenAI-compatible API underneath (same as any mesh client)
- iOS first (Swift + iroh-ffi), Android follow-up
- "AirDrop for AI" — one scan and you're talking to a 235B parameter model

This is the best way to show what mesh-llm does: zero setup, zero config, just scan and chat.

## Multimodal

Vision, audio, and image generation/editing routed across the mesh. Capability advertisement gossiped so requests find compatible peers automatically. See [docs/design/MULTI_MODAL.md](docs/design/MULTI_MODAL.md).

Done:
- Vision input on capable models (Qwen3-VL, MiniMax-M2.5, etc.)
- Audio input (transcription, multimodal audio understanding)
- Capability-aware routing — image/audio requests only go to peers that advertise the capability
- Blob plugin for request-scoped media storage

Wanted:
- **Image generation models** (SDXL, FLUX, etc.) as first-class mesh peers — same gossip + capability + routing story, just emits PNG bytes instead of tokens
- **Image editing / inpainting** — accept an input image + mask + prompt, return edited image
- Audio generation (TTS) as a peer role
- Video generation as a future peer role

The goal is "every modality is just another model behind the mesh's OpenAI-compatible facade." Same QR-code-to-join story works for image-gen as for chat.

## Speculative decoding

Verify draft tokens against the target model to accelerate generation. Experimental, opt-in. See PR #567.
More work around using ngrams, drafting models is in progress. 
Some experimental work around predictive prompt completion into prefile has been done (yet to be proven, prefil is parallel so latency tolerant)
MTP work with llama.cpp ongoing, but should be part of this to accelerate inference and especially reduce vulnerability to latency between layers. 

## Demand-based rebalancing

Partially done. Unified demand map via gossip, standby nodes promote to serve, and large-VRAM hosts can opt into fresh active-demand upgrades for local artifacts. Next: download-backed upgrades, split-aware upgrades, and replica-count balancing.

## Blackboard ✅

Blackboard is moving out to its own plugin repository. The mesh-llm host keeps the generic plugin transport and CLI dispatch; blackboard installs through the plugin manager and owns its own CLI/MCP surface there.

## MoE expert sharding ✅

Implemented. Auto-detects MoE, computes overlapping expert assignments, splits locally, and uses session-sticky routing with zero cross-node expert traffic.
Best thought of as experimental, most results show this doesn't perform as well as one would hope, more research is needed to see if expert sharding this way is actually practical.

## Platform targetting, Desktop apps, embedding of mesh SDK, distribution

Mesh is packaged for many platforms, and can be run as a background process, but it would make sense to have desktop/GUI apps which host it in way that can offer utility to end consumer utility as well as being able to yield compute when needed locally. 

Mesh also needs to be packaged as an SDK which can be used from various client languages to launch as a client/serve node as seamlessly as possible.
