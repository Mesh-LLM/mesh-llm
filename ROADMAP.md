# Roadmap

High-level directions for mesh-llm. Not promises — just things we're thinking about.

## Smart model router ✅

Implemented. Heuristic classifier detects Code/Reasoning/Chat/Creative/ToolCall with Quick/Moderate/Deep complexity. Task-dominant scoring ensures the right model handles each request. Tool capability is a hard filter. Multi-model per node with auto packs by VRAM tier. Auto-fallback ladders walk to the next-best model when the top pick's peers are unhealthy.

## Mixture of Agents (MoA) ✅

Implemented as the `mesh` virtual model. Fan-out across multiple worker models on the mesh, reducer synthesizes the result. Streaming output, tool-call passthrough, opinionated no-think default, configurable first-answer grace. See [docs/design/MOA_GATEWAY.md](docs/design/MOA_GATEWAY.md).

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

## Demand-based rebalancing

Partially done. Unified demand map via gossip, standby nodes promote to serve. Next: large-VRAM hosts auto-upgrade models when demand warrants it.

## Blackboard ✅

Implemented. Shared ephemeral text messages across the mesh — agents post status, findings, questions, and answers. Multi-term OR search, convention prefixes (STATUS/QUESTION/FINDING/TIP/DONE), PII auto-scrub, flood-fill propagation with digest sync. Works on any node with or without models. MCP server (`mesh-llm blackboard --mcp`) exposes tools for agent integration. Agent skill installable via `mesh-llm blackboard install-skill`.

## MoE expert sharding ✅

Implemented. Auto-detects MoE, computes overlapping expert assignments, splits locally, and uses session-sticky routing with zero cross-node expert traffic.
