# Anthropic compatibility

The Messages surface is a protocol adapter in `openai-frontend`. The host normalizes `/v1/messages` into `/v1/chat/completions` **before** classification, affinity, model selection, MoA, plugin dispatch or remote forwarding. Only the ingress keeps the original response protocol. Remote peers and OpenAI providers receive chat requests. JSON responses and SSE events are translated at the client boundary.

The embedded router uses the same backend, hook wrappers, guardrails, request context, cancellation and lifecycle helpers as chat. Host relay adaptation retains the original route for logging, terminal usage for routing, and capsule nonce headers. Stream requests explicitly request terminal usage from upstreams.

## Supported request semantics

- User/assistant text, top-level system text and text blocks. The installed Claude Code client also emits inline `role: system` text turns; these map directly to shared system messages as a compatibility extension.
- User image blocks with URL or base64 sources; mixed content order is preserved.
- Function tools, assistant tool-use blocks, and user tool-result blocks including images.
- `max_tokens`, temperature, top-p, top-k and stop sequences.
- Tool choice auto, any, tool and none; `disable_parallel_tool_use` maps to the shared parallel-tool setting.
- `metadata.user_id` maps to the shared user/affinity input.
- `output_config.effort` maps to the existing reasoning-effort control. JSON schema output maps to the existing response-format control. Backend capability checks still apply.
- Claude Code's adaptive/enabled/disabled `thinking` controls map to the shared reasoning settings. Signed `thinking` and `redacted_thinking` replay blocks are accepted on assistant turns and omitted from the OpenAI-shaped prompt because provider-specific signatures cannot be forwarded to a different backend.
- Claude Code's `clear_thinking_20251015` context-management edit is accepted as a compatibility no-op: the shared prompt already omits provider-specific thinking blocks.
- Ephemeral prompt-cache markers on system/message text, tool results and tools are validated, removed from the OpenAI-shaped content blocks, and enable in-memory prompt retention. Both the default five-minute marker and explicit one-hour TTL are accepted.
- Existing mesh hooks/guardrails extensions and explicit prompt-cache keys/retention are retained.

Unsupported fields and content kinds return a 400 error, rather than silently changing their meaning. Server tools, documents, unsupported context-management edits and service tiers remain outside this adapter. This surface targets complete Claude Code client interoperability, not every separate Anthropic platform API (for example, batches and file storage).

`/v1/messages/count_tokens` has an independent request schema: it does not require `max_tokens`. Local staged serving renders the same chat template and uses the loaded tokenizer without generation. A backend without this capability, or a media prompt whose token count cannot be determined by that path, returns an explicit unsupported error. It does not substitute a character estimate.

## Agent verification

`tests/anthropic_contract.rs` covers protocol translation, hooks, terminal usage, multiple streamed tools, images, error termination and token counting. These are deterministic frontend tests, not a Claude process.

The host's `runtime::proxy::tests::claude_cli_executes_read_tool_through_host_ingress` launches the pinned real Claude Code executable against host ingress and a deterministic OpenAI upstream. It verifies the client's default prompt caching, adaptive thinking, streamed Read invocation and tool-result round trip. The test is enabled by the `claude-code-integration` feature; the Linux Rust-test lane installs Claude Code 2.1.273 and runs it whenever `mesh-llm-host-runtime` is in the affected package batch.

The harness uses a temporary fixture, a separate `CLAUDE_CONFIG_DIR`, `--bare`, a dummy local API key and only the Read tool. It never bypasses permissions. Prompt caching and thinking remain enabled so the test exercises Claude Code's real default request shape.

This deterministic test proves client/host interoperability. A live Claude-model smoke still requires an Anthropic credential and is a separate pre-merge deployment gate; the deterministic CI test cannot substitute for it.
