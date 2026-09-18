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
- Existing mesh hooks/guardrails extensions and explicit prompt-cache keys/retention are retained.

Unsupported fields and content kinds return a 400 error, rather than silently changing their meaning. This includes Anthropic-specific thinking blocks/signatures, server tools, prompt-cache control/TTL, documents, context-management edits and service tiers. This is a supported Messages subset, not the entire Anthropic platform API (for example, batches and file storage).

`/v1/messages/count_tokens` has an independent request schema: it does not require `max_tokens`. Local staged serving renders the same chat template and uses the loaded tokenizer without generation. A backend without this capability, or a media prompt whose token count cannot be determined by that path, returns an explicit unsupported error. It does not substitute a character estimate.

## Agent verification

`tests/anthropic_contract.rs` covers protocol translation, hooks, terminal usage, multiple streamed tools, images, error termination and token counting. These are deterministic frontend tests, not a Claude process.

The host's `runtime::proxy::tests::claude_cli_executes_read_tool_through_host_ingress` launches a real installed Claude CLI against host ingress and a deterministic OpenAI upstream. It verifies a streamed Read invocation and the tool-result round trip. It is ignored in ordinary unit runs because it requires the external CLI. Run the full host package suite, then explicitly run this harness with `MESH_CLAUDE_BIN` pointing to the installed executable.

The harness uses a temporary fixture, a separate `CLAUDE_CONFIG_DIR`, `--bare`, a dummy local API key and only the Read tool. It never bypasses permissions or calls the Anthropic service. Compatibility settings `DISABLE_PROMPT_CACHING=1` and `CLAUDE_CODE_DISABLE_THINKING=1` select the supported surface; see the [Claude environment-variable reference](https://code.claude.com/docs/en/env-vars).

This deterministic test proves client/host interoperability. Model quality, live remote model execution and real plugin-provider support require their own deployment evidence.
