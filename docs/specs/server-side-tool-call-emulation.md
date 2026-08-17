# Server-side tool-call emulation

Small / non-tool-trained models served through the mesh `/v1/chat/completions`
endpoint handle the OpenAI `tools` field poorly: the chat template either
ignores the schemas or the model loops, re-issuing the same call. goose already
solved this for its local-inference provider (a text-convention emulation plus a
tolerant parser), but that path is bypassed when a client talks to a mesh, and
every other OpenAI client hits the same wall.

The serving node is the right place to fix this: it is the only party that knows
the actual **chat-template capability** of the loaded model. Clients only know a
model name.

## Where it lives

- `crates/skippy-server/src/frontend/tool_emulation.rs` — detection, request
  adaptation, and response parsing. Self-contained and unit-tested.
- `crates/skippy-server/src/frontend/prompting.rs` — wires emulation into
  `prepare_chat_prompt` (request adaptation) and `parse_chat_output` (response
  parsing).
- `crates/skippy-server/src/frontend.rs` — `parse_emulated_chat_output`, the
  streaming-aware bridge into `ParsedChatMessage`.

## Detection

Emulation is gated on **template capability, not model size**. When the chat
template is applied with tools, the patched llama.cpp staged runtime returns
`metadata_json`. A tool-capable jinja template yields a tool-call **grammar
trigger** (for example `<tool_call>`), so `grammar_triggers` is non-empty; a
template with no native tool support (for example SmolLM2-135M) yields an empty
`grammar_triggers` list. Native support is therefore detected as a non-empty
`grammar_triggers`.

This is the mesh-llm analogue of goose's
`template_result_supports_native_tool_calling`. goose reads llama.cpp's
`parse_tool_calls` + parser fields, but in mesh-llm's patched runtime
`parse_tool_calls` is true for *every* tools request and `chat_parser` is always
a non-empty PEG structure, so neither distinguishes a tool-capable template.
`grammar_triggers` is the field that does.

A tool-trained model whose template emits a tool-call grammar trigger (for
example Qwen2.5-0.5B and Qwen3.5-0.8B) keeps native tool calling and sees **zero
behavior change**; a model whose template has no tool grammar (for example
SmolLM2-135M) is routed through emulation.

## Request adaptation

When a request carries `tools` for a non-tool-capable template, the prompt is
re-rendered:

1. `tools` / `tool_choice` are stripped from the template input.
2. A compact instruction is injected into the system message: for each tool, its
   name, description, and a **compact parameter schema** (property names + types,
   with `?` marking optional args). The live prototype showed that name +
   description alone made a 0.6B model hallucinate argument names; the compact
   schema fixes that.
3. Conversation history is rewritten so the template never sees tool roles:
   assistant `tool_calls` become assistant text in the `TOOL_CALL {json}`
   convention, and `role: "tool"` messages become `user` messages prefixed with
   `Tool result:`.

The convention the model is taught is a single line:

```
TOOL_CALL {"name": "the_tool_name", "arguments": {"arg": "value"}}
```

## Response parsing

Generated text is scanned for `TOOL_CALL` lines, which become real OpenAI
`tool_calls` with `finish_reason: "tool_calls"`. Parsing is tolerant of
surrounding prose and `<think>` blocks (a reasoning model's scratchpad never
triggers or hides a call). Emulated calls ride the same downstream assembly as
native ones, so `call_mesh_*` ids and streaming behave identically.

Streaming holds back the trailing incomplete line and withholds tool calls until
finalization, so a half-formed marker is never streamed as content and calls are
emitted once — matching native tool-call streaming semantics.

## Reference

Ported from goose's local-inference provider
(`crates/goose-local-inference/src/tool_emulation.rs`, `tool_parsing.rs`,
`prompts/tiny_model_system.md`) and the client-side mesh-console prototype.
