# Mesh native serving plugin host

This crate loads and validates one native serving plugin for Mesh's dedicated
local-model OpenAI surface. Plugin proposal calls run on an isolated worker;
Skippy's decode thread waits only until its own absolute deadline and never
joins or drains plugin work. Lifecycle and authoritative verification outcomes
remain ordered through the same bounded queue.

At activation, the host passes a model-bound tokenizer capability over the
plugin ABI. Its inventory view and input-piece slices are borrowed, while the
capability callback context remains owned by the active plugin host until
shutdown. The host bounds and copies structured input before calling the
loaded tokenizer. Opaque controls are never decoded as text: unsupported
controls return `UNSUPPORTED_INPUT`. Tokenizer calls are preparation-time
operations and are not part of proposal dispatch.
