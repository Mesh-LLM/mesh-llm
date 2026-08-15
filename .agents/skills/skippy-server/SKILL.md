---
name: skippy-server
description: Use this skill when running, configuring, debugging, or embedding skippy-server, binary stage transport, OpenAI frontend integration, activation wire dtype settings, stage configs, lifecycle status, or nonblocking telemetry.
metadata:
  short-description: Run and debug skippy serving
---

# skippy-server

Use this skill for skippy serving, embedded runtime lifecycle, and binary
stage-to-stage transport.

## Current Repo Shape

The mesh integration embeds `skippy-server` through Rust APIs instead of
launching it as mesh's public OpenAI surface. Public OpenAI compatibility
belongs in `openai-frontend`; `skippy-server` should remain the backend stage
runtime.

Important crates:

```text
crates/skippy-server
crates/skippy-protocol
crates/skippy-runtime
crates/mesh-llm/src/inference/skippy
```

## Validation

Run cargo commands serially:

```bash
cargo check -p mesh-llm
cargo test -p skippy-server --lib
cargo test -p skippy-protocol --lib
cargo test -p mesh-llm-host-runtime --lib inference::skippy
```

For lifecycle/status changes, also run:

```bash
cargo test -p mesh-llm-host-runtime --lib
```

## Rules

Do not reintroduce standalone `kv-server` or `ngram-pool` dependencies into
mesh. Keep structured outputs, tools, logprobs, and `/v1/responses`
compatibility in `openai-frontend`.

Embedded-server code reachable while mesh owns the TUI alternate screen must
not write directly to stdout/stderr (`print!`, `println!`, `eprint!`,
`eprintln!`, or raw writers). Emit `OutputEvent` or `tracing` instead so the
host's shared sink can project the message into the dashboard. When changing
stage startup or diagnostics, run the TUI regression checks: prove a physical
terminal clear/invalidation, verify narrow resize guidance, and verify `q` or
Ctrl-C restores the prompt without raw stderr lines.

Stage status exposed by mesh should be backend-neutral at the API boundary.
Backend-specific details can remain in internal skippy structs.
