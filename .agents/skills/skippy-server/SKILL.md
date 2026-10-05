---
name: skippy-serving
description: Use this skill when running, configuring, debugging, or embedding skippy-serving, binary stage transport, OpenAI frontend integration, activation wire dtype settings, stage configs, lifecycle status, or nonblocking telemetry.
metadata:
  short-description: Run and debug skippy serving
---

# skippy-serving

Use this skill for skippy serving, embedded runtime lifecycle, and binary
stage-to-stage transport.

## Current Repo Shape

The mesh integration embeds `skippy-serving` through Rust APIs instead of
launching it as mesh's public OpenAI surface. Public OpenAI compatibility
belongs in `skippy-inference-api`; `skippy-serving` should remain the backend stage
runtime.

Important crates:

```text
skippy/crates/skippy-serving
skippy/crates/skippy-protocol
skippy/crates/skippy-runtime
mesh/crates/mesh-llm/src/inference/skippy
```

## Validation

Run cargo commands serially:

```bash
cargo check -p mesh-llm
cargo test -p skippy-serving --lib
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
compatibility in `skippy-inference-api`.

Stage status exposed by mesh should be backend-neutral at the API boundary.
Backend-specific details can remain in internal skippy structs.
