# OpenAI Guardrails Rollout

This doc captures the v1 rollout shape for the OpenAI guardrail layer, the
operator contract, and the benchmark evidence path.

## Rollout Defaults

Phase 0 stays off by default unless an operator turns it on explicitly.

- `off` is the default deployment shape.
- `metrics` is opt-in.
- `enforce` is opt-in.

Server-side activation is now an explicit runtime setting for mesh-hosted Skippy
backends:

```bash
# startup default for this process and all later runtime-loaded Skippy models
mesh-llm serve --model MiniMax-M2.5-Q4_K_M --mesh-guardrails metrics

# no-restart update for the running process
mesh-llm runtime guardrails --mode enforce --port 3131

# equivalent local management API call
curl -s -X POST localhost:3131/api/runtime/mesh-guardrails \
  -H 'Content-Type: application/json' \
  -d '{"mode":"enforce"}' | jq .
```

The runtime command updates the shared server-side `GuardrailPolicy.mode`, so
already wrapped hosted models and future runtime-loaded/replacement Skippy
backends observe the new mode without a process restart. The current posture is
visible at `/api/status.runtime.openai_guardrails`.

The goal is validated emulation, not a general OpenAI tool runtime.

## Request Mode Contract

The corpus runner's `--guardrail-mode` does not mutate server config. It only
records the request mode and the expected server posture.

- `off` sends and records `mesh_guardrails: false`.
- `metrics` sends and records `mesh_guardrails: true`.
- `enforce` sends and records `mesh_guardrails: true`.

The operator still has to launch or reconfigure the server with
`--mesh-guardrails`, `mesh-llm runtime guardrails`, or the management API to
match the chosen mode. A request flag never upgrades a server that is still
running in a different mode.

## v1 Limits

The v1 surface is intentionally narrow.

- streaming is pass-through
- no tool execution happens inside the guardrail layer
- no hard constrained decoding is promised
- real tools plus strict structured output is unsupported in v1

That means the layer can validate and emulate shaped responses, but it does not
become a full agent runtime.

## Retry and Exhaustion

Guardrail flow is rescue before retry.

- rescue tries to turn malformed output into a valid assistant response first
- retry follows only after rescue fails or the policy requires another pass
- retry budget means `max_retries + 1` total attempts

The supported exhaustion modes are:

- `FailClosed`
- `PassLastText`

`PassLastText` only succeeds when the final text is safe and representable.
Otherwise the run fails closed.

## Guardrail Behavior Provenance

The rollout preserves the reliability-layer behavior that informed this
adaptation:

- a synthetic `respond` tool can be injected and stripped by the proxy
- rescue happens before retry
- retry exhaustion is `max_retries + 1` total attempts
- guardrail outcome recording can propagate optional arguments while remaining
  backward compatible

## Telemetry Privacy

Telemetry stays metrics only and bounded.

- no prompts or completions
- no schemas or tool arguments
- no request paths or endpoint URLs
- no raw IDs or raw device IDs

The guardrail telemetry inventory stays aligned with the allowlist in
`docs/plugins/telemetry.md`, and new attributes must be added there before they
can be exported.

## Evidence Collection

Two scripts cover the rollout evidence path:

- `scripts/run-openai-guardrail-corpus.py` for guardrail reliability runs
- `scripts/run-llama-benchy-openai.sh` for throughput and latency runs

The guardrail corpus runner should write its JSON artifact under
`.sisyphus/evidence/`. The corpus is deterministic and small, so the default
validation pass can use `20 trials` without turning the check into a full-day
benchmark.

If a Python sidecar baseline is available, you may optionally run a smoke
comparison against the same corpus to compare behavior before and after this
adaptation. That comparison is only an aid for local validation; it is not a
release gate.

Suggested artifact layout:

```text
.sisyphus/evidence/
  task-9-doc-disclaimers.txt
  task-9-benchmark-scaffold.txt
  openai-guardrail-corpus.json
```

The corpus should include a small set of prompts that cover:

- streaming pass-through
- real tool-call reliability
- synthetic `_mesh_respond` rescue
- strict structured output
- unsupported real tools plus strict structured output

## Context Compaction

Context compaction now exists as a sibling hosted-model decorator instead of as
part of the retry/validation guardrail engine. The pure compaction policy lives
in `mesh-llm-guardrails`; the OpenAI-facing decorator is
`CompactingOpenAiBackend`. Requests can force the behavior with the
`mesh_compact` extra field, while Skippy-hosted backends fill the runtime
context limit from the loaded stage config before composing compaction with the
tool-call guardrail wrapper.

The intended composition is still middleware-style and model-adjacent:
compaction sits next to the guardrail wrapper, not inside OpenAI ingress routes
or transport code. Future work is tuning the default thresholds with live agent
evidence and expanding compaction diagnostics, not adding the decorator shape
itself.
