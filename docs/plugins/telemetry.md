# Telemetry And Metrics Plugin

mesh-llm exports metrics-only OTLP/HTTP telemetry from host runtime code when
`[telemetry]` config enables an explicit endpoint. No collector or project-owned
destination is hard-coded.

The external `metrics` plugin lives at [Mesh-LLM/metrics](https://github.com/Mesh-LLM/metrics).
It advertises metrics support through the plugin API, but it does not receive
prompts, completions, logs, traces, endpoint URLs, or raw host identifiers.

## Configuration

Configure an OTLP metrics endpoint:

```toml
[telemetry]
enabled = true
service_name = "mesh-llm"
endpoint = "https://otel.example.com"
headers = { "authorization" = "Bearer TOKEN" }
export_interval_secs = 15
queue_size = 2048

[telemetry.metrics]
endpoint = "https://otel.example.com/v1/metrics"
```

Install and enable the optional metrics plugin when you want the plugin
capability advertised:

```bash
mesh-llm plugins install metrics
```

```toml
[[plugin]]
name = "metrics"

[plugin.startup]
connect_timeout_secs = 75
init_timeout_secs = 90
optional = true
lazy_start = true
```

The startup block is optional. It is useful on slow legacy machines where the
plugin process may take longer than the default startup budget, or where metrics
should be advertised only after the plugin is actually used.

Endpoint precedence is:

1. `telemetry.metrics.endpoint`
2. `telemetry.endpoint` normalized to `/v1/metrics`
3. `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`, only when `telemetry.enabled = true`
4. `OTEL_EXPORTER_OTLP_ENDPOINT` normalized to `/v1/metrics`, only when
   `telemetry.enabled = true`

If no endpoint is configured, telemetry export stays disabled. Ambient OTel
environment variables are not consumed unless telemetry is explicitly enabled in
mesh-llm config.

## Exported Metrics

Request and route metrics are emitted per fronting node. A collector or
dashboard can aggregate `mesh_llm_requests_inflight` across nodes for a
mesh-wide in-flight request view.

Counters:

- `mesh_llm_model_launch_total`
- `mesh_llm_model_launch_success_total`
- `mesh_llm_model_launch_failure_total`
- `mesh_llm_model_unload_total`
- `mesh_llm_model_exit_unexpected_total`
- `mesh_llm_model_request_total`
- `mesh_llm_route_attempt_total`
- `mesh_llm_guardrail_decision_total`
- `mesh_llm_guardrail_outcome_total`

Gauges:

- `mesh_llm_loaded_models`
- `mesh_llm_model_loaded`
- `mesh_llm_model_context_length`
- `mesh_llm_requests_inflight`

Histograms:

- `mesh_llm_model_launch_duration_ms`
- `mesh_llm_model_uptime_s`

## Privacy Boundary

Runtime telemetry exports metrics only. The external metrics plugin advertises a
capability only. Neither path exports prompts, completions, logs, traces,
hostnames, mesh gossip, relay messages, raw node IDs, raw GPU stable IDs,
endpoint URLs, or prompt hashes.

Guardrail telemetry follows the same boundary. It exports only bounded labels for
guardrail mode, contract kind, decision, bypass reason, parser stage, and retry
bucket. It does not export prompt text, completion text, schemas, tool
arguments, raw tool names, reserved sentinel names, request paths, endpoints, or
hostnames.

Guardrail v1 is validated emulation, not hard constrained decoding. Streaming is
pass-through, no tool execution happens inside the guardrail layer, and real
tools plus strict structured output stays unsupported in v1. See
`docs/design/OPENAI_GUARDRAILS.md` for the rollout contract and evidence path.

Local absolute and path-like model labels are reduced to filenames before export.
Hugging Face refs are preserved. GPU stable IDs and node IDs are exported as
stable pseudonymous hashes, not raw identifiers. Route-attempt metrics label
local, remote, and endpoint target kinds; remote target IDs are exported only as
stable hashes so collectors can aggregate node-to-node traffic without exposing
raw peer IDs.

Telemetry attributes are intentionally allowlisted in code. Any new exported
attribute must update the allowlist, tests, and this document before it is added
to an OTLP record.

| Attribute | Used by | Privacy handling |
|---|---|---|
| `mesh_llm.model` | lifecycle, request, route | Local/path-like labels are reduced to filenames; Hugging Face refs are preserved. |
| `mesh_llm.launch_kind` | lifecycle | Bounded enum. |
| `mesh_llm.gpu_count` | lifecycle | Count only. |
| `mesh_llm.is_soc` | lifecycle | Boolean only. |
| `mesh_llm.service_version` | lifecycle, request, route, in-flight | Build version only. |
| `mesh_llm.architecture` | lifecycle | GGUF architecture string when available. |
| `mesh_llm.quantization` | lifecycle | Derived quantization label. |
| `mesh_llm.gpu_name` | lifecycle | Hardware product label; no hostname or stable device ID. |
| `mesh_llm.gpu_stable_id` | lifecycle | Stable pseudonymous hash of the GPU ID. |
| `mesh_llm.backend_device` | lifecycle | Backend-local slot label such as `CUDA0`, `ROCm0`, `Vulkan0`, or `MTL0`. |
| `mesh_llm.backend` | lifecycle | Runtime/backend label. |
| `mesh_llm.context_bucket` | lifecycle | Bucketed context length, not the exact configured value. |
| `mesh_llm.failure_reason` | lifecycle | Bounded enum. |
| `mesh_llm.source_node_role` | request, route, in-flight | Bounded node role label such as `client` or `worker`. |
| `mesh_llm.source_node_id` | request, route, in-flight | Stable pseudonymous hash of the source node ID. |
| `mesh_llm.route_service` | request | Bounded service label: `local`, `remote`, `endpoint`, or `unavailable`. |
| `mesh_llm.request_outcome` | request | Bounded enum. |
| `mesh_llm.route_attempt_bucket` | request | Bounded retry bucket: `1`, `2`, `3_4`, or `5_plus`. |
| `mesh_llm.target_kind` | route | Bounded target kind: `local`, `remote`, or `endpoint`. |
| `mesh_llm.target_node_id` | route | Stable pseudonymous hash for local/remote node targets; omitted for endpoint targets. |
| `mesh_llm.attempt_outcome` | route | Bounded enum. |
| `mesh_llm.guardrail.mode` | guardrail decision, guardrail outcome | Bounded enum: `disabled`, `metrics`, or `enforce`. |
| `mesh_llm.guardrail.contract` | guardrail decision, guardrail outcome | Bounded enum: `tools` or `structured`. |
| `mesh_llm.guardrail.decision` | guardrail decision | Bounded enum: `eligible`, `bypassed`, `unsupported`, or `rejected`. |
| `mesh_llm.guardrail.bypass_reason` | guardrail decision | Bounded enum: `disabled`, `streaming`, `no_contract`, `unsupported_surface`, `reserved_collision`, or `mixed_tools_structured`. Omitted when no bypass reason applies. |
| `mesh_llm.guardrail.outcome` | guardrail outcome | Bounded enum: `pass_through`, `valid`, `rescued`, `retried`, `failed`, or `metrics_only_failure`. |
| `mesh_llm.guardrail.parser_stage` | guardrail outcome | Bounded enum: `none`, `json_exact`, `json_fenced`, or `json_substring`. |
| `mesh_llm.guardrail.attempt_bucket` | guardrail outcome | Bounded retry bucket: `1`, `2`, or `3_plus`. |

## Review Checklist

Before adding, renaming, or removing OTLP metrics or attributes:

1. Run the repo-local telemetry privacy review skill:
   `.agents/skills/telemetry-privacy-review/SKILL.md`.
2. Keep export destination behavior explicit: no default collector and no ambient
   OTel env export unless `telemetry.enabled = true`.
3. Update `TELEMETRY_ATTRIBUTE_ALLOWLIST` in
   `crates/mesh-llm-host-runtime/src/runtime/survey.rs`.
4. Update the attribute inventory above.
5. Add or update focused tests proving private paths, raw node IDs, raw GPU
   stable IDs, endpoint URLs, prompts, and completions are not exported.
6. Keep guardrail corpus evidence under `.sisyphus/evidence/`, separate from
   OTLP export and from the telemetry metric payloads themselves.

## Runtime Safety

Telemetry exporter setup failures disable telemetry without failing inference
startup. Runtime events are buffered through a bounded queue; when the queue is
full, the oldest event is dropped instead of blocking inference.
