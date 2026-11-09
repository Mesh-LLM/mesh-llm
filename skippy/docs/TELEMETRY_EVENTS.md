# Skippy telemetry events

Skippy reports point events as OpenTelemetry
[log-based events](https://opentelemetry.io/docs/specs/semconv/general/events/):
OTLP log records whose `event_name` identifies their structure, stamped with
the time they happened and the trace and span that were active, with no
body. An event marks a moment inside a longer operation: a lookup's
outcome, a state change, a stall. Work with a duration is a span, and facts
about a whole operation are span attributes.

Events go to the OTLP endpoint traces use and, when
`SKIPPY_TELEMETRY_OTLP_LOGS_FILE` names a file, to that file as OTLP/JSON
lines in the collector's file-exporter format. With neither, nothing is
collected. The resource is `service.name=skippy` with the stage's lifecycle
attributes.

## Events

Severity numbers follow OpenTelemetry: 9 is info, 13 warn, 17 error.

| Event | Severity | When | Attributes |
|---|---|---|---|
| `skippy.kv.lookup` | 9 | A prefix lookup at one cache tier decided whether reusable state exists: L1 when a request probes the resident radix cache, L2 and L3 when an L1 miss falls through to them | `skippy.kv.tier` (`l1`, `l2`, `l3`), `skippy.kv.result` (`hit`, `miss`), `skippy.kv.matched_tokens` on a hit |
| `skippy.kv.eviction` | 9 | The resident cache released prefixes to make room | `skippy.kv.tier`, `skippy.kv.evicted_entries`, `skippy.kv.evicted_tokens` |
| `skippy.kv.admission_shortfall` | 13 | A request was admitted below the KV pool's free-space watermark after every releasable prefix was evicted | `skippy.kv.shortfall_tokens` |
| `skippy.graph.reserved` | 13 | After the model was ready, the backend scheduler reserved compute buffers again (`llama.sched_reserve`), blocking the next graph | `skippy.native.duration_ms`, `skippy.native.detail` when the runtime names what it reserved for |
| `skippy.gpu.pipeline_compiled` | 13 | After the model was ready, a GPU kernel pipeline was compiled on first use (`mtl.pipeline_compile`) | as above; the detail names the kernel |
| `skippy.gpu.buffer_allocated` | 13 | After the model was ready, a device buffer was allocated (`mtl.buffer_alloc`) | as above |
| `skippy.attention.path` | 9 | A backend's attention was first seen, or switched path | `skippy.gpu.backend`, `skippy.attention.path` (`row_lists`: flash attention over each sequence's own rows; `flash`: flash attention over the whole masked K/V span; `unfused`: masked soft max over the whole span), `skippy.attention.kv_cells`, `skippy.attention.queries` |
| `skippy.gpu.op_profiling` | 13 | A backend first split a graph at op class boundaries (`GGML_METAL_OP_PROFILE=1`): its timings are inflated and only their shares compare | `skippy.gpu.backend` |

The native-stall events sit on the native span that did the work; the
attention events on the call that submitted the GPU command; cache events
on the span active where the cache decided.

The point events serving has recorded on spans (`stage.*`, `skippy.resources`)
are sent as log-based events too, under the same names and attributes, at
severity 9, except `stage.scheduler_worker_panic` and
`stage.connection_worker_panic` at 17, and `stage.binary_connection_error`,
`stage.connection_admission_rejected`,
`stage.openai_downstream_lane_return_failed` and
`stage.openai_linear_proposal_discard_failed` at 13. The span events remain,
as OpenTelemetry's migration from span events to log-based events advises.

## GPU work by op class

With a runtime that has `SKIPPY_FEATURE_NATIVE_TRACING`, every
`<backend>.command_buffer` span says what its graph nodes did:

| Attribute | Meaning |
|---|---|
| `skippy.gpu.op_class` | `attention`, `matmul`, `recurrent`, `kv_write`, `other`, or `mixed` when the command holds several |
| `skippy.gpu.nodes.<class>`, `skippy.gpu.input_bytes.<class>` | Nodes of each class present, and the bytes they read: a gather from or row write into a cache counts only its rows, and attention only the K/V rows its nodes address |
| `skippy.gpu.layer` | The model layer of the command's first layer-named node |
| `skippy.attention.path` | `row_lists`, `flash` or `unfused`, as for the event |
| `skippy.attention.kv_cells`, `skippy.attention.queries` | The most KV cells one query addresses (for unfused attention, the K/Q product's K span), and the most query rows one attention node processes |
| `skippy.attention.row_list_nodes`, `skippy.attention.flash_nodes` | Attention nodes that read per-sequence row lists, and that are flash attention |
| `skippy.recurrent.states`, `skippy.recurrent.states_carried` | For a graph with recurrent state: the most states one gather reads (llama.cpp gathers from the batch's first state cell to its last), and the most of those copied back unchanged because no sequence in the batch owns them |
| `skippy.gpu.split_by_class` | The graph was split at class boundaries |

Attention is flash attention, or the masked soft max and K/Q, K/Q/V
products of unfused attention. Recurrent is SSM, RWKV, gated linear
attention and delta-net state, and copies to or from its cache. Views and
reshapes do no work and have no class.

A normal graph runs in one to three command buffers, so their GPU time
covers several classes and only the input bytes divide it. With
`GGML_METAL_OP_PROFILE=1` in the runtime's environment, Metal instead
encodes one command buffer per run of same-class nodes, in order on the
calling thread, so each command's GPU time belongs to one class and one
layer. Splitting serialises the graph and stops fusion across class
boundaries: totals grow, while each class's share of GPU time is what such a
trace measures. Compare shares between two profiled runs, not a profiled run
against an unprofiled one. A split graph's command buffers have no `encode`,
`queue` and `execute` child spans, only the `skippy.gpu.*_ms` attributes, so
that a profiled run, with hundreds of command buffers per graph, stays within
the exporter's queue. The `skippy.gpu.class_busy` and `skippy.gpu.input_bytes`
metrics sum every command, including any whose spans were dropped.

`<backend>.idle` spans mark where the GPU waited for the host: at least
50 µs between one command's execution and the next, under the call whose
work ended the wait. `skippy.gpu.idle_kind` is `within_graph` when both
commands belong to one graph compute (the host encoded slower than the GPU
ran) and `between_graphs` otherwise (host work between steps: sampling,
scheduling, building the next graph).

## Telling runs apart

Every span, metric and event carries the serving configuration that decides
which kernels run and how much KV they read, as resource attributes:
`skippy.kv.cache_type_k`, `skippy.kv.cache_type_v`, `skippy.attention.flash`
(the requested mode: `auto`, `enabled` or `disabled`), `skippy.kv.ctx_size`,
`skippy.scheduler.lanes`, and when set, `skippy.batch.size`,
`skippy.batch.ubatch_size`, `skippy.kv.unified`, `skippy.kv.offload` and
`skippy.kv.swa_full`. The path attention actually took is the
`skippy.attention.path` event and span attribute.

`OTEL_RESOURCE_ATTRIBUTES` adds the operator's own resource attributes, as
OpenTelemetry SDKs define it: comma-separated `key=value` pairs with
percent-encoded values, such as `skippy.run.label=q8-fa`. The stage's own
attributes win over a conflicting key.

The configurations worth comparing are operator settings: K and V precision
(`--kv-cache-type-k`, `--kv-cache-type-v`: `f16`, `q8_0`, `q4_0`; quantized V
needs flash attention), flash attention (`--flash-attention`: `disabled`
selects unfused attention over the whole span), the KV pool size
(`--ctx-size`) and prefix retention (`--prefix-cache off` keeps retained
prefixes out of the pool). Whether flash attention reads per-sequence row
lists is a runtime fact, not a setting: the runtime uses them wherever the
backend supports the stored K/V type (on Metal: F16, Q4_0, Q4_1, Q5_0, Q5_1
and Q8_0), and Skippy's KV pool is always unified.
