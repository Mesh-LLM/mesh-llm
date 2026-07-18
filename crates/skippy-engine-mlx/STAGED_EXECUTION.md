# MLX partial-layer staged execution

## Status

Dense Llama-family MLX stages now run as separate OS processes from partial
SafeTensors artifacts and communicate over Skippy's existing binary stage wire.
The first proof uses `HuggingFaceTB/SmolLM2-135M-Instruct` split at layer 15.

This is now integrated with the explicit `mesh-llm serve --split` launch path,
while automatic split selection remains future work:

- `skippy-engine` owns the engine-neutral `StageEngine` contract and residual
  buffer descriptors.
- `skippy-server::engine_transport` serves that contract using the existing
  `StageWireMessage`, ready handshake, activation codec, and reply codec.
- `skippy-server::llama_engine` proves the existing llama `RuntimeState` can
  implement the same dense contract, including F16/BF16/F32 residual conversion
  and checkpoint/restore/trim delegation, without changing the native ABI.
- `MlxStageEngine` auto-detects the materialized SafeTensors family. Dense
  Llama stages own per-session KV caches; the first frontier adapter executes
  one internal, stateless Nemotron-H Nano MoE layer. MLX objects remain on a
  dedicated worker thread.
- `mlx-stage` starts a stage process or drives a chain as a proof client.
- `StagePrepare` / `StageLoad` with `backend=mlx` and an immutable
  `hf-model://org/repo@<commit>` reference now derive or reuse a validated
  quantized stage and start the same engine through the normal host
  stage-control loop.
- The mesh advertises an additive `backend-mlx` capability, plans exact
  SafeTensors ranges through its ordinary topology, and exposes the chain via
  the normal OpenAI frontend.

No process in the proof has access to the complete checkpoint. The tokenizer
and config files are small shared metadata; tensor data comes only from that
process's `model.safetensors`.

The same branch also includes a complementary single-node proof. Ordinary
`mesh-llm serve --model HuggingFaceTB/SmolLM2-135M-Instruct` resolves the full
SafeTensors checkpoint, automatically quantizes eligible unquantized dense
tensors to affine-4 during MLX load while preserving frontier/pre-quantized
representations, and serves normal and streaming OpenAI chat. See
`SERVE_INTEGRATION_STATUS.md` for the integrated status and limitations.

## Verified result

On Apple Silicon Metal, using two materialized 155.28 MiB partial files:

| Process | Layers | Tensor file available | RSS after the proof |
| --- | ---: | ---: | ---: |
| stage 0 | `0..15` | 155.28 MiB | 188,784 KiB |
| stage 1 | `15..30` | 155.28 MiB | 189,168 KiB |

The processes exchanged F16 residual activations and generated:

```text
[284, 260, 2240, 314, 1343, 327, 624, 8685]
```

That exactly matches the whole-model and in-process split reference for the
same prompt across prompt prefill and seven subsequent decode calls. Each stage
kept an independent per-layer KV cache, and `Stop` cleared the session in both
processes.

The host-managed proof also passed from a clean MLX stage cache. Both ranges
shared checkpoint identity
`303b5a31e5226edb03a48f6f77464736a91a404b1500f385ec43d0951ce81e87`,
but retained distinct stage cache keys:

| Layers | Planned HTTP payload | Complete source shard | Avoided | Requests |
| --- | ---: | ---: | ---: | ---: |
| `0..15` | 162,857,381 bytes | 269,060,552 bytes | 106,204,032 bytes | 3 |
| `15..30` | 162,858,533 bytes | 269,060,552 bytes | 106,202,880 bytes | 4 |

The test submitted Prepare, polled inventory, submitted Load, checked the
materialized status identity/path, generated the same eight reference tokens,
and submitted Stop through `spawn_stage_control_loop`. The runtime status does
not mislabel the derived slice as the full source model or claim a cache pin
that does not exist.

The next engine-level proof enabled tensor-at-a-time JIT weight quantization.
The pinned safemlx loader visits one dense tensor at a time, quantizes it, and
eagerly evaluates and synchronizes the packed weight/scales/biases before
visiting the next tensor. This bounds the lazy graph, but the TensorView and
stream-copy path can temporarily hold more than one physical source copy. With
affine 4-bit, group size 64, the whole 30-layer reference and the two
independently loaded 15-layer stages generated the same quantized-model tokens:

```text
[260, 2240, 314, 253, 1379, 282, 25801, 28]
```

The two-stage processes retained 349 MLX parameters each and had post-proof RSS
of 87,392 KiB and 87,952 KiB, versus roughly 189 MiB each at source precision.
This proves deterministic per-stage quantization and quantized stage execution;
it does not by itself prove peak RSS or remove the dense partial artifact.

The next proof removed that dense partial artifact. `mlx-stage derive` consumes
the sequential exact-range session from `model-hf`, quantizes and synchronizes
one matrix at a time, copies packed results into a bounded host-side output
shard, and deletes each dense source tensor before fetching the next. Pure-Rust
SafeTensors I/O avoids linking MLX's bundled GGUF symbols into the existing
Skippy/llama.cpp binary.

With 16 MiB output shards, the two SmolLM2 halves produced:

| Layers | Dense ranges fetched | Quantized artifact | Shards | Largest source temp | MLX peak active | Process max RSS | macOS peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0..15` | 162,825,984 B | 45,887,848 B | 3 | 56,623,208 B | 72,548,352 B | 140,853,248 B | 240,157,440 B |
| `15..30` | 162,827,136 B | 45,889,726 B | 3 | 56,623,208 B | 72,548,352 B | 140,722,176 B | 241,550,080 B |

Both derived directories loaded without a quantization request because their
config records the affine-4/group-64 encoding. They again produced exactly:

```text
[260, 2240, 314, 253, 1379, 282, 25801, 28]
```

No complete dense stage or source shard was written. The report separates the
checkpoint/plan/quantizer recipe hash from an output-content digest and records
every output shard hash. Repeating the one-layer derivation produced a
byte-identical weight shard; the whole directories intentionally differ because
reports include local paths and runtime memory evidence. The shard-size option
is a soft bundle target, so one packed tensor may exceed it. This is a bounded
`model_type=llama` artifact builder, not evidence that frontier expert-bank
transforms fit the same bound. Artifact byte counts and the measured
source-plus-output working-disk high-water mark exclude the report, lock files,
and filesystem allocation overhead.

`mlx-stage derive-cached` then proved the reusable cache seam. It maps the
strong recipe identity to a locked managed directory and validates schema,
recipe, aggregate artifact bytes, output-content digest, and every shard hash
before accepting a hit. On the same pinned layer-14 slice, the cold call made 9
tensor-payload range requests; the warm call returned the identical recipe and
content hashes with `cache_hit=true`, made 0 tensor-payload range requests, and
used 17,809,408 B max RSS. It still re-plans lightweight config/index/header
metadata to reconstruct the strong recipe key.

The host control path now consumes this cache directly. `StagePrepare` maps the
load request to a derivation recipe and builds or validates it on a blocking
worker; `StageLoad` validates the same entry and loads MLX from the derived
directory. It fails on a cache miss instead of downloading or quantizing tensor
payloads during Load. The load request carries an additive quantization profile:
`auto`, affine 4-bit, affine 8-bit, or MXFP4. An absent profile from an older
peer means `auto`; an unknown value fails closed. On the current Apple Metal
backend, `auto` selects affine 4-bit with group size 64. The chosen profile is
part of the recipe identity and is carried through inventory, preparation, and
running status. Inventory responses echo the requested profile, so one profile
cannot satisfy readiness for another, including across mixed-version peers.

The host's claimed checkpoint identity is verified from the lightweight
metadata plan before the first tensor payload request. Prepare cancellation is
also threaded into cache-lock waits and the sequential visitor. It is checked
before every payload request and before and after each quantization callback;
an HTTP transfer or MLX operation already in flight finishes cooperatively
before its temporary file is removed.

A clean host-control run built both halves without retaining a dense stage:

| Layers | Exact source tensor bytes | Derived artifact | Payload requests |
| --- | ---: | ---: | ---: |
| `0..15` | 162,825,984 B | 45,859,713 B | 136 |
| `15..30` | 162,827,136 B | 45,861,308 B | 137 |

It completed Prepare, Load, Start, generation, and Stop in 120.74 seconds and
produced the established affine-4 token reference. An immediate identical run
hit both validated entries, performed the same lifecycle in 8.87 seconds, and
used 258,162,688 B max RSS. `MESH_MLX_DERIVED_CACHE_DIR` can isolate or relocate
the host cache for testing and operations. Cache capacity and eviction still
need an owner; warm lookup also still probes lightweight upstream metadata to
reconstruct the strong recipe, then streams each cached shard once to verify
both its shard hash and the aggregate content digest.

The two partial files are the exact-range artifacts described in
`../../spikes/mlx-safetensors-stages/FINDINGS.md`. Tied input/output embeddings
are intentionally duplicated across the stages; that is why the sum of the two
files is larger than the full checkpoint even though neither process downloads
the full checkpoint.

The production `model-hf` planner now also understands the `nemotron_h`
architecture used by Nemotron 3 Nano, including its `backbone.layers.*` layout
and first/final boundary tensors. Against pinned
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` layer `1`, it selected
2,594,936,576 bytes in 261 tensors from one 4,991,210,024-byte shard; the
largest individual tensor was 19,955,712 bytes. This is metadata/range-planning
evidence only for the general family layout. The derived builder and stage
engine now support exactly one internal Nemotron-H Nano MoE layer at a time.
Mamba, attention, first/final boundaries, and multi-layer hybrid stages remain
fail-closed until their state and boundary semantics are implemented.

Reproduce the metadata-only proof (it downloads the pinned config, index, and
one SafeTensors header, but no tensor payloads):

```bash
cargo test -p model-hf --lib \
  plans_real_nemotron_h_moe_layer_without_tensor_payloads -- \
  --ignored --nocapture
```

The bounded affine4 implementation has also been exercised against that exact
pinned layer. It streamed 2,594,936,576 BF16 bytes through 261 individual range
requests, quantized 258 matrices while retaining three dense tensors, and wrote
730,324,736 tensor bytes. Maximum process RSS was 822,165,504 bytes and the
largest ephemeral source tensor file was 19,955,848 bytes. The resulting
artifact strict-loaded into safemlx's real layer-1 `TransformerBlock` and
produced a finite `[1, 1, 2688]` output for a deterministic nonzero input.
The same artifact then loaded through `MlxStageEngine`; execution through the
shared F32 `StageActivation` contract matched direct block execution within
`atol=1e-4`, `rtol=1e-4` (across repeated validation runs, worst observed max
absolute difference `1.1920929e-7`, max relative difference `1.8225228e-5` for
reference values above `atol`). It
compared two session IDs, reset session 1, and independently compared its
repeated output too. Separate sparse executions were not bit-identical, so the
validator records both hashes and enforces the declared numerical tolerance.
Here, bounded memory means bounded by the final packed layer: the six routed
bank buffers total 718,405,632 bytes. It does not mean derivation stays at the
one-expert (~20 MB source tensor) footprint. The forward is an executable smoke
test, not a dense-versus-quantized numerical parity result.

```bash
just mlx-stage-build
just mlx-stage derive \
  --repo nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
  --revision 97ab8012882a655dc38df4fee47422aca9caca07 \
  --layer-start 1 --layer-end 2 \
  --output /tmp/nemotron-nano-layer1-affine4 \
  --weight-quantization affine4
just mlx-stage validate-nemotron-h \
  --model /tmp/nemotron-nano-layer1-affine4 --layer 1
just mlx-stage validate-nemotron-h-stage \
  --model /tmp/nemotron-nano-layer1-affine4 --layer 1
just mlx-stage validate-nemotron-h-wire \
  --model /tmp/nemotron-nano-layer1-affine4 --layer 1 --tokens 32 \
  --wire-dtype f32
just mlx-stage validate-nemotron-h-wire \
  --model /tmp/nemotron-nano-layer1-affine4 --layer 1 --tokens 32 \
  --wire-dtype f16
```

The last two commands deliberately put the real layer-1 engine in an
unnecessary two-stage loopback chain. The downstream stage is a synthetic
capture/final engine, not another Nemotron layer. It asserts the forwarded
`PrefillFinal` kind, session, all token/position sidebands, and
`[1, 32, 2688]` residual; it
returns a sentinel prediction; and it records the session reset before the
upstream Stop/ACK completes. The F32 boundary matched direct block execution
with maximum absolute error `2.3841858e-7` under `atol=1e-4`, `rtol=1e-4`.
The F16 boundary had maximum absolute error `0.000923872` and maximum relative
error `0.00048756658` under `atol=5e-4`, `rtol=1e-3`. The corresponding
activation payloads were 344,064 F32 bytes and 172,032 F16 bytes per boundary.
Runtime active memory stayed at 730,404,608 bytes and peak MLX memory was
820,697,688 bytes. Those thresholds are empirical evidence for this layer and
deterministic 32-token input, not a family certification. The input values are
multiples of 1/32, so the F16 result mostly exercises output-boundary rounding
rather than difficult input rounding. The validator defaults to one token and
accepts `--tokens` for larger prefill checks.

This proves the real Skippy TCP framing, activation codec, sideband forwarding,
predicted reply propagation, and chained Stop/ACK around one real MLX frontier
layer, including a 32-token prefill. It does not prove a second real model
stage, decode, Nemotron recurrent state, host/QUIC orchestration, or end-to-end
token logits.

## Boundary-fence benchmark

`mlx-stage bench-boundary` is the first instrumented pass over the independent
boundary cost. It creates one evaluated F32 MLX array, applies a synthetic lazy
F32 add, and times four release-mode phases separately:

1. completion of the synthetic add through MLX eval/synchronize, with graph
   construction outside the timer;
2. the evaluated host view plus allocation/copy into the F32 byte buffer;
3. the production Skippy F32 or F16 activation-payload encoder; and
4. post-receive activation-payload reconstruction into F32.

The host-copy phase includes the MLX evaluated-view call. The decode phase does
not include message framing, socket reads, TCP, QUIC, or receive-buffer
allocation. This is not a model-layer benchmark.

All samples are collected before the telemetry exporter starts, then their
original timestamps are emitted as bounded OTLP spans to an explicitly
configured metrics-server. The benchmark fails on codec drift, non-finite
values, telemetry loss, or a canonical span-count mismatch; it finalizes the
run and saves metrics-server's canonical `report.json`. The report's
`eval_and_host_copy_total` and `codec_total` percentiles are calculated from
paired per-iteration sums, not by adding independent phase percentiles.

F32 encoding and decoding are straight byte copies. F16 includes numeric
conversion in both directions, so paired F32/F16 results come from separate
per-dtype runs and are not a within-run equivalence comparison. MLX memory
counters do not include the Rust host buffers.

No prompt, activation values, local paths, collector endpoints, hardware IDs,
or model contents enter telemetry. The required HTTP and OTLP endpoints are
transport targets and are not copied into spans or run config. The operator run
label is validated to a bounded URL-safe character set; a local output report
may contain its explicitly requested report path.

### V2 evidence

The commit `d381bbd3` matrix ran serially on an Apple M5 Max with 128 GB
unified memory and macOS 26.5.2. Each release-mode process used three warmups
and 20 measured iterations. Boundary p50 is the paired
`eval_and_host_copy_total`; codec p50 is the paired `codec_total`.

| Width | Tokens | F32 payload | F32 boundary | F32 codec | F16 payload | F16 boundary | F16 codec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,688 | 32 | 0.328 MiB | 0.629 ms | 0.010 ms | 0.164 MiB | 0.236 ms | 0.259 ms |
| 4,096 | 1 | 0.016 MiB | 0.244 ms | 0.001 ms | 0.008 MiB | 0.234 ms | 0.010 ms |
| 4,096 | 32 | 0.500 MiB | 0.580 ms | 0.014 ms | 0.250 MiB | 0.570 ms | 0.306 ms |
| 4,096 | 512 | 8 MiB | 0.832 ms | 0.217 ms | 4 MiB | 0.920 ms | 4.986 ms |
| 8,192 | 1 | 0.031 MiB | 0.255 ms | 0.001 ms | 0.016 MiB | 0.640 ms | 0.020 ms |
| 8,192 | 32 | 1 MiB | 0.631 ms | 0.031 ms | 0.500 MiB | 0.604 ms | 0.596 ms |
| 8,192 | 512 | 16 MiB | 1.212 ms | 0.437 ms | 8 MiB | 1.340 ms | 10.036 ms |
| 16,384 | 1 | 0.063 MiB | 0.613 ms | 0.003 ms | 0.031 MiB | 0.716 ms | 0.042 ms |
| 16,384 | 32 | 2 MiB | 0.609 ms | 0.052 ms | 1 MiB | 0.680 ms | 1.211 ms |
| 16,384 | 512 | 32 MiB | 1.572 ms | 0.927 ms | 16 MiB | 1.651 ms | 20.187 ms |

Do not over-interpret the one-token or independent F32/F16 boundary timings:
at that scale dispatch, allocator, and process-level noise are material. The
large-prefill codec result is much more stable and nearly linear. At widths
4K, 8K, and 16K, F16 saved 4, 8, and 16 MiB while adding 4.77, 9.60, and
19.26 ms over the F32 codec. With serialized conversion and transfer, that is
an approximately 0.81 GiB/s (7.0 Gbit/s) effective-payload break-even point:
below it F16 should recover its conversion cost from bytes saved; above it F32
should be faster. Actual selection must be measured per host/link because
conversion can be optimized or overlapped and TCP/QUIC costs are absent here.

All F32 round trips were exact. All F16 runs stayed finite with maximum absolute
error `0.00045216084`, below the declared `0.001` synthetic-range gate. The 20
canonical reports contain 20 completed runs and exactly 1,600 spans: 400 for
each phase, zero drops, and zero export errors. Only the 400 encode spans carry
`skippy.activation_bytes_sent`.

Reproduce one matrix cell with metrics-server running in another terminal:

```bash
just metrics-server \
  db=/tmp/mlx-boundary.sqlite \
  http_addr=127.0.0.1:18081 \
  otlp_addr=127.0.0.1:14317

just mlx-stage-build
just mlx-stage bench-boundary \
  --width 16384 --tokens 512 --wire-dtype f16 \
  --warmup-iterations 3 --measured-iterations 20 \
  --metrics-http http://127.0.0.1:18081 \
  --metrics-otlp-grpc http://127.0.0.1:14317 \
  --metrics-run-id mlx-boundary-w16384-t512-f16-v2 \
  --metrics-report /tmp/mlx-boundary-metrics.json \
  --output /tmp/mlx-boundary-local.json
```

### Production loopback-TCP follow-on

`mlx-stage bench-tcp-boundary` moves the same synthetic activation through the
production engine-neutral Skippy TCP server. Its paired round-trip timer starts
before F32/F16 activation encoding and ends after the predicted-token reply, so
it includes sender encoding, binary framing/write, loopback TCP, server
read/framing, F32 reconstruction, the synthetic final `StageEngine` adapter,
and reply framing/read. It also includes construction and destruction of the
message plus its token/position sidebands and the sink/reply assertions. The
reported `wire_activation_payload_bytes` excludes the fixed frame, eight bytes
per token of sidebands, and the reply. It excludes MLX/model compute, QUIC,
remote links, and the outbound activation encoding of a non-final stage.

Connection bind/connect/READY and teardown are outside the timer. Samples run
sequentially over one warmed persistent connection with client and server as
threads in the same process. This is steady-state loopback latency, not
connection startup, multi-process behavior, concurrent throughput, or pipeline
overlap.

The first warmup validates the complete decoded tensor against the source using
the same exact-F32 / bounded-F16 gate. Measured samples finish before telemetry
starts. Each sample then becomes one
`stage.mlx_boundary_tcp_roundtrip` span in a canonical metrics-server run.

#### TCP V1 evidence

Commit `6350e3a9` ran release-mode F32/F16 pairs on the same M5 Max host as the
codec matrix, using three warmups and 20 sequential samples over one connection.

| Width | Tokens | F32 payload | F32 p50 / p95 | F16 payload | F16 p50 / p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,688 | 32 | 0.328 MiB | 0.396 / 0.428 ms | 0.164 MiB | 0.627 / 0.677 ms |
| 4,096 | 512 | 8 MiB | 2.764 / 4.336 ms | 4 MiB | 8.086 / 8.231 ms |
| 8,192 | 512 | 16 MiB | 5.655 / 6.002 ms | 8 MiB | 13.601 / 13.814 ms |
| 16,384 | 512 | 32 MiB | 4.936 / 9.361 ms | 16 MiB | 25.398 / 25.622 ms |

F32 wins on this high-bandwidth loopback path in every pair, consistent with
the codec-only prediction that F32 wins above the roughly 7.0 Gbit/s effective
payload break-even. The non-monotonic 16K F32 p50 and its wider p95 tail also
show why these numbers must not be converted into a remote-link bandwidth
claim: same-process allocation, kernel buffering, scheduling, and host copies
are part of this steady-state round trip.

The eight canonical runs contain exactly 160 round-trip spans with zero drops
or export errors. F32 warmup reconstruction was exact; F16 maximum absolute
error was `0.00045216084`. This validates the production TCP framing direction,
but the next policy gate remains a controlled remote TCP/QUIC sweep.

```bash
just mlx-stage bench-tcp-boundary \
  --width 16384 --tokens 512 --wire-dtype f16 \
  --warmup-iterations 3 --measured-iterations 20 \
  --metrics-http http://127.0.0.1:18081 \
  --metrics-otlp-grpc http://127.0.0.1:14317 \
  --metrics-run-id mlx-tcp-boundary-w16384-t512-f16-v1 \
  --metrics-report /tmp/mlx-tcp-boundary-metrics.json \
  --output /tmp/mlx-tcp-boundary-local.json
```

### Separate-process and two-host TCP fence

TCP v2 can move the validating sink into a separate process or host while
retaining the same production `engine_transport` framing and reconstruction
path. The sink is intentionally a benchmark tool: it is unauthenticated,
unencrypted TCP and must be bound only on a trusted private network or behind a
firewall. Its `width`, `tokens`, and `wire-dtype` must exactly match the sender.

Start the sink in the foreground on the receiving host:

```bash
just mlx-stage serve-tcp-boundary-sink \
  --bind 0.0.0.0:19090 \
  --width 16384 --tokens 512 --wire-dtype f16
```

Then add `--connect <sink-private-address>:19090` to
`bench-tcp-boundary` on the sending host and use a fresh metrics run ID. The
runner allows 10 seconds for connect/READY and 30 seconds for each write and
reply. It uses one warmed persistent connection per invocation; the foreground
sink can accept later invocations without a restart. Run only one benchmark
client at a time per sink: its deliberately bounded validation cache retains
the most recent session, so concurrent clients can force revalidation into a
measured sample.

Each run derives a distinct wire session from its metrics run ID. The sink
validates the first activation for that session and returns the observed
maximum absolute error as an explicit acknowledgement. It records the session
only after the exact-F32 / bounded-F16 gate succeeds, so a failed attempt cannot
poison a retry. The sender requires and independently gates that acknowledgement
before recording samples. Reports use the neutral `external_tcp` label because
an address supplied with `--connect` may still be localhost; the address itself
is excluded from the local/canonical telemetry payload.

The current READY handshake does not carry sink build identity. For controlled
two-host evidence, copy the exact same release `mlx-stage` artifact to the sink
host and compare its SHA-256 on both hosts. Record that out-of-band checksum
alongside the client `code_revision`; do not infer sink provenance from the
client revision alone.

The acknowledgement is a claim made by the sink, not cryptographic remote
attestation. `warmup_validation_ack_received` means the expected structured
reply arrived, and `warmup_sink_acknowledged_max_abs_diff` is the value reported
by that sink. The identical-artifact checksum procedure above is therefore part
of the controlled evidence, not an optional provenance detail.

#### SSH-forwarded two-host V2 evidence

Commit `27bd5880` was built once in release mode, signed once, and copied
unchanged to an M4 Max receiver from an M5 Max sender. Both hosts reported the
same executable SHA-256:
`ba5d1ea6f2613d0171d36eeaf9dfd86904d3ae19f6d335b3185bbb4ebe5a2222`.

The receiver's application firewall allowed local sink traffic but suppressed
data after a direct-LAN TCP handshake for the ad-hoc research binary. The
controlled sweep therefore used one persistent SSH local forward to the
receiver's loopback-bound sink. Every timed sample still covers production
sender encoding/framing, cross-host transfer, production receiver
framing/reconstruction, sink acknowledgement, and the reply, but it also
includes SSH tunnelling and encryption. These are not raw-LAN or QUIC numbers.

Each cell used three warmups and 20 sequential measured samples. The 16K pair
was repeated in reverse dtype order on a fresh tunnel after the initial F16 run
showed a severe transient.

| Width | Tokens | F32 payload | F32 p50 / p95 | F16 payload | F16 p50 / p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,688 | 32 | 0.328 MiB | 16.447 / 21.106 ms | 0.164 MiB | 16.532 / 22.415 ms |
| 4,096 | 512 | 8 MiB | 150.690 / 173.837 ms | 4 MiB | 107.376 / 115.066 ms |
| 8,192 | 512 | 16 MiB | 288.479 / 874.937 ms | 8 MiB | 200.168 / 677.533 ms |
| 16,384 A | 512 | 32 MiB | 603.895 / 2346.909 ms | 16 MiB | 1441.627 / 6323.343 ms |
| 16,384 B | 512 | 32 MiB | 572.663 / 625.476 ms | 16 MiB | 345.498 / 923.549 ms |

The actual 2,688×32 Nemotron boundary was effectively tied, consistent with
fixed SSH/tunnel overhead being large relative to its small payload. F16
reduced p50 by about 29% at 4K and 31% at 8K. In the fresh-tunnel 16K repeat it
reduced p50 by about 40%, but the opposite result and multi-second tails in the
first 16K pair show that this setup cannot select a production wire dtype. It
is a functional two-host proof, with results consistent with payload reduction
mattering on this constrained SSH-forwarded path. Raw LAN/QUIC, repeated
interleaved trials, and pipeline overlap remain required for automatic policy.

The 10 completed canonical runs contain exactly 200
`stage.mlx_boundary_tcp_roundtrip` spans. All use schema
`mlx-tcp-boundary-v2`, revision `27bd588087b8186ccc902b000e79a90cc3b39d43`,
and transport `external_tcp`, with zero dropped spans or export errors. No span
falls outside its run lifecycle. F32 acknowledgements were exact and every F16
acknowledgement reported maximum absolute error `0.00045216084`.

## Reproduce

Build once:

```bash
just mlx-stage-build
```

This writes `target/release/mlx-stage` and the required sibling
`target/release/mlx.metallib`. Copy both files together when moving the CLI to
another Apple-Silicon host. For this pinned build the Metal library is about
157 MiB; its generated size can change with MLX or the Apple toolchain. It is a
runtime resource shared by every model on that host, not part of a stage
artifact.

Derive both quantized stage directories directly from immutable source ranges:

```bash
just mlx-stage derive \
  --repo HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --layer-start 0 --layer-end 15 \
  --output /tmp/mlx-derived-smol-stage0 \
  --weight-quantization affine4 --shard-size-mib 16

just mlx-stage derive \
  --repo HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --layer-start 15 --layer-end 30 \
  --output /tmp/mlx-derived-smol-stage1 \
  --weight-quantization affine4 --shard-size-mib 16
```

To use the identity-bound cache instead of an explicit output path, replace
`derive` with `derive-cached`, omit `--output`, and optionally pass
`--cache-root`. Repeating the command reports `cache_hit=true` and
`source_range_request_count=0`.

The derived directories are already quantized; do not pass
`--weight-quantization` when serving them.

Start the final stage:

```bash
just mlx-stage serve \
  --model /tmp/mlx-derived-smol-stage1 \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --stage-index 1 --layer-start 15 --layer-end 30 \
  --bind 127.0.0.1:19091 --wire-dtype f16 --compute-dtype bf16
```

The directly derived directories are already affine-4. For a separate dense
`/tmp/mlx-split-smol` proof, use those paths and add
`--weight-quantization affine4` to both stage commands. The `prove` default is
the established affine-4 reference
`260,2240,314,253,1379,282,25801,28`.

Start the first stage in another terminal:

```bash
just mlx-stage serve \
  --model /tmp/mlx-derived-smol-stage0 \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --stage-index 0 --layer-start 0 --layer-end 15 \
  --bind 127.0.0.1:19090 --downstream 127.0.0.1:19091 \
  --wire-dtype f16 --compute-dtype bf16
```

Drive the chain:

```bash
just mlx-stage prove --connect 127.0.0.1:19090 --wire-dtype f16
```

### Real two-host split proof

The same M4 Max used for the boundary sweep independently derived only the
final `15..30` range from the immutable SmolLM2 checkpoint. In this recorded
pinned run it made 137 tensor payload requests for 162,827,136 source tensor
bytes and wrote a 45,889,726-byte affine-4 stage in three shards. All three
shard hashes exactly matched the earlier M5 Max derivation, demonstrating
deterministic stage-local materialization across the two machines for this
checkpoint, safemlx revision, and quantization recipe.

The first real `MlxStageEngine` then loaded the pre-derived `0..15` stage on the
M5 Max and forwarded its width-576 residuals through an SSH local forward to
the M4 Max's real `15..30` final stage. Both used BF16 compute and the same
affine-4 artifacts. Two successive F16-wire runs, including session reset and
reuse, and one F32-wire run all produced the reference sequence:

```text
[260, 2240, 314, 253, 1379, 282, 25801, 28]
```

This closes the intentionally unnecessary two-host small-Llama execution
proof: each layer server can hold only its own directly derived SafeTensors
stage, and the existing production binary stage protocol composes the two real
MLX engines. The transport was SSH-forwarded because of the receiver firewall,
and startup was manual through `mlx-stage`; this is not yet mesh coordinator,
placement, capability advertisement, OpenAI stage-0 orchestration, or raw
LAN/QUIC evidence.

## Deliberate limitations of this checkpoint

- `MlxStageEngine` supports dense Llama ranges and exactly one internal,
  stateless Nemotron-H Nano `E`/MoE layer. It rejects Nemotron Mamba, attention,
  dense-MLP, first/final, and multi-layer ranges. Inkling is not exposed through
  the partial-stage adapter.
- The derived builder handles ordinary rank-2 Llama weights and one Nano split
  expert bank. Inkling still needs its transformed rank-3 grouped-expert loader;
  unsupported families are not silently treated as Llama.
- The pinned safemlx Nemotron-H implementation matches the 52-layer Nano
  schema, not Nemotron 3 Ultra's 108-layer latent-MoE schema. Ultra range plans
  are storage-locality evidence, not executable-family support.
- Bounded Nemotron-H derivation and execution currently accept exactly one
  internal `E`/MoE layer. They do not expose a hybrid multi-layer stage or
  recurrent state on the wire.
- The Nemotron binary-wire validator uses a synthetic adjacent final stage and
  configurable loopback prefill (one and 32 tokens have been exercised). Its
  three-layer synthetic topology exists only to exercise the transport harness;
  it is not a deployable 52-layer model topology.
- Greedy sampling only; sampling metadata is preserved in the contract and
  rejected explicitly when enabled.
- No KV page import/export, cache trim/checkpoint, MTP, speculative verify,
  multimodal projection, or transport batching yet.
- `engine_transport` is the reduced compatibility lane. The mature llama.cpp
  binary server remains unchanged and still owns telemetry, exact-prefix cache,
  batching, and OpenAI orchestration.
- Explicit `mesh-llm serve --split` now capability-gates MLX participants,
  produces stage assignments, and drives them from an OpenAI stage-0 frontend.
  Automatic split selection and additional family adapters remain. Explicit
  host requests derive and reuse quantized artifacts, but cache eviction and
  an optional local request-to-recipe locator remain. The quantization field
  is an additive mesh protocol change; old peers omit it and therefore mean
  `auto`, while unknown
  values fail closed on new peers. Automatic placement must capability-gate
  explicit non-default profiles before mixed-version deployment. No Skippy ABI
  changed.
