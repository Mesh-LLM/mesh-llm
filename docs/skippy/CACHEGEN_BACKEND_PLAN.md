# CacheGen Backend Qualification (#1652)

Status: **LMCache-compatible direct Metal restore passes the local F32/F32 and
F32+F16 gates; F16 and the lower-width sampled types remain stopped on local
latency, and CacheGen remains opt-in pending selection-path wiring**. Owner:
jian yang.
Reviewed against: #1652 scope, scama's directives of 2026-09-10 (v4
contract, CPU+Metal parity, six measurements, stop rule).

## Where each backend stands

| Backend | Kernel | Status | Evidence |
|---|---|---|---|
| CPU (reference) | scalar Rust | **Correctness reference only** — portable F32, F16, Q8_0, Q4_0, and mixed K/V adapters are implemented | Python/Rust fixtures pin LMCache revision `b5d109e`; typed archive fixtures cover every current user-selectable runtime K/V type |
| Metal (Apple GPU) | native MSL | **All typed restores implemented; F32/F32 and F32+F16 clear the local gate** — F16, F32, Q8_0, and Q4_0 match native fixture layouts; Q8_0/F32 straddles the latency boundary across repeats, while F16 and the lower-width sampled cases remain stopped locally | Apple M1 Ultra device fixture and typed 19K gates below |
| CUDA (NVIDIA) | shared native CUDA/HIP source | **Typed kernels landed; F16 fixture qualified** — the real RTX 5080 fixture matches F16 bytes, while F32, quantized, and matched typed 19K runtime gates remain | Real NVIDIA F16 fixture; remaining typed hardware gates pending |
| HIP/ROCm (AMD) | shared native CUDA/HIP source | **Typed kernels landed; compile/package qualified** — no real AMD runtime claim yet | Real AMD fixture and typed 19K gates remain |

Nothing may be marked implemented until it runs on real hardware and
matches the CPU reference bit-for-bit. Compile-only checks prove the
kernel lowers; they say nothing about the hardware.

Native Metal fixture builds must set `GGML_CCACHE=OFF`. The embedded source is
included through a generated assembly `.incbin`; an assembly cache can otherwise
reuse an object after the Metal source changes and run a stale kernel.

## The six spike measurements (2026-09-11 re-run, M2 Max, 4096x128 tile,
## 524,288 f16 values = 1,048,576 raw bytes, release build, synchronized
## stages, bitwise equality, transfers timed)

Every timed stage ends with `client.sync()` inside the timer: the
numbers cover real completion, not launch enqueue. H2D and D2H are
measured to completion as well. Cold JIT is the first synchronized
launch of the kernel specialization in the process (cubecl 0.10.0 has
no on-disk kernel cache on this path — the only one, SPIR-V, is
Vulkan-only and not enabled — so per-process first launch is the true
cold path for both backends). Equality is exact `==` on symbols and f32
bits, with mismatch counts printed.

| Metric | cubecl-cpu | wgpu (Metal) |
|---|---|---|
| Cold JIT/compile (quantize+delta; undelta+dequantize) | ~38-42 ms / ~13 ms | ~8-9 ms / ~4 ms |
| Warm dispatch, synchronized (avg per launch) | ~570-640 us | ~3.3-4.6 ms |
| H2D bytes (f32 tile + 8 B calibration), timed | 2,097,160 in ~12-18 ms | 2,097,160 in ~0.9-1.4 ms |
| D2H bytes (u32 symbols + f32 rebuilt), timed | 4,194,304 in ~36-44 us | 4,194,304 in ~3.0-4.1 ms |
| Live device buffer peak (tile + calibration + both outputs) | 6,291,464 | 6,291,712 |
| Encoded-size ratio (rANS over device symbols / raw) | **0.075 (13.3x)** | **0.075 (13.3x)** |
| Output equality vs CPU reference (bitwise) | symbols + values exact, 0/524,288 mismatches | symbols + values exact, 0/524,288 mismatches |

The live peak is what the harness actually holds while kernels run
(2,097,152-byte tile + 8-byte calibration + 2,097,152-byte symbol
buffer + 2,097,152-byte rebuilt buffer); Metal's allocator rounds its
copies slightly differently, hence the 248-byte difference. Warm-dispatch
convergence was checked across iteration counts (5/50/200/500). These
numbers replace both the 2026-09-10 enqueue-only measurements (5/4 us)
and the 2026-09-11 first re-run's output-only peak: the completion wait
and the input buffers are now inside the reported figures. Correctness
claims are unchanged: both backends were already symbol-exact, and the
values are proven bit-equal rather than within 1e-6.

Caveats, stated rather than buried: warm dispatch at 4096x128 is now
dominated by the synchronization round-trip plus launch overhead, not
bandwidth; the per-column scan is the correct bit-exact baseline, not
the fastest shape (a parallel scan is the follow-up). The 2x H2D cost
versus raw f16 bytes exists because the spike uploads f32; shipping f16
halves it and is a trivial follow-up.

## Buffer interop and the stop rule

The store's exported segments are plain byte buffers. CubeCL's runtime
consumes host buffers via `create_from_slice` and returns via
`read_one`; no zero-copy path into the store's packed segment files
exists today, so the spike measured the honest version: one H2D upload,
one D2H return. On this tile that is 2 MB up / 4 MB down against a
0.075-ratio encoded payload — the copy cost is real and is the thing the
later quality/performance gate must beat on the ~19K acceptance
workload. Per the agreed stop rule this evidence comes back before any
expansion: CubeCL is **not** a committed dependency, the spike lives
behind the `cachegen-spike` feature, and nothing in the library links it.

## Native exact control

The exact control arm uses `native-kv-page/1` per-segment identity. KV bytes
exported by the active runtime are written and restored verbatim, including
F32, F16, Q8_0, and Q4_0 layouts supported by that runtime; there is no storage
transcode. Mixed KV plus recurrent payloads cut at the representation boundary,
so auxiliary continuation state remains exact `raw/1`. Runtime page metadata is
validated before segment reads, while the existing exact-state identity binds
the runtime ABI, platform, model, layer range, and KV configuration. This is the
baseline every CacheGen result must beat end to end.

## Capability failure policy

A backend that cannot run a codec fails explicitly through the v4
per-segment identity gate (`SegmentCodecIdentity::is_supported` /
negotiation naming the segment index). There is no hidden fallback: an
unsupported `cachegen/1` segment is a clean miss with a named reason,
never a silent decode on another backend or a raw reinterpretation.
Lossy entries additionally carry `calibration_digest`; a lookup matches
only identically-calibrated entries and can never satisfy an exact
lookup.

## LMCache-compatible reference port

The active gate now uses a Rust port of LMCache revision
`b5d109ea99a89b4d8a670ee4fc2e8cb76411ee5c`. The source records the exact
upstream Python and CUDA files and retains Apache-2.0 attribution. It reproduces
the parts that determine reconstruction quality and wire size:

- per-token maximum-magnitude scaling across channels;
- LMCache's generic 32/16-bin K and V layer schedule;
- the normalized 33-entry CDF for every layer/channel pair;
- the CUDA implementation's 32-bit arithmetic coder and 256-token chunk limit;
- token-major K/V handling, including Skippy's transposed-V page layout.

Skippy uses a bounded portable envelope instead of LMCache's Python pickle
container. Fixtures generated by the independent Python scalar transcription
pin both 16-bin and 32-bin streams, and normal Rust tests require byte-for-byte
encoder agreement and decoder agreement with those fixtures.

This remains an opt-in correctness path and is not selected by storage or the
request path. The matched 19K result below proves the reference recovers
continuation quality, while also proving that scalar arithmetic decode cannot
meet the restore-to-first-token gate.

## Sequencing after the direct Metal gate

1. Preserve the scalar implementation and its pinned fixtures as the
   deterministic oracle for device kernels.
2. Keep the completed Skippy-owned compressed-page transaction and optional
   backend-registry hook as the integration boundary. Metal and CUDA/HIP decode
   validated records directly into allocated resident cells; capability or
   execution failure rolls back without a scalar fallback.
3. Qualify the shared CUDA/HIP decoder against the scalar fixtures and matched
   19K gate on real NVIDIA and AMD hardware.
4. Add independently decodable substreams or an equivalent parallel entropy
   layout before retrying local-tier Metal promotion. Preserve the current
   scalar stream as the compatibility oracle for the new revision.
5. Add encode from resident K/V storage and copy only the compact archive back
   to the persistence layer.
6. Run separate quality and latency gates for Q8_0/Q8_0, Q4_0/Q4_0, and the
   supported mixed K/V combinations now that every typed record is wired into
   the native backend contract.

## Typed portable record boundary

The portable page archive uses one unchanged LMCache-compatible F16 entropy
segment. Record kinds describe the runtime edge adapter: F32 and F16 scalar
rows, Q8_0 and Q4_0 block rows, plus F32/F16 transposed V. Encode converts each
native row into the shared F16 stream; decode reconstructs that stream and
packs the selected native row type. K and V carry independent kinds, so mixed
selections do not require a second container or codec revision.

The pure-Rust fixtures cover F32/F32, F16/F16, Q8_0/Q8_0, Q4_0/Q4_0,
Q8_0/Q4_0, and transposed F32 V. Metal, CUDA, and ROCm share a typed native
decode contract for direct F16 and F32 resident writes, including transposed V,
and fused Q8_0/Q4_0 block quantization into row-major resident storage. The
former F16-only backend symbol was removed rather than retained as a
compatibility alias. The Metal device fixture compares both quantized outputs
byte-for-byte with ggml's native quantizer. No scalar restore fallback is
permitted. Quantized destinations remain unqualified until matched end-to-end
quality gates measure the combined CacheGen and native repacking loss.

## Native runtime integration boundary

The existing native page API is host-buffer oriented. Rust passes a `&[u8]`
to `skippy_import_kv_page`; the C ABI receives a `const void *`; and
`llama_kv_cache::stage_import_kv_page` allocates cells before copying each run
with `ggml_backend_tensor_set`. A device decoder above this API would have to
materialize the complete decoded page in host memory and upload it again. That
would erase the main benefit on discrete CUDA and ROCm devices.

CacheGen therefore belongs inside the Skippy state-transfer transaction while
its portable envelope and scalar oracle remain in `skippy-cache`. The public
runtime accepts a compressed portable page. The native implementation validates
all records and destination coverage, allocates all target cell runs, resolves
the codec hook for every owning backend, dispatches decode into the resident K
and V tensors, synchronizes, and only then commits the session position. A
validation, capability, launch, or synchronization failure restores the prior
cell state. Unsupported backends return an explicit unsupported result; the
performance gate cannot silently select the scalar oracle.

The hook is an optional function obtained through
`ggml_backend_reg_get_proc_address`, following the extension mechanism already
used by Metal backend tuning. CacheGen state transfer is intentionally not a
new global ggml graph operation: it runs outside model execution, and making it
an op would also require global enum, scheduler, graph-identity, slice-planning,
shape, and backend-support changes. The registry hook keeps the patch local to
the state capability and the backends that implement it while still receiving
the active backend context needed for ordered execution.

CUDA and ROCm share the `ggml-cuda` source path, which llama.cpp already builds
through CUDA or HIP. Metal implements the same contract in MSL. Each launch
batches many 256-token arithmetic streams: one thread serially decodes at most
256 symbols for one channel while hundreds of thousands of independent channel
streams run in parallel. Destination metadata maps each stream to a K/V tensor,
allocated cell run, row stride, and optional transposed-V stride. This avoids a
launch per tile and permits dequantization and final layout writes in one pass.

## 19K direct Metal result (2026-09-11): QUALITY PASS, LOCAL LATENCY STOP

The direct-device gate was run from exact commit
`c86cf848b0fa183f2c2e594fe1e51bf1024d7185` on an Apple M1 Ultra (128 GiB,
Metal) with the same pinned Qwen3 0.6B Q8_0 model and 19,000-token workload as
the scalar result below. Native and CacheGen continuations run in isolated
sessions. The scalar decoder still runs as an independently timed correctness
oracle, but its 41.44 seconds are excluded from the direct-device TTFT. The
compact result is
[`cachegen-metal-device-qwen3-0.6b-19k-summary.json`](cachegen-metal-device-qwen3-0.6b-19k-summary.json).

| Metric | Native | Direct Metal CacheGen | Decision |
|---|---:|---:|---|
| Persisted bytes | 2,179,072,000 | 446,903,003 | 20.51% of native (4.88x smaller) |
| Persist path | 424.19 ms | 21,332.42 ms including encode | Fail for synchronous persistence |
| Read | 296.25 ms | 54.21 ms | CacheGen saves 242.04 ms |
| Resident import | 56.69 ms | 540.74 ms | CacheGen spends 484.05 ms more reconstructing K/V |
| Restore to first token | 385.41 ms | 604.84 ms | Fail (1.57x slower) |
| Scalar oracle decode | — | 41,436.07 ms, excluded from device TTFT | Correctness-only reference |
| Continuation throughput | 122.34 tok/s | 128.07 tok/s | No steady-state regression |
| p99 decode | 32.47 ms | 9.88 ms | Within the 5% regression budget |
| Greedy-token agreement | 64/64 control | 64/64 (100%) | Pass versus 95% gate |
| Estimated codec working bytes | — | 2,627,965,638 | Reported; no memory cap was supplied |

The Metal decoder is correct after aligning every staged tile before typed
metadata reads; the regression test covers two differently sized consecutive
tiles. Replacing the 64-bit arithmetic division with a float reciprocal estimate
and exact integer correction, plus a binary CDF search, reduced the same 19K
resident import from 744.95 ms to 540.74 ms. It still loses the local gate
because every restore reconstructs about 1.09 billion F16 values through
per-channel arithmetic streams of up to 256 symbols. On unified-memory M1 Ultra,
the native 2.18 GB copy is unusually fast: CacheGen's 242.04 ms read saving does
not recover its 484.05 ms reconstruction penalty. The result does not decide a
remote tier, where transfer time and pipelining differ; that tier needs its own
matched end-to-end gate under #1427.

## 19K typed Metal matrix (2026-09-12): F32 PASS, OTHER LOCAL PROMOTION STOPS

The typed gate was run from exact commits
`2e26d46ea87e1b8ee783460998e703f669513f91` (quantized and mixed rows) and
`74b4f60719d09dee7d9579c1365ac6f95d14c20c` (F32/F32) on the same Apple M1
Ultra, pinned Qwen3 0.6B Q8_0 model, 19,000-token prefix, and 64-step
continuation. The F32 run also caught and fixed a native config defect where
GGML enum value zero was interpreted as an unset cache type and silently
replaced with F16. The native regression now pins both the F16 default and an
explicit F32 request.
The F32/F16 crossover probes were run from exact commit
`ddddf34aa5e64262c9e113d07f9dcb506e5f7aab`.
The gate now accepts independent `--cache-type-k` and `--cache-type-v` values
and records them in its report. The compact matrix is
[`cachegen-metal-typed-qwen3-0.6b-19k-summary.json`](cachegen-metal-typed-qwen3-0.6b-19k-summary.json).

| K/V type | Native bytes | CacheGen bytes | CacheGen/native | Agreement | Native TTFT | CacheGen TTFT | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| F32/F32 | 4,358,144,000 | 446,903,003 | 10.25% | 64/64 (100%) | 1,199.13 ms | 766.12 ms | Pass: quality, size, local latency, and p99 |
| F32/F16 | 3,268,608,000 | 446,903,003 | 13.67% | 64/64 (100%) | 878.61 ms | 686.15 ms | Pass: quality, size, local latency, and p99 |
| F16/F32 | 3,268,608,000 | 446,903,003 | 13.67% | 64/64 (100%) | 840.32 ms | 671.93 ms | Pass: quality, size, local latency, and p99 |
| Q8_0/F32 | 2,757,888,000 | 565,441,003 | 20.50% | 64/64 (100%) | 800.57 / 727.17 ms | 796.94 / 792.13 ms | Unstable boundary: one pass, one latency stop |
| Q8_0/Q8_0 | 1,157,632,000 | 589,803,358 | 50.95% | 64/64 (100%) | 333.36 ms | 650.64 ms | Quality and size pass; local latency fails |
| Q4_0/Q4_0 | 612,864,000 | 635,037,607 | 103.62% | 61/64 (95.31%) | 189.84 ms | 699.35 ms | Quality floor passes; size and local latency fail |
| Q8_0/F16 | 1,668,352,000 | 565,441,003 | 33.89% | 64/64 (100%) | 558.28 ms | 792.05 ms | Quality and size pass; local latency fails |
| Q4_0/F16 | 1,395,968,000 | 610,269,862 | 43.72% | 61/64 (95.31%) | 483.98 ms | 807.03 ms | Quality and size pass; local latency fails |

These are representative runtime-valid layouts, not qualification of every
pairing. F32/F32 and both F32+F16 directions clear the matched local gate because
reading the compact archive saves enough time against native pages of at least
3.27 GB to absorb device reconstruction. At 2.76 GB, Q8_0/F32 is inside run
variance: it won once by 3.63 ms and lost once by 64.96 ms, so it remains stopped.
The nearly format-independent 521-586 ms CacheGen import times, compared with
native import scaling from 80 to 495 ms with page width, identify arithmetic
reconstruction as the next Metal target. F32/Q8_0 is not a valid control on this
model: llama.cpp requires Flash Attention for quantized V, while that mixed
cache pair does not have a compatible Flash Attention kernel.

An exact-head Metal staging optimization at
`e2bdc935e24cbed3bff767762547098db6028459` removes the second host copy that
previously occurred when each validated job was accumulated in `NSMutableData`
and then copied into a shared `MTLBuffer`. The backend now sizes and validates a
job before writing its payload, tile descriptors, and stream prefixes directly
into the final shared buffers. Q8_0/F32 import fell from 580.45 ms across the two
boundary runs above to 491.92 ms across two post-change runs, a 15.25% reduction.
F16/F16 import fell from 540.74 ms to 451.82 ms, a 16.44% reduction. Both cases
remain stopped on local TTFT: the optimized F16/F16 run took 514.34 ms versus
343.16 ms native, and the optimized Q8_0/F32 repeats took 713.51/701.97 ms versus
571.88/547.78 ms native. The remaining gap is device arithmetic decode rather
than redundant host staging.

A diagnostic 128-row archive doubled tile count from 4,200 to 8,344, expanded
Q8_0/F32 storage from 20.50% to 30.84% of native, and increased import to
727.74 ms. Smaller independently decoded tiles therefore do not solve this
format's Metal crossover; shared calibration with multiple arithmetic
substreams would require a separate format change.

The quantized K cases expose a consistent quality split: Q8_0 preserves all 64
greedy continuation tokens, while Q4_0 first diverges at step 15 and finishes at
61/64, barely above the declared 95% floor. Every completed CacheGen continuation
stays within the p99 decode-regression budget after restore. F16 and the
lower-width sampled cases remain stopped for the local tier; Q4_0/Q4_0 also has
no storage benefit. The remaining runtime-valid mixed permutations retain
fixture-level coverage and need separate 19K runs before any broader typed
qualification claim.

## 19K LMCache-compatible CPU result (2026-09-11): QUALITY PASS, LATENCY STOP

The opt-in gate was run from exact commit
`4bf0865be6e7a13dbf7159b0fb6e6c41f29db72c` on an Apple M1 Ultra (128 GiB,
Metal) with the pinned Qwen3 0.6B Q8_0 model
(`sha256:12fae8b8f78f0360b498d04c8db7d33aff29ab7d8080231f93a17c18119e6735`),
a 19,000-token prefix, F16 K/V, 256-row LMCache chunks, and 64 teacher-forced
continuation steps. The compact result is
[`cachegen-lmcache-qwen3-0.6b-19k-summary.json`](cachegen-lmcache-qwen3-0.6b-19k-summary.json).

| Metric | Native | LMCache-compatible CPU | Decision |
|---|---:|---:|---|
| Persisted bytes | 2,179,072,000 | 446,903,003 | 20.51% of native (4.88x smaller) |
| Persist path | 2,402.91 ms | 21,426.43 ms including encode | Fail |
| Read | 281.06 ms | 58.83 ms | Encoded path wins bytes/read time |
| Decode codec | — | 41,285.73 ms | Fail |
| Restore to first token | 374.38 ms | 41,411.13 ms | Fail (110.61x slower) |
| Continuation throughput | 119.85 tok/s | 124.17 tok/s | No steady-state regression |
| p99 decode | 33.24 ms | 8.59 ms | Within the 5% regression budget |
| Greedy-token agreement | 64/64 control | 64/64 (100%) | Pass versus 95% gate |
| Estimated codec working bytes | — | 2,627,965,638 | Reported; no memory cap was supplied |

This result resolves the earlier 18.75% agreement as a defect in the
simplified prototype rather than a limitation of CacheGen. The faithful
quantization schedule recovers all 64 continuation tokens. Scalar arithmetic
coding is still far outside the restore budget, so the CPU path remains an
oracle and no request-path wiring is allowed. A parallel device implementation
must pass the same gate before promotion.


## Historical 19K simplified-prototype result (2026-09-11): STOP

The implementation measured here is a Mesh-owned prototype inspired by
CacheGen. It uses per-segment min/max affine 4-bit calibration. The published
CacheGen design instead calibrates per model and applies mixed quantization
across tensor dimensions. This gate therefore rejects the current prototype;
it is not evidence that a paper-faithful CacheGen implementation fails.

The opt-in gate in `skippy-correctness state-handoff --cachegen-gate` was run
from exact commit `2677ad62e5295f6da2ac72ae7b8c978f87753d11` on an Apple M1 Ultra
(128 GiB, Metal) with the Qwen3 0.6B Q8_0 model
(`sha256:9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`),
a 19,000-token prefix, F16 K/V, 4,096-row codec tiles, and 64
teacher-forced continuation steps. The compact machine-readable result is
[`cachegen-quality-gate-qwen3-0.6b-19k-summary.json`](cachegen-quality-gate-qwen3-0.6b-19k-summary.json).

| Metric | Native | CacheGen | Decision |
|---|---:|---:|---|
| Persisted bytes | 2,179,072,000 | 202,373,739 | CacheGen is 9.287% of native (10.77x smaller) |
| Persist path | 1,369.84 ms | 15,508.82 ms including encode | Fail |
| Read | 239.14 ms | 22.29 ms | CacheGen wins bytes/read time |
| Decode codec | — | 23,521.69 ms | Fail |
| Restore to first token | 311.61 ms | 23,604.40 ms | Fail (75.75x slower) |
| Continuation throughput | 120.23 tok/s | 121.25 tok/s | No steady-state regression |
| p99 decode | 18.86 ms | 9.30 ms | Within the 5% regression budget |
| Greedy-token agreement | 64/64 control | 12/64 (18.75%) | Fail versus 95% gate |
| First mismatch | — | step 1 | Fail |
| Estimated codec working bytes | — | 2,412,371,750 | Reported; no memory cap was supplied |

Writes call `sync_all`; the same-run reads may still be page-cache warm, so the
read figures are not a cold-device bandwidth claim. That limitation cannot
reverse this decision: CacheGen's 23.52-second CPU decode alone is more than
75 times the complete native restore-to-first-token path, and continuation
quality fails independently.

Per the issue's stop rule, this result ended production work on that simplified
prototype. The LMCache-compatible port above replaces it in the opt-in gate;
its results must be measured separately. Native exact `native-kv-page/1`
remains the selected representation until the new result passes.
