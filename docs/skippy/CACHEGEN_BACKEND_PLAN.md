# CacheGen Backend Qualification (#1652)

Status: **LMCache-compatible CPU reference passes quality and fails restore
latency; native passthrough remains the promoted path**. Owner: jian yang.
Reviewed against: #1652 scope, scama's directives of 2026-09-10 (v4
contract, CPU+Metal parity, six measurements, stop rule).

## Where each backend stands

| Backend | CubeCL runtime | Status | Evidence |
|---|---|---|---|
| CPU (reference) | scalar Rust | **Correctness reference only** — passes the 19K quality threshold but is too slow for promotion | Python/Rust fixtures pin LMCache revision `b5d109e`; 19K gate below |
| Metal (Apple GPU) | `cubecl/wgpu` | **Prototype-only spike** — its affine+delta path is not the LMCache algorithm and is not eligible for promotion | Historical spike parity on M2 Max only |
| CUDA (NVIDIA) | `cubecl/cuda` | **Not implemented** — compile-only gate where a toolchain exists | No CUDA hardware in the fleet lane; no runtime claim is made |
| HIP/ROCm (AMD) | `cubecl/hip` | **Not implemented** — compile-only gate where a toolchain exists | No AMD hardware in the fleet lane; no runtime claim is made |

Nothing may be marked implemented until it runs on real hardware and
matches the CPU reference bit-for-bit. Compile-only checks prove the
kernel lowers; they say nothing about the hardware.

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

## Sequencing after this slice

1. Preserve the scalar implementation and its pinned fixtures as the
   deterministic oracle for device kernels.
2. Add a Skippy-owned compressed-page import/export contract at the native KV
   boundary. Skippy keeps ownership of cell allocation, rollback, tensor
   layout, and session-position commit.
3. Expose one optional backend-registry codec hook from Metal and CUDA/HIP.
   Import passes validated compressed records plus tensor/cell-run
   destinations; it never exposes backend-specific device pointers through the
   public Rust or C ABI.
4. Implement the parallel arithmetic path on Metal and decode directly into
   resident K/V storage, including transposed-V destination addressing. Prove
   fixture parity before another 19K gate.
5. Add encode from resident K/V storage and copy only the compact archive back
   to the persistence layer.
6. Compile the same CUDA-family source through CUDA and HIP/ROCm, then qualify
   each backend on real hardware before marking it implemented.

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
