# CacheGen Backend Plan (#1652)

Status: proposal backed by the spike in this slice. Owner: jian yang.
Reviewed against: #1652 scope, scama's directives of 2026-09-10 (v4
contract, CPU+Metal parity, six measurements, stop rule).

## Where each backend stands

| Backend | CubeCL runtime | Status | Evidence |
|---|---|---|---|
| CPU (reference) | `cubecl/cpu` | **Quantize+delta kernel spike verified** — full CacheGen backend/container path unimplemented; rANS stays CPU | Spike parity pass on this machine |
| Metal (Apple GPU) | `cubecl/wgpu` | **Quantize+delta kernel spike verified** — full CacheGen backend/container path unimplemented; rANS stays CPU | Spike parity pass on M2 Max (wgpu Metal adapter) |
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

## Sequencing after this slice

1. CacheGen quality/performance gate versus native on the ~19K acceptance
   workload, matched release builds, before any wiring.
2. Only then: CUDA and HIP/ROCm on real hardware, each proven against
   the CPU reference bit-for-bit before either is marked implemented.
