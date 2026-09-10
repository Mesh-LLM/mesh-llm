# CacheGen Backend Plan (#1652)

Status: proposal backed by the spike in this slice. Owner: jian yang.
Reviewed against: #1652 scope, scama's directives of 2026-09-10 (v4
contract, CPU+Metal parity, six measurements, stop rule).

## Where each backend stands

| Backend | CubeCL runtime | Status | Evidence |
|---|---|---|---|
| CPU (reference) | `cubecl/cpu` | **Real, verified** | Spike parity pass on this machine |
| Metal (Apple GPU) | `cubecl/wgpu` | **Real, verified** | Spike parity pass on M2 Max (wgpu Metal adapter) |
| CUDA (NVIDIA) | `cubecl/cuda` | **Not implemented** — compile-only gate where a toolchain exists | No CUDA hardware in the fleet lane; no runtime claim is made |
| HIP/ROCm (AMD) | `cubecl/hip` | **Not implemented** — compile-only gate where a toolchain exists | No AMD hardware in the fleet lane; no runtime claim is made |

Nothing may be marked implemented until it runs on real hardware and
matches the CPU reference bit-for-bit. Compile-only checks prove the
kernel lowers; they say nothing about the hardware.

## The six spike measurements (2026-09-10, M2 Max, 4096x128 tile,
## 524,288 f16 values = 1,048,576 raw bytes)

| Metric | cubecl-cpu | wgpu (Metal) |
|---|---|---|
| Cold JIT/compile (quantize+delta; undelta+dequantize) | ~0 ms / ~0 ms | ~0 ms / ~0 ms |
| Warm dispatch (avg per launch) | 6 us / 4 us | 5 us / 4 us |
| Host->device bytes | 2,097,152 (f32 tile) | 2,097,152 (f32 tile) |
| Device->host bytes | 4,194,304 (u32 symbols + f32 rebuilt) | 4,194,304 |
| Peak temporary device memory | 4,194,304 | 4,194,304 |
| Encoded-size ratio (rANS over device symbols / raw) | **0.075 (13.3x)** | **0.075 (13.3x)** |
| Output equality vs CPU reference | symbols + values exact | symbols + values exact |

Caveats, stated rather than buried: cold-JIT is ~0 here because CubeCL
had a warm compilation cache from the compile pass of this same run; a
true cold-start number needs a cleared cache directory and belongs in
the quality/performance gate. Warm dispatch at 4096x128 is dominated by
launch overhead, not bandwidth; the per-column scan is the correct
bit-exact baseline, not the fastest shape (a parallel scan is the
follow-up). The 2x H2D cost versus raw f16 bytes exists because the
spike uploads f32; shipping f16 halves it and is a trivial follow-up.

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

1. Native runtime-format passthrough remains the exact control arm
   (unchanged #1652 scope).
2. CacheGen quality/performance gate versus native on the ~19K acceptance
   workload, matched release builds, before any wiring.
3. Only then: CUDA and HIP/ROCm on real hardware, each proven against
   the CPU reference bit-for-bit before either is marked implemented.
