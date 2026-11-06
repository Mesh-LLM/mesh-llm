# Indexed KV attention: review candidate

## Current candidate: upstream per-owner row lists

The working candidate replaces the custom indexed operator with the per-owner
row-list and ownership-bitmap implementation from upstream PR29510, based on
`006fc936d1dcb8077c0bccfad31c72ee453960e8`. It stays on the existing upstream pin.
CPU, CUDA and Metal use `ggml_flash_attn_ext_rows` with one I32 row list per
query stream and the existing attention mask. Runtime admission probes the
actual assigned backend and stored KV representation; unsupported row layouts
retain ordinary attention on that backend. Model names do not select a path.
Radix ownership, recurrent checkpoints and persistent L3 remain separate.

The [layout-matched full-model control](https://mlflow.internal.generalinference.xyz/mlflow/#/experiments/6/runs/9562cd5443cd4bd78573dcf0eb61ca38)
passes all 23 mandatory comparisons byte-for-byte at the unchanged `1e-6`
bound for the fixed Qwen3.5-35B/F16 fixture. Vanilla kernels independently
reproduce the full PR's logits, saved states and restored continuations when
given the matching attention layout. This is a diagnostic intervention, not
untouched vanilla or universal model qualification. The original cross-layout
failure and two optional ordinary-restore failures remain recorded.

CPU validation passes 2,452 finite operator comparisons, 36 exceptional controls
and 294 native tests; one GPU-only test is explicitly skipped. Initial M1
validation passed bitmap/codec checks and 1,318 finite comparisons before the
unchanged operator bound failed at F16 head width 40. The inherited row-gather
copy loop omitted entries at head boundaries for widths 40 and 72. The generic
repair assigns copy chunks within each head, preserving the original mapping
for aligned heads. The repaired M1 diagnostic passes the complete unchanged
operator program and F16/Q8/F32 resident-model gates. Actual loaded-library
hashes match the sealed provider. F16 and Q8 use row nodes; stored F32 retains
ordinary attention with F16 attention inputs on Metal. The diagnostic reuses
unchanged host objects and replaces the Metal auxiliary shader; it is not the
shipping runtime. The subsequent production builds are attributed separately
below.

The final candidate also wires row inputs through both the full-attention and
sliding-window cache pools. Both ordinary and hybrid graph reuse refresh the
current pool inputs; each layer uses its existing window metadata. The new CPU
fixture exercises the actual two-pool graph, shared owners, reordered query
streams and graph reuse. It passes alongside the existing graph-input tests.
No model-family rule, new setting, kernel arithmetic change or CPU fallback was
added. Clean replay of all 104 patches produces native `aec61426350541d9f6c4762ef4ca1850e4f34f49`,
tree `22cbb3144e2926b15ab850016d6fce5eb40d2ceb`, queue digest
`11701d0ac354c8d240d4813e5ff60b83d07d4f9a2daecde45ebe8ccd6ef3f18f`.

The preceding `cc855ab2` candidate completed the early serving pairs below.
Every arm completed 69 requests and 18,713 output tokens with clean client and
server exits. These are single ordered diagnostics, not ABBAAB qualification,
generated-content parity or results for the final two-pool candidate.

| Backend / model at c4 | Skippy output tok/s | Standard llama output tok/s | Difference |
|---|---:|---:|---:|
| [CUDA / hybrid dense 4B](https://mlflow.internal.generalinference.xyz/mlflow/#/experiments/6/runs/da26189089e5493f9dbb5a59e8ce3c57) | 26.476 | 26.654 | -0.67% |
| [CUDA / hybrid MoE 35B](https://mlflow.internal.generalinference.xyz/mlflow/#/experiments/6/runs/30aa91d8233f4a599d8436f4673fa446) | 24.971 | 25.056 | -0.34% |
| [CUDA / attention MoE 30B](https://mlflow.internal.generalinference.xyz/mlflow/#/experiments/6/runs/7ae025fc9ac741758d0b0c9e604e2796) | 15.782 | 13.705 | +15.16% |
| [M3 Metal / hybrid dense 4B](https://mlflow.internal.generalinference.xyz/mlflow/#/experiments/6/runs/33bc478fecf44f3bbe8e5861610b4f99) | 26.613 | 26.228 | +1.47% |

Metal TTFT p95 remains worse in that pair (74.58 versus 29.81 seconds). The
dense CUDA pair failed in the benchmark client after its exact HTTP cache
gate passed; no partial throughput is qualified. Its failed attempt remains
retained. The original Gemma ordinary-attention resident failure reproduces
with the old `00983cf4` runtime at the same observed deltas; neither failure is
waived. The final candidate's actual two-pool GPU tests are separate evidence.

The [owning Metal runtime's targeted M1 gates](https://mlflow.internal.generalinference.xyz/mlflow/#/experiments/6/runs/fe742c2cf8944ef69e0fdce624475938)
pass all five native/admission tests with clean postchecks. Gemma F16 and Q8
each execute 13 full-attention and 13 sliding-window row nodes on MTL0;
resident fork logits have zero differences and both page comparisons are exact.
Dense F16/Q8 execute 36 row nodes, while stored F32 preserves the selected Metal
ordinary-attention conversion. These tests retain the original `1e-6` limit.

The [final CUDA runtime gates](https://mlflow.internal.generalinference.xyz/mlflow/#/experiments/6/runs/accb9ce48b87441888fc1f4fb100a99f)
pass the operator catalog, six resident storage pairs, exact four-lane/32-step
checkpoint continuation and fresh-process L3 restore. The producer reused
unchanged CUDA kernel objects and rebuilt the changed graph consumers; all 19
numeric exits and 59 artifact hashes pass audit.

The [final CUDA Gemma F16/Q8 gates](https://mlflow.internal.generalinference.xyz/mlflow/#/experiments/6/runs/a459675abb7240088b65f2892d0c663c)
execute all 26 attention layers through row nodes, with zero fork-logit
differences and exact restored pages. The [final dense CUDA gates](https://mlflow.internal.generalinference.xyz/mlflow/#/experiments/6/runs/602d580a52b443c79acd38c920766a80)
also pass F16, Q8 and mixed Q8/F16 with all 36 row nodes and the same unchanged
resident bound. These actual-model checks are distinct from the CPU two-pool
graph fixture and do not relabel the older ordinary-attention failures.

Remaining qualification uses one frozen final runtime per backend, all four
models and c1/c2/c4 in ABBAAB order. Audit complete work, source/model/runtime/
client identities, lifecycle, latency, cache reads and sampled memory before
publishing performance through the typed owner into existing MLflow experiment
6. Historical campaigns and failed attempts keep their original attribution.

The preceding candidate's cold replay produced native
`cc855ab2f443f77626ebf0f4901610dea737fd14`, tree
`7ff0e7d6b92bf3145b08dd8492b174d2c1bf7ee0`, queue
`a3628a88bae20819756d22e1fc2a68781de2cedae711ed9b08590e7cd9a3a116`.
The Skippy ABI remains 0.1.76; this GGML constructor change adds no Rust ABI
symbol. All performance evidence below remains attributed to its original
source and does not qualify this candidate.

## Preceding repair: retain ordinary attention execution

The automatic indexed graph is being withheld because it fails the unchanged
full-model baseline arithmetic contract. This also happens without saved cache
owners or checkpoint imports: fresh base/head controls have exact same-source
repeats, but cross-source maximum logit deltas are `0.223699808` at c1 and
`0.857750654` at c4, with zero and three greedy differences respectively.
These failures remain recorded; operator accuracy tests do not waive them.

The conservative native repair candidate is `00983cf474939f13c2d8a151efe7355518642906`,
tree `e709f20cf29ae4df087e4d20719fec5e267b554d`, queue
`72b73c5eaf35a818d8b38043f41955565e4786eca998e4209456951bf00b38db`.
It declines automatic indexed read plans at the common graph-admission point,
consistently during construction and reuse. Ordinary attention retains its
existing query layout, physical key extent and backend selection. This is not
a CPU fallback, model exception, new configuration switch or tolerance change.
The public indexed GGML operator remains independently available and tested.
Default-model fixtures explicitly require absent index inputs and the selected
backend; stored KV types and attention-input types are reported separately.

The preceding ordinary-graph candidate `4c133af3` matched the retained CUDA c4
canonical and fresh-context baseline exactly, but still differed at c1.
Graph-disabled controls reproduced the same c1 difference. Nsight found 234
registers and six VEC partial reductions in the baseline versus 150 registers
and nine partials in the modified kernel. The occupancy-based launcher changed
the partitioning without a selector change. This repair therefore preserves
the original ordinary VEC body separately from indexed row loading. Selection
depends only on the presence of index inputs; register and grid counts are not
hardcoded. Source-matched CUDA builds now pass the unchanged fresh c1/c4
baseline comparisons: all six repeat/cross comparisons have maximum logit
delta zero and no greedy differences (`job_mkz3w1kj6zqmpy5`). The retained
canonical c4 replay also passes all four comparisons exactly
(`job_wjm2d0ge3h65req`). Actual Nsight launches restore 234 registers/thread,
grid 1x6x16, block 32x4 and 16,640 shared bytes across all 256 observed VEC
decode launches (`job_vs97wb6qvq6z4fz`). All 51 downloaded artifacts across
these three controls passed independent size and SHA-256 verification.

Strict cross-placement resident continuation remains a separate open contract.
The actual Qwen3.5-4B CUDA fixture fails identically on retained ordinary `f71`
and candidate `4c133af3`; saved prefix and cold-export bytes pass, but computed
continuation logits and KV pages differ. Extra Llama-1B Metal tests also fail
the unchanged strict continuation comparison. A separate public-API control
against true unpatched llama.cpp reproduces the same F16 continuation deltas
on Metal; full trace comparison remains separate from equal delta summaries.
These failures remain recorded, rather
than being hidden by passing operator tests or output-count hashes.

Resident radix ownership, recurrent checkpoints, persistent L3, cooperative
hybrid scheduling and longest RAM/L3 prefix selection remain independent.
Restoring the ordinary graph also restores its physical-pool scaling cost.
The historical idle-prefix scaling improvements below therefore do not describe
this repair candidate. Exact numerical, mixed-KV, additional-model and AgentX
qualification of the repaired default path remains incomplete. The local Just
CLI/native package build passes at this exact native identity. A separate
true-vanilla Gemma2 Metal control crosses its 4,096-token sliding window with
a 6,144-token prefix and softcap 50: all six c1/c4 repeat/cross full-vocabulary
comparisons are exact, with all 26 attention layers on the selected GPU.
This is additional-model correctness evidence, not performance qualification.
ABI remains 0.1.76.

## One-lane resident-cache correction

Default native and MTP contexts now select unified KV even at one lane.
Previously, dense one-lane contexts selected non-unified KV while advertising
resident caching; native prefix saves then returned unsupported and repeated
HTTP requests reused zero tokens. Explicit non-unified attention contexts now
omit the resident capability flags that their save API cannot honor.

Native `11aa69c53e118958f3dc0ab0870f89f3d24d7897`, tree
`b085f1e7672c73be8ec821c84bdad7401600fe91`, queue digest
`0378e014f6c267fd03b078a285dff03fb25f42502e3fbed8907926010296299f`
replays cleanly. The expanded dense resident fixture passes on CPU and Metal:
default one-lane save, source release, byte-exact restore and next logits;
explicit non-unified one- and two-lane contexts report no resident support and
reject saves. CUDA and actual dense-model HTTP reuse are pending on this fix.
ABI 0.1.76 and the attention kernels are unchanged.

## Historical combined candidate with indexed vector decode

The earlier combined candidate integrated Jared Wyles's `18f0fa7a49142344c9297cfcbaf9a16352352189`.
For eligible indexed CUDA queries of at most two tokens, F16/BF16 KV can now
use the existing vector kernel; larger prefill and verification windows retain
their existing selection. The owning patch was regenerated to correct stale
mail-patch hunk counts after the guard removal. The combined candidate also
checks compiled CUDA softcap capabilities: unsupported V widths are refused;
supported wide shapes without an MMA softcap variant use the existing tile
kernel. D64 softcapping remains explicitly unsupported on CUDA and cannot count
as GPU qualification through CPU fallback.

Clean queue replay produces native `9e5ce4cd088fd30193702e7cb176857ef3f47dee`,
tree `bc57c588724ca8cfedb714ea357a5f3370b6068a`, queue digest
`ef74e663109626492be260e879ac0492b71a3963dab035d54d3e8b430eec0297`.
Expanded operator cases cover Q2, mixed F16/BF16, no-mask/no-sink execution,
softcap eligibility and four alternating metadata phases spanning graph replay.

CUDA correctness and performance qualification for this combined source are
recorded separately from the one-lane fix above. Source `382c6c62a` passes the
CUDA operator, real-model cache/L3, fresh-process and all-eight-layer FP64
gates; actual F16 attention uses the vector kernel. Its complete same-host c4
pair measures 26.805 Skippy versus 26.577 standard llama output tokens/s,
69 requests and 18,713 outputs per arm with zero errors. This single pair is
interim evidence, not a complete ABBAAB group. Its verified metrics and artifacts
are in [the existing MLflow experiment](https://mlflow.internal.generalinference.xyz/mlflow/#/experiments/6/runs/90be7573cb6c49fcb3cdc7032861f656).
The older measurements below remain attributed to their original source;
they do not qualify the new dispatch. The full four-model c1/c2/c4 ABBAAB sweep
will record a new campaign within the existing MLflow experiment
`agentx-skippy-vs-llama` (ID 6), preserving historical runs.

## Prior-source qualification (`1b3f2e6fc36d`)

The following results qualify the preceding source, before Jared's vector
selection change. All eight real-model
attention layers match existing dense CUDA attention exactly at the same logical
layout. Strict cache tests pass on CUDA and Metal. Comparison against the old
physical layout still fails; that diagnostic is retained below. The final CUDA
AgentX arm completes at 26.91 output tokens/s versus a fresh same-host standard
vanilla arm at 26.34. All four selected models pass their CUDA KV-format matrix.

## Change

The new core patch, `0013-feat-ggml-read-request-local-KV-indices-in-flash-att.patch`,
extends the existing `FLASH_ATTN_EXT` operation with per-request physical row
indices and logical lengths. CUDA uses the existing vector, tiled, and MMA
attention implementations; Metal and CPU implement the same indexed contract.
Quantized rows are read directly without converting the complete physical pool.
Allocator ownership, persistent L3 representation, and the public Skippy ABI
are unchanged.

The branch also includes resident prefix ownership and frontend KV admission
changes preceding this kernel work. The kernel patch is a separate commit so
reviewers can inspect it independently.

## Validation

- Clean replay from native upstream `620d2225cc436ab0b001e3200e590e3b286ef9ab`
  produces native HEAD `1b3f2e6fc36d020260b61c672e168da51d8310a7`, tree
  `bdb3081b92c7af8261500aec37af89046104201c`, queue digest
  `4035b2c99d9ddc9c9597f363210e7734a884b84f5f3b0b9b18e1730d51710740`.
- Operator comparisons pass: 910 CPU, 654 Metal, and 608 CUDA. CUDA graphs are
  enabled. The existing per-request NMSE limit remains `5e-4`. Cases already
  unsupported by a backend's dense operator remain explicitly unsupported.
- Real Qwen3.5-4B smoke tests pass for all eight supported same-type Metal KV
  formats and all 64 CPU mixed K/V pairs. Graph checks require indices, lengths,
  the requested KV type, and the selected backend throughout prefill and decode.
  These are execution smoke checks, not cross-path numerical parity evidence.
- The four selected Q4_K_M models each pass all 64 CUDA mixed K/V pairs: dense
  Qwen3-4B (36 attention layers), hybrid Qwen3.5-4B (8), Qwen3-Coder-30B (48),
  and hybrid MoE Qwen3.5-35B (10). Source-identical Spark 000907 runs cover the
  middle two; the final performance runtime on Spark 000137 covers the other
  two. Each matrix loads its model once and checks every executed attention
  graph's indices, lengths, K/V types and GPU backend. The final checked-in
  driver additionally requires the exact expected attention-layer count and
  all-vocabulary finite logits; earlier frozen smoke drivers check NaN-free
  logits and a finite greedy-selected logit. Their proof scopes remain distinct.
- Final CUDA proof job `job_x7fs8vb280zjz00`, run `run_nd0p378y1k069ct`, passes
  all eight real-model attention-layer replays, resident ownership checks, and
  the unchanged Qwen3.5-4B chat-parity and growing-checkpoint tests. The final
  local Metal runtime passes those same two cache regressions and exact-leaf
  runtime discovery.
- The prior local native suite completed with 285 executed tests passing and
  one explicitly skipped long GPU test.

## Retained physical-layout CUDA diagnostic

JY job `job_yg6c28ny2f5k9hn`, run `run_wkzw1rnq1ccps94`, on Spark 000137
built and packaged this candidate and passed operator and small resident tests.
The actual Qwen3.5-4B proof imported the reference's exact canonical context,
verified all eight physical sequence layouts and active KV/recurrent bytes,
and confirmed indexed execution on CUDA for every attention layer.

For four lanes with 32 teacher-forced decode steps, dense/dense and
indexed/indexed repeats each have zero logit delta and zero greedy differences.
Dense/indexed comparisons fail the unchanged `1e-6` full-vocabulary gate:
maximum logit delta `0.284694672`, with one greedy difference. Divergence is
already present at the first decode step; this is not only accumulated drift.
That failed comparison remains recorded and has not been relabeled as a pass.

## Controlled attention replay

Jobs `job_fp9kyakwncmatr5` and `job_nxze85xz8tsvhd5` isolate the first actual
attention layer using the saved canonical state, without another prefill.
Queries, complete physical K/V payloads, selected rows, and selected masks are
byte-identical across the two model executions. The newly written token is
present in each request's 23,498 visible rows. Capturing the layer does not
change either execution's final logits.

Replaying the captured inputs produces these results:

- Original physical dense replay reproduces the captured dense output exactly.
- Indexed replay reproduces the captured indexed output exactly.
- Existing dense CUDA attention with compact request-local K/V produces
  **bit-identical output to indexed attention**, across all 16,384 output values.
  Both select the same MMA specialization, launch geometry, and logical extent.
- Changing only the existing dense operation's query layout changes its output
  by up to `0.00948334`. Compacting its physical extent then changes its output
  by up to `0.00588727`. Neither comparison uses the indexed reader.

The existing CUDA kernel accumulates probability-times-value products in half
precision. Layout and span change its partitioning and rounding history. An
independent FP64 calculation on the worst differing head finds errors in both
layouts; the compact/indexed result is not uniformly more accurate than the old
physical result. The ordinary CPU FA implementation also uses half accumulation
for this F16-V shape, so it is not an independent FP32 oracle here.

The completed controls retain the existing numerical limits: indexed versus
existing dense attention at the **same logical geometry** (`1e-6`), independent
FP64 attention references using the existing operator per-request NMSE limit
(`5e-4`), and unchanged full-model checkpoint/cache assertions. All eight actual
attention layers match compact dense bit for bit. Across every query head and
layer, CUDA's worst per-request NMSE is `3.321e-6` against true F32 queries or
`3.324e-6` against rounded F16 queries; the new CPU indexed reader's worst values
are `1.356e-7` and `8.187e-10`, respectively. The old physical-layout full-model
comparison remains a separate failed diagnostic; its tolerance was not raised.

## Measured cost and final host change

The v3 pressure run (`job_01ge0nmkfdfp0xg`) completed all 25 requests. Keeping
additional idle prefixes no longer increases attention work: three equal cold
waves took `28.69 / 28.59 / 28.42 s`, versus `33.15 / 53.81 / 74.22 s` before
indexed reads. CUDA attention time stayed `4.26 / 4.28 / 4.26 s`, versus
`10.11 / 29.92 / 49.57 s` before. Standard vanilla's waves took
`24.36 / 24.74 / 25.12 s`, so removing that scaling cost alone did not establish
parity. Warm repeats retained real prefix hits. These are profiled pressure
diagnostics, not full AgentX throughput scores.

The final host change fills zero-bias causal masks in contiguous spans instead
of calling the F32-to-F16 conversion helper once per visible key. Positive ALiBi
keeps its existing calculation. A bounded CPU microbenchmark verifies identical
mask bytes, including gaps, empty rows, padded rows and positive ALiBi. For four
requests with 23,552 padded keys, measured construction falls from `14.54` to
`0.53 ms` at 64 queries and from `29.14` to `1.16 ms` at 128 queries. Actual
captured CUDA inputs, attention outputs and step-zero logits remain byte-identical
to v3; no GPU kernel changed in this last step. The final pressure run
(`job_7ap9pd6carpy2b2`) passed: equal cold waves took
`27.19 / 26.96 / 27.34 s`, while attention remained `4.24 / 4.24 / 4.31 s`.
The unchanged GPU cost and lower total time agree with a host-side improvement.
The time without GPU activity falls to `1.63 / 1.36 / 1.43 s`, from
`3.09 / 2.97 / 2.87 s` in v3; standard vanilla measures
`0.73 / 1.06 / 1.39 s`. Most of the remaining pressure gap is GPU-active work,
including indexed attention's roughly `4.24 s` versus vanilla's `2.88 s`.
This still trails the saved standard-vanilla pressure control. Full fixed-cohort
job `job_agr2k8yy2ry29qe` is the separate performance qualification.

## Full fixed-work replay

The final candidate job `job_agr2k8yy2ry29qe`, run `run_spranb971p0czpp`,
completed successfully on Spark 000137: **26.9097 output tokens/s**, all 69
requests and 18,713 output tokens, zero errors. It read 3,562,313 cached prompt
tokens out of 3,940,542 presented tokens, the same cache work as the preceding
admission-fixed baseline. Peak sampled process RSS was 10.38 GiB.

The fresh same-host standard vanilla confirmation (`job_jhe5qzycenack95`, run
`run_ghdjyknby3jdsmd`) also succeeds: **26.3370 output tokens/s**, 69 requests,
18,713 outputs and zero errors. It reads 3,566,742 cached prompt tokens and
peaks at 13.92 GiB sampled process RSS. Standard uses four separate KV slots;
Skippy retains unified KV. Both resolve 20 generation and 20 batch threads,
batch 2,048, microbatch 512 and F16 KV. Standard's thread settings resolve from
defaults and are verified in its log; Skippy passes them explicitly.

This sequential single pair puts Skippy 2.17% ahead and supports throughput
parity on this workload, not a universal speedup claim. The previous standard
control was 26.8779, which illustrates run variation. This is not a six-arm
ABBAAB campaign or qualification of every model/concurrency. Source, runtime,
model, fixed-work, raw artifacts and GPU clock/temperature evidence are retained
in the independent comparison receipts.

The reusable four-model CUDA/Metal/CPU validation work is tracked in
[todo.md](../../../todo.md).
