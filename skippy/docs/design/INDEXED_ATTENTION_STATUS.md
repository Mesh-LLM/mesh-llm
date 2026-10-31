# Indexed KV attention: review candidate

Status on 2026-10-31: implementation available for review. All eight real-model
attention layers match existing dense CUDA attention exactly at the same logical
layout. Strict cache tests pass on CUDA and Metal. Comparison against the old
physical layout still fails; that diagnostic is retained below. Final performance
qualification and four-model CUDA coverage are in progress.

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

The reusable four-model CUDA/Metal/CPU validation work is tracked in
[todo.md](../../../todo.md).
