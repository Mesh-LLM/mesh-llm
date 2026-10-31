# Indexed KV attention: review candidate

Status on 2026-10-31: implementation available for review. The first real-model
attention layer matches existing dense CUDA attention exactly with the same
logical layout. Comparison against the old physical layout still fails. Full
model and cache validation remains in progress; no new throughput is qualified.

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
  produces native HEAD `3e756354af978f8a3432a22df7e1465363538ee8`, tree
  `d9963c3bedb9f1d6658821784a897c149ad1bccf`, queue digest
  `194c5d51170a0e9a47dbe8268f1a2422323e8088fb0cd68aa20ad52f9dd94c88`.
- Operator comparisons pass: 910 CPU, 654 Metal, and 608 CUDA. CUDA graphs are
  enabled. The existing per-request NMSE limit remains `5e-4`. Cases already
  unsupported by a backend's dense operator remain explicitly unsupported.
- Real Qwen3.5-4B smoke tests pass on CPU and Metal for all eight supported
  same-type KV formats. Graph checks require indices, lengths, the requested
  KV type, and the selected backend throughout prefill and decode. These are
  execution smoke checks, not cross-path numerical parity evidence.
- The prior local native suite completed with 285 executed tests passing and
  one explicitly skipped long GPU test.

## Blocking CUDA result

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

The next gates retain the existing numerical limits: indexed versus existing
dense attention at the **same logical geometry** (`1e-6`), independent FP64
attention references using the existing operator per-request NMSE limit
(`5e-4`), and unchanged full-model checkpoint/cache assertions. These controls
are being extended across all attention layers. The old physical-layout
full-model comparison remains a separate failed diagnostic; its tolerance is
not being raised to qualify the candidate.

The reusable four-model CUDA/Metal/CPU validation work is tracked in
[todo.md](../../../todo.md).
