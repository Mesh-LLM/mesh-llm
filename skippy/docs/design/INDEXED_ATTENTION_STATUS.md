# Indexed KV attention: review candidate

Status on 2026-10-31: implementation available for review; actual-model CUDA
numerical parity fails. This candidate is not qualified for deployment or
throughput comparisons.

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
Performance runs and other-model fanout remain blocked pending diagnosis.

The reusable four-model CUDA/Metal/CPU validation work is tracked in
[todo.md](../../../todo.md).
