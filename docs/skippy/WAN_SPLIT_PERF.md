# WAN Split Performance Model

This documents the throughput/latency model for Skippy stage-split serving over a
network, and the measured evidence behind it. Use it to predict whether a given
model + link + node count will be compute-bound or latency-bound, and therefore
whether adding stages helps or hurts.

## Single-stream per-token cost

For one in-flight request (generation concurrency = 1), the stages run
**serialized per token**: stage 0 computes its layers, forwards activations to
stage 1, ... the last stage produces the token, and the token/return path walks
back. Total compute across stages equals the whole model, so splitting does not
reduce single-stream compute — it only adds network hops.

```
TPOT ≈ C_total + (S - 1) · 2 · RTT + (S - 1) · P
```

- `TPOT` — time per output token (ms)
- `C_total` — compute time for all layers of one token (ms); ≈ solo single-GPU
  decode ms/token, independent of `S`
- `S` — number of pipeline stages (nodes in the split)
- `RTT` — round-trip time between adjacent stages (ms)
- `2 · RTT` — activation-forward hop + token/return hop per inter-stage boundary
- `P` — per-boundary protocol/serialization overhead (ms)

Single-stream throughput ceiling:

```
tok/s_max = 1000 / TPOT
```

### Compute-bound vs latency-bound

Define the network share of a 2-stage split:

```
network_fraction = (2 · RTT + P) / TPOT
```

- `network_fraction` high (→1): **latency-bound**. Adding stages makes it worse
  (each stage adds `2·RTT + P`). Minimize stage count; use speculation.
- `network_fraction` low (→0): **compute-bound**. `C_total` dominates and is not
  reduced by splitting single-stream — but concurrent/batched load benefits from
  pipeline overlap (see below).

## Measured evidence (2026-07-18)

2-node split, `meshllm/GLM-4.7-Flash-MTP-GGUF:Q4_K_M` (48 layers, MoE),
M5 Max (Metal, layers 0..25) ↔ RTX A5000 Melbourne (CUDA, layers 25..48),
direct iroh hole-punch, RTT ≈ 20 ms.

| Quantity | Value |
| --- | ---: |
| Solo GPU decode (all layers, no network) `C_total` | 12.9 ms/token (77.3 tok/s) |
| 2-stage split observed `TPOT` | 57.8 ms/token (~17 tok/s) |
| Split overhead vs solo | +44.9 ms/token |
| `2 · RTT` at 20 ms | 40 ms |
| Implied protocol `P` | ~4.9 ms |

Decomposition of the 57.8 ms/token: `12.9 (compute) + 40 (2·RTT) + 4.9 (protocol)
= 57.8`. The model closes to the measured value.

**Verdict for this workload: latency-bound.** Compute was ~22% of per-token time;
network round-trips were ~70%. A small MoE (few active params/token) has low
`C_total`, so `network_fraction ≈ 0.78`.

### Consequence: adding a stage here would slow it down

At RTT 20 ms, each extra stage adds ~44.9 ms/token. Going 2→3 stages:

```
TPOT(3) ≈ 12.9 + 2 · (2·20 + 4.9) = 12.9 + 89.8 = 102.7 ms/token  (~9.7 tok/s)
```

Only justified if a 3rd stage is needed for memory residency, or under
concurrent load, or if `C_total` per stage is large (dense big models).

## When does adding a stage help?

### 1. Memory residency
If the model + KV cache does not fit in `S-1` nodes' VRAM, you must add stages.
This is a correctness constraint, not a speed choice.

### 2. Concurrent / batched throughput (pipeline overlap)
With `N` concurrent requests and a pipeline of `S` stages, stages work on
different requests simultaneously. Aggregate throughput approaches:

```
throughput_agg ≈ min(N, S) · (1 / max_stage_compute)   (network-hidden regime)
```

when compute per stage ≥ inter-stage latency, so hops overlap with compute. This
is the regime where more stages raise aggregate tok/s — but it needs enough
concurrency and enough per-stage compute to hide `2·RTT`.

### 3. Compute-bound models (dense, large)
For a dense model, `C_total` is large and splitting across `S` stages makes each
stage's compute ≈ `C_total / S`. Adding a stage helps single-stream throughput
only while per-stage compute still dominates the added `2·RTT`:

```
add a stage if:  C_total / (S·(S+1)) > 2·RTT + P
```

i.e. the compute saved per token by finer splitting must exceed the latency of
the new hop. Low-RTT links and heavy dense models satisfy this; WAN + small MoE
do not.

## Speculation as the WAN lever

Speculative decoding (native MTP + N-gram) commits `k` tokens per verify
round-trip, amortizing the fixed `2·RTT` boundary cost across accepted tokens:

```
TPOT_spec ≈ C_verify + (S-1)·2·RTT / k_accepted + (S-1)·P
```

where `k_accepted` is the mean accepted tokens per verify window. This is why the
measured MTP + N-gram gain grows on longer/coding output (higher `k_accepted`):
+13–19% over MTP-off at RTT 20 ms.

## Planning checklist

1. Estimate `C_total` from a solo run (or per-layer compute × layers).
2. Measure `RTT` between candidate nodes.
3. Compute `network_fraction` for `S=2`.
4. If latency-bound: use `S=2`, enable speculation, prefer lower-RTT peers.
5. If compute-bound or memory-forced: increase `S`, verify
   `C_total/(S·(S+1)) > 2·RTT + P` still holds before adding each stage.
6. Under concurrency, prefer more stages for aggregate throughput once per-stage
   compute ≥ `2·RTT`.

## Speculative recovery cost over WAN (measured)

Enabling N-gram speculation on a large MoE (MiniMax-M2.7, 2-stage M5↔Melbourne,
~18ms RTT, agentic OSL512) was **~40% slower** than no speculation (≈15 vs ≈17
tok/s) despite a **high 0.89 token accept rate**. The cause is recovery cost on
rejected windows, not proposal quality.

Measured (metrics-server, coordinator stage-0 spans):

- accept_rate 0.89, accepted 660 / proposed 2214
- full_accept_windows 45, **early_reject_windows 231**, rejected_windows 234
- **recovery_restores 231**, recovery_ms ≈ 18,971, recovery_restore_downstream_wait_ms ≈ 4,620
- **window_shrinks 0** despite 231 early rejects — the adaptive window stayed at
  12 the entire run

Mechanism (verified in `frontend/embedded_generation.rs` +
`frontend/embedded_execution.rs`): each early-reject window pays **two serial
extra WAN round-trips** — `restore_embedded_stage_session` (write + wait for
ACK) then the repair `execute_embedded_stage_message` (write + wait for reply).
A rejected window therefore costs ~3 round-trips to commit what plain decode
commits in 1. Over WAN those round-trips dominate.

The smoking gun is `window_shrinks 0`: the adaptive policy never narrowed the
window under a sustained reject storm, so it kept proposing deep, kept
rejecting, and kept paying the 3× round-trip recovery.

Fix directions (not required for merge — speculation is opt-in and off by
default; this only affects deployments that explicitly enable N-gram):

1. Adaptive window must actually shrink on early-reject (it did not here).
2. Fuse restore+repair into a single round-trip instead of two serial waits.
3. For single-token repair (`repair_input_count == 1`), avoid the separate
   restore round-trip entirely (piggyback the correction).

Takeaway for large-model MTP work: over WAN, speculation only pays off when the
reject/recovery rate stays low. Native MTP (high acceptance) plus a
reject-responsive adaptive window is the regime to target; blind deep N-gram
proposal on a latency-bound split is counterproductive.
