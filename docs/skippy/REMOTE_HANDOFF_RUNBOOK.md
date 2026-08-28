# Remote Handoff Runbook (PD disaggregation, Phase 3 step 1)

`skippy-correctness remote-handoff` runs the full-prefill → full-decode
handoff between two machines: the sender prefills a prompt, exports the
continuation state, streams it to the receiver in digest-verified segments,
and the receiver imports nothing until the commit record validates
completeness — then both sides greedy-decode the same continuation and the
tokens are compared one-for-one. With `--baseline` the receiver also
measures prefill-in-place, so one run yields the disaggregated-vs-local TTFT
comparison with the counters from `EXPERIMENTS.md` (export/transfer/import
bytes and seconds, attach, first-decode).

## Requirements

- Same model file, `--ctx-size`, `--layer-end`, lane count, and payload kind
  on both sides (validated in the handshake; mismatches are rejected).
- A fresh native ABI build: the full-state header parsing was fixed in patch
  0023 (2026-08-27), so stale `.deps/llama-build` archives fail with
  "full-state import restored native position 0". Run `just llama-prepare &&
  just llama-build` if in doubt, then `cargo build -p skippy-correctness`.
- Payload kinds: `full-state` (default; dense attention models) or
  `kv-recurrent` (hybrid models — untested until a hybrid package is
  available, see #1425).

## L3 store integration

Pass `--store-dir <path>` on either side to route the handoff through the
L3 segment store (`skippy-cache::l3`): the sender spills its exported state
(segments + manifest, off the transfer critical path), and the receiver
write-behinds incoming segments to disk and imports via the store's
`assemble` path — every segment digest, the tiling, and the whole-payload
digest re-verified. `--store-budget-bytes` caps the on-disk footprint
(oldest manifests evict first). The manifest records the sender's reference
continuation, so state in the store is self-verifying.

Restart survival: reattach from a store with no network and no exporter —

```bash
target/release/skippy-correctness remote-handoff --role restore \
  --store-dir <path> --model <same.gguf> --layer-end <n_layers> \
  --ctx-size 8192 --n-gpu-layers 99 --decode-tokens 32 \
  --report-out restore-report.json
```

`--manifest <payload-digest>` selects a specific manifest (default:
newest). The restore refuses manifests whose `exact_state_identity` (the
numerical identity: weights, cache dtypes, flash-attn, backend, layer
range, context shape — never stage/topology placement) does not match the
local configuration.

## Streaming handoff (overlap + two-phase commit)

Pass `--streaming` on **both** sides: the sender exports the KV page for
each prefill chunk (`--stream-chunk-tokens`, default 512) and streams it
while later chunks compute, so transfer and the receiver's import hide
inside the prefill wall. The recurrent snapshot (hybrid families) is the
serialized tail. The receiver stages pages into a session as they arrive
but **cannot generate until the commit record validates** page tiling,
counts, and the running payload digest — any failure drops the staged
session. The report's `overlap_wall_ms` plus the receiver's
`attach_residual_ms` and `first_decode_ms` compose the streaming TTFT.
Restore from a page-stream manifest re-imports page by page (pass
`--streaming` to restore too so identity matches).

## Peer fetch (`skippy-kv/1`) — cross-node prefix reuse

Any node can serve its store and any node can pull by digest:

```bash
# node A: serve the store (no model load)
target/release/skippy-correctness remote-handoff --role serve \
  --listen 0.0.0.0:19092 --store-dir <path> --model <same.gguf> --layer-end <n>

# node B: pull the newest manifest, then restore + decode from it
target/release/skippy-correctness remote-handoff --role fetch \
  --peer <node-a>:19092 --store-dir <local-path> \
  --model <same.gguf> --layer-end <n> --ctx-size 8192 --n-gpu-layers 99 \
  --decode-tokens 32 --report-out fetch-report.json
```

Fetches are idempotent (content-addressed: held segments transfer zero
bytes) and every segment is digest-verified before it lands locally.

## Serving-path L3 tier

The same store backs real serving: set `SKIPPY_L3_DIR=<path>` (and
optionally `SKIPPY_L3_BUDGET_BYTES`) on a stage with the exact-state prefix
cache enabled, and recorded exact-state entries write through to disk while
radix misses fill back from it — prefix reuse that survives restarts and
RAM eviction. The tier identity is the radix namespace hash, so a
configuration change refuses stale state.

## Two-machine run

Use a **release** build for measurements (`cargo build --release -p
skippy-correctness`) — debug-build byte handling distorts transfer and
export timings.

Receiver (decode node) first — it loads the model, then listens:

```bash
target/release/skippy-correctness remote-handoff --role recv \
  --listen 0.0.0.0:19081 \
  --model <same.gguf> --layer-end <n_layers> --ctx-size 8192 \
  --n-gpu-layers 99 --prefix-token-count 4096 --decode-tokens 32 \
  --report-out recv-report.json
```

Sender (prefill node), once the receiver prints `ready`:

```bash
target/release/skippy-correctness remote-handoff --role send \
  --peer <receiver-ip>:19081 \
  --model <same.gguf> --layer-end <n_layers> --ctx-size 8192 \
  --n-gpu-layers 99 --prefix-token-count 4096 --decode-tokens 32 \
  --baseline --report-out send-report.json
```

The sender's report is the primary artifact: `ttft_disaggregated_ms`
(prefill + export + transfer + attach + first decode) vs `ttft_local_ms`
(receiver's prefill-in-place + first decode), `ttft_speedup`,
`transfer_gbps`, and `matches` (exact token agreement, the correctness
gate). Non-zero exit on mismatch unless `--allow-mismatch`.

## Sweep for the perf matrix

Start the receiver once with `--accept-count <n> --allow-mismatch` (it
serves n handoffs, writing `report-1.json … report-n.json`), then drive the
sender side with `scripts/remote-handoff-sweep.sh`:

```bash
scripts/remote-handoff-sweep.sh <receiver-ip>:19081 <model.gguf> <layer-end> \
  out/ 512 2048 4096 8192
```

It prints a summary table (state MiB, link Gbps, per-phase ms, TTFT
disaggregated vs local, speedup, match). Run both role assignments (fast box
sends, then fast box receives). Keep `--ctx-size` at least prefix + decode
tokens on both sides. `transfer_gbps` on the Thunderbolt bridge tells you
whether the link, not the runtime, bounds the handoff.

Timing caveat: Metal execution is asynchronous, so `source_prefill_ms` can
under-report with the balance absorbed into `state_export_ms` (export
synchronizes). The TTFT aggregates are correct; per-phase attribution
between those two columns is approximate.

## Interpreting

- `matches: true` — exact-state handoff is deterministic; the correctness
  half of the #1427 step-1 gate.
- `ttft_speedup > 1` at some prefix length — disaggregation pays on this
  pair; the break-even length feeds the Phase 4 cost gate.
- This prototype transfers after prefill completes (no chunk streaming) and
  runs one request; it is the measurement harness for the EXPERIMENTS.md
  falsifier, not the serving integration.
