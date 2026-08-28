# Prefill/Decode Disaggregation — Implementation Plan

Status: draft for review. Companion to #1427 and the #skippy-radix-L1-L2-L3 workstream.

Progress (2026-08-28) — implemented, tested (752 tests across the four
crates), verified end-to-end on loopback:

- **L3 substrate** — `skippy-cache::l3`: content-addressed exact-state
  segment store (ordered manifests, completeness gate, idempotent puts,
  capped budget, prefix index). `exact_state_identity` implements the
  numerical half of the numerical-vs-placement identity split.
- **Radix L2/L3 integration** — `skippy-cache::tier::L3Tier` +
  `skippy-server::kv_integration`: exact-state records write through to the
  durable tier and radix misses fill back from it (re-warming RAM off the
  request path). Enabled in serving via `SKIPPY_L3_DIR`
  (+ `SKIPPY_L3_BUDGET_BYTES`); identity-guarded, best-effort on disk
  failure.
- **`skippy-kv/1` peer fetch** — `skippy-cache::l3_remote` + the ALPN in
  `skippy-protocol`: serve a store to peers, pull manifests and segments by
  digest with per-segment verification; idempotent re-fetch moves zero
  bytes. Harness roles `serve`/`fetch` demonstrate cross-node prefix reuse
  (fetch a peer's prefilled state, decode it, byte-exact).
- **Streaming handoff with two-phase commit** — `remote-handoff
  --streaming`: KV pages export per prefill chunk and stream while later
  chunks compute; the receiver stages pages into a session as they arrive
  but cannot generate until the commit record validates tiling, counts, and
  the running digest; uncommitted state is dropped on any failure. The
  recurrent snapshot is the serialized tail, as the byte math predicts.
- **Cost-based phase placement** — `skippy-topology::phase_placement`:
  role assignment from gossiped compute/bandwidth signals and the
  `HandoffCostModel` break-even gate (per-token KV slope, fixed
  recurrent floor, link throughput, overlap fraction).
- **Harness** — `skippy-correctness remote-handoff` roles
  send/recv/restore/serve/fetch with EXPERIMENTS.md counters, TTFT
  baseline, per-connection reports, and a sweep script. See
  REMOTE_HANDOFF_RUNBOOK.md.

Remaining, gated on other workstreams by #1427's own sequencing:
multi-request serving waits on the #1416 iteration-level scheduler
cutover; split-prefill → collapsed decode is the step-6 generalization;
routing the openai ingress through `phase_placement` lands with #1416.

## Goal

Serve a request with prefill on a compute-strong node and decode on a
bandwidth-strong node, with continuation state streamed between them while
prefill is still running, as a per-request routing decision — not a static
fleet mode. Short prompts keep prefilling in place; long prompts hand off.
Unlike layer split there is no per-token network dependency: the decoder
holds the full model and generates locally.

Lab target (per #1427): M3 Ultra prefill → M1 Ultra decode,
Nemotron 3 Super Q4 + MTPv2 — with the reverse arm (M1 prefill → M3 decode)
in the perf matrix so role assignment is settled by measurement, not
assumption.

Prior art: EXO 1.0 shipped exactly this shape (DGX Spark prefill → M3 Ultra
decode, layer-by-layer KV streaming) in Oct 2025, and every datacenter stack
(vLLM, SGLang, TRT-LLM, Dynamo) has a PD mode. The concept needs no proving.
What is open, and what this plan targets: PD disaggregation in the
llama.cpp/GGUF world, composed with continuous batching, chunked prefill,
admission control, and quantized KV caches — as a byproduct of a general KV
mobility layer that also gives cross-node prefix-cache reuse, which EXO does
not have.

## Design stance

**Disaggregation is remote prefix restore.** We already have the exact shape
of the decode-side mechanism: `ProbePrefill` / `TryRestorePrefillDecode`
control frames let a driver attach a sequence whose KV was produced earlier,
with each stage restoring from its own `UnifiedRadixCache`. The only thing
missing is that today the cache a stage restores from must be local. If the
radix cache gains an L3 tier whose pages can live on disk **or on a peer**,
then "prefill over there, decode here" is just "restore a prefix whose pages
happen to be remote." One mechanism, three features: restart-surviving prefix
cache, cross-node prefix reuse, PD disaggregation.

This is deliberately **not** a revival of the wire
`StateExport`/`StateImport` path (`binary_messaging/connection.rs:285` rejects
it, and should keep rejecting it). State moves over a dedicated backpressured
QUIC handoff stream, outside the stage activation lanes, addressed by content
digest. Two consequences of the existing plumbing must be designed out:
`MAX_STAGE_STATE_IMPORT_BYTES` (512 MiB) caps useful contexts, so the handoff
needs chunked streaming export/import APIs (none exist today —
`kv_pages.rs` is whole-buffer); and the transfer must be **two-phase
committed** on the decode side, so partially imported state can never
generate.

**The handoff is hybrid state, not just KV.** The lab target makes this
unavoidable: Nemotron 3 Super is 8 attention + 40 Mamba + 40 MoE layers, so
an exact handoff carries KV pages + recurrent/SSM + conv state + position
metadata + MTP context/bookkeeping + committed token history for the suffix
proposer (#1037's request-local suffix history must be rebuildable on the
decoder). The per-layer-range export/import primitives for all of this exist
in `skippy-runtime/src/kv_pages.rs` (including `_for_token_count` variants);
what's missing is streaming, transport, and commit semantics.

**MVP topology: full replicas, same backend.** One prefill node and one
decode node, each holding the whole model (single-stage), Metal↔Metal.
This sidesteps paired multi-stage pipelines and cross-backend portability.
But the identity guard does **not** come for free even then:
`PageIdentity` (`skippy-cache/src/identity.rs`) pins topology/split, so a
prefill replica and a decode replica will not match without splitting
identity into **numerical fields** (backend numerics, arch, cache types,
layer content — must match) and **placement fields** (topology, split,
node — allowed to differ). That split is the subtle correctness work and it
lands in Phase 1, not later.

## Byte math (why per-request policy, not a mode)

For the Nemotron 3 Super target: ~8 KiB attention-KV per prompt token →
256 MiB at 32K, plus ~160 MiB *fixed* recurrent/SSM state; roughly 350 ms at
10 Gb/s, 140 ms at 25 Gb/s line rate. Streaming KV pages behind later prompt
chunks hides most of the bulk; the recurrent snapshot + MTP context + commit
are the uncovered tail, because recurrent state is only final once prefill
finishes. The fixed 160 MiB floor also means short prompts are strictly
worse to disaggregate — break-even prompt length is a measurable function of
(state bytes, link throughput, prefill speedup ratio), and placement must be
cost-gated: reject short prompts and poor links.

## Phases

### Phase 0 — Falsifier benchmark (days, no product code)

Measure the handoff cost before building anything, per the accounting already
specified in `docs/skippy/EXPERIMENTS.md:323`:
`state_export_bytes/seconds`, `state_import_bytes/seconds`,
`kv_attach_seconds`, TTFT, TPOT.

- Use the existing offline harness (`skippy-correctness/src/runner/state_handoff.rs`)
  driving `export_full_state`/`import_full_state`
  (`skippy-runtime/src/kv_pages.rs`) between processes on the lab pair, with
  a plain file/socket copy over the Thunderbolt bridge standing in for the
  transport.
- Sweep prompt length {512, 2k, 8k, 32k} × cache type {f16, Q4_0} on
  Nemotron 3 Super (hybrid: measures the fixed recurrent floor) and one
  dense attention-only model (isolates the KV curve).
- Compare projected disaggregated TTFT (prefill-on-fast + transfer + attach)
  vs. measured prefill-in-place on the decode node — **both role
  assignments** (M3 prefill → M1 decode and the reverse), since the #1427
  perf matrix currently has no reverse arm.

**Gate:** disaggregation must project a TTFT win at some realistic prompt
length with transfer *not* overlapped (overlap only improves it). If flat
transfer never wins below 32k tokens, stop here and write up why.

### Phase 1 — KV page mobility in `skippy-cache` (the L3 substrate)

This phase *is* the radix L1/L2/L3 work — the gating dependency, owned by
that workstream; #1427 consumes its L3 stream contract (segment
identity/ordering, completeness, backpressure, idempotency) with disk and
network as interchangeable backends. Note #1399 deliberately **removed**
durable disk persistence from main (deleted `disk_tier.rs`, `miss_reason.rs`,
most of `exact_state.rs`), so this is a redesign informed by that removal,
not a small delta. Mic's in-channel acceptance criteria apply: solo-mode
proof across families, measurable 19K-prefix warmup reuse, concurrent
non-duplicating loads, capped write-behind disk budget.

- Page codec: serialize KV pages via `export_kv_page` into content-addressed
  blobs in the existing BLAKE3 `blob_store`, described by `RuntimeKvPageDesc`
  (layer range, token range, k/v types, row bytes, codec).
- **`PageIdentity` split into numerical vs placement fields** — numerics
  (backend numerics, arch, cache types) must match across the handoff;
  placement (topology, split, node) must not be pinned. The correctness-
  sensitive change; do it here where the harness can gate it.
- Chunked streaming export/import APIs (today `kv_pages.rs` is whole-buffer
  only, and `MAX_STAGE_STATE_IMPORT_BYTES` = 512 MiB caps a full-context
  import).
- L3 disk backend: spill/fill under the radix index, eviction ladder
  alongside the existing `SparseCheckpointPolicy`.

Standalone deliverable: prefix cache that survives process restart. Ship and
validate this before any networking.

### Phase 2 — Peer page fetch (`skippy-kv/1`)

- New iroh subprotocol (pattern: `STAGE_ALPN_V2` registration in
  `skippy-protocol/src/validation.rs` + the tunnel bridging in
  `mesh-llm-host-runtime/src/network/tunnel.rs`): request pages by digest,
  stream blobs, mesh-membership auth, bandwidth accounting.
- Extend the probe step: a `ProbePrefill` miss on local L1/L2/L3 may consult
  peer inventory. `CacheAffinityAdvertisement` (salted prefix digests,
  already gossiped — `mesh-llm-routing/src/cache_inventory.rs`) tells us
  which peer to ask without leaking tokens.

Standalone deliverable: cross-node prefix reuse — a prompt prefilled anywhere
in the mesh warms every node. This is the feature EXO doesn't have, and it is
also the entire decode-side machinery for Phase 3.

### Phase 3 — Disaggregated serving, single-request prototype

Single-request correctness first (#1427 is explicit); multi-request
production waits for the #1416 iteration-level scheduler cutover in Phase 4.

- Step 1: **full-prefill → full-decode** — prefill completes, the entire
  hybrid state snapshot transfers, decode attaches and continues. Behind the
  `state-handoff` correctness gates before any streaming.
- Step 2: **streaming handoff with two-phase commit** — KV pages push per
  completed prefill chunk over the backpressured QUIC stream, overlapping
  later chunks; recurrent snapshot + MTP context + commit record are the
  final segment; the decoder attaches nothing until the commit record
  validates completeness. Partial state can never generate.
- Config-pinned roles for the lab: `--phase-role prefill|decode|auto`
  (planner auto-placement deferred to Phase 4).
- First token samples on the prefill side (`prefill_final_frame_sampled`
  already exists); decode continues from token 2.
- Failure = fallback, not resume: prefill peer dies mid-handoff → decode
  node prefills locally from scratch; uncommitted segments are discarded.
- **Correctness gates** (the harness arms, in bisection order):
  1. dense attention-only model — isolates transport/commit bugs from
     state-family bugs;
  2. Nemotron 3 Super without MTP — adds recurrent/SSM + conv state;
  3. Nemotron 3 Super + MTPv2 + suffix N-gram — post-handoff output must
     deterministically match local continuation, which requires rebuilding
     request-local suffix history (#1037) on the decoder. Watch the #1385
     failure mode: MTP weights shipped but `speculative_decoding` omitted
     from the package manifest → MTP silently never selected (depends on
     the open #1425 Nemotron package).
- Family gate: add a `phase_disaggregation` capability to
  `reviewed-family-capabilities.json`, granted per family as it passes the
  harness ladder.

**Gate:** llama-benchy A/B on the lab — disaggregated vs decode-node-solo vs
2-stage layer split, both role assignments, reporting TTFT/TPOT and the
Phase 0 counters. Win condition: TTFT improves at long prompts with TPOT no
worse than solo decode.

### Phase 4 — Cost-based placement and multi-request production

- Feed the capability signals nodes already gossip but the planner ignores
  (`compute_tflops_fp16`, `mem_bandwidth_gbps` in `PeerAnnouncement`) into
  phase placement scoring in `skippy-topology/src/planning.rs` — there is
  no role concept in the planner today.
- Activate the peer-to-peer RTT/bandwidth matrix (`edge_order.rs` is
  currently dead code in production — known gap in
  `docs/skippy/TOPOLOGY_PLANNER.md`); handoff needs link throughput between
  the *pair*, not coordinator RTT.
- Cost-gated routing: break-even prompt length computed from measured link
  throughput, state bytes/token, and the fixed recurrent floor; reject
  short prompts and poor links. Decode-side admission accounts imported-
  state budget in the existing `MemoryComponent` capacity model.
- Multi-request serving via the #1416 iteration-level scheduler cutover.

### Phase 5 — Extensions (post-MVP, in pitch-value order)

1. **Quantized-KV transfer** (Q4_0 pages over the wire) — 4× fewer bytes
   exactly where the network is the constraint; nobody has shipped this.
2. **Split-prefill → collapsed decode** — multiple prefill workers each
   prefilling a slice, converging on one decoder (the #1427 generalization).
3. **Cross-backend canonical page layout** — CUDA/DGX-class prefill for a
   Mac decode fleet; requires an interchange codec and extending the
   numerical/placement identity split with an explicit portability mode.
4. **Disagg × pipeline composition** — prefill *pipeline* feeding a decode
   *pipeline* for models that fit on neither pool's single node, stage-i to
   stage-i page streaming.

## Non-goals

- Re-enabling wire `StateExport`/`StateImport` (rejected by the server as
  "not executable"; #1427 says do not re-enable — superseded by the
  dedicated backpressured QUIC handoff stream).
- In-flight sequence migration or resume after node failure.
- WAN disaggregation — byte math rules it out; LAN/Thunderbolt only.
- Recurrent-state mobility *during decode* (sticky-owner stands for layer-
  split serving; the one-time prefill→decode snapshot is in scope, per-token
  mobility is not).

## Open questions for review

1. Does the L3 page granularity match the restore ladder (`SparseCheckpointPolicy`
   checkpoints) or do we need a finer page size for streaming overlap?
2. Push vs pull for the Phase 3 hot path: push-on-chunk-complete is simplest;
   is pull-by-digest with prefetch hints worth the extra round trips to keep
   one code path with Phase 2?
3. Where does the disaggregation routing decision live — host ingress
   (`mesh-llm-host-runtime/src/api`) or driver? Ingress sees the prompt
   before tokenization; the driver knows scheduler state.
4. Where exactly does the numerical/placement line fall in `PageIdentity` —
   in particular `ctx_size`: operationally pin it equal across the replica
   pair, or classify it as placement and prove numerics are ctx-independent?
5. Role assignment prior: M1 Ultra (~800 GB/s) and M3 Ultra (~819 GB/s) are
   near-equal on bandwidth but ~2× apart on compute, which argues M3=prefill
   as #1427 pins it — but decode also carries MTP draft/verify compute, so
   the reverse arm in the Phase 0/3 matrices settles it by measurement.
