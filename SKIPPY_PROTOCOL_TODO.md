# Skippy Protocol RTT TODO

This list tracks protocol and serving-path work to reduce RTT cost in staged
Skippy serving. The current chain sends activations downstream stage by stage
and waits for ACK or predicted-token replies to walk back upstream. That is
correct, but it exposes one or more network round trips for every prefill
chunk and every normal decode token.

> **Reconciled with code 2026-06-24.** Several items below shipped since this
> doc was last revised (2026-05-13) and have been ticked with a `DONE:` note
> pointing at the landing code. Speculative decode is no longer "defaults off":
> there is a full `[defaults.speculative]` config surface (`auto / disabled /
> draft / ngram`), native MTP is implemented in the runtime and defaults on
> (auto-resolved from layer-package MTP generation metadata, see #888), GLM DSA
> MTP runs as a split-stage sidecar (#899), and direct (reverse-relay-free)
> predicted-token / span returns exist (`binary_transport/direct_return.rs`).
> Checkboxes that remain `[ ]` are still open or unverified.

## Current Observations

- Prefill and decode use the same neighbor-chain shape:
  `stage0 -> stage1 -> ... -> final`, with replies returning hop by hop.
- Non-final prefill messages can be early-ACKed in the binary stage handler,
  but embedded stage0 OpenAI prefill still writes one chunk and waits for the
  downstream ACK before computing or forwarding the next chunk.
- Mesh-launched remote stages currently disable async prefill forwarding even
  though the binary stage path has an async forwarder and bounded reply-credit
  support.
- Normal decode is one `DecodeEmbd` request per token and waits for one
  `PredictedToken` reply from the final stage chain.
- `VerifySpan` already supports speculative decode windows and batched
  `PredictedTokens`. As of 2026-06-24 mesh embedded **does** wire speculative
  decode: `[defaults.speculative]` config (`auto / disabled / draft / ngram`,
  `draft_model_path` / `draft_hf_repo` / `draft_max_tokens` / etc.) and native
  MTP defaulting on via package generation metadata.
- `TryRestorePrefillDecode` already fuses exact-prefix restore with the first
  decode step for a narrow warm-cache path.
- Activation transport already supports `f32`, `f16`, and `q8`; decode is
  usually RTT-bound, while prefill is more sensitive to activation bytes.

## A. RTT Reduction

- [ ] Pipeline embedded stage0 prefill.
  - Replace the chunk-by-chunk write-and-wait loop in embedded stage0 with
    bounded deferred ACK handling similar to the binary stage handler.
  - Preserve cancellation behavior and error propagation when a deferred
    downstream ACK fails.
  - Emit telemetry for prefill credit limit, pending replies, credit waits, and
    deferred replies drained.

- [x] Enable async prefill forwarding for mesh-launched remote stages.
  - DONE: `async_prefill_forward` exists and defaults on
    (`skippy-server/src/binary_transport/options.rs`:
    `async_prefill_forward: args.async_prefill_forward || !args.no_async_prefill_forward`).
    Revisit credit-limit-from-mesh-policy + flush-on-control if not yet covered.
  - Start with conservative bounded credit.
  - Make the credit limit configurable from mesh stage load policy.
  - Validate that Stop, generation config, checkpoint, restore, trim, and prefix
    cache control all flush pending forwards before continuing.

- [x] Add direct final-stage predicted-token replies.
  - DONE: `skippy-server/src/binary_transport/direct_return.rs` — a direct
    prediction-return listener (`PredictionReturnKey`, return address bound by
    stage0; final stage sends `send_reply_predicted_tokens_with_stats` /
    `send_reply_predicted_with_tokens_and_stats` directly back, bypassing the
    reverse hop relay).
  - Let stage0 provide a direct reply lane or return address for predicted-token
    replies.
  - Keep intermediate stages on the forward activation path but remove reverse
    token relay where possible.
  - Define how final-stage errors and per-stage stats are returned or
    summarized when reverse hops are bypassed.

- [x] Add final-stage direct commit for accepted decode spans.
  - DONE (shares the direct-return path above): accepted-span / predicted-token
    results return directly to stage0 via `direct_return.rs`, avoiding per-hop
    reverse relay. Verify batched `VerifySpan` windows exercise this lane under
    benchmark before closing fully.
  - For speculative verification, let the final stage return a span result
    directly to stage0.
  - Avoid per-hop reverse relay for accepted windows.

## B. RTT Amortization

- [x] Expose speculative decode controls through mesh skippy config.
  - DONE: `[defaults.speculative]` schema in `mesh-llm-config` (strategy
    `auto/disabled/draft/ngram`, `draft_model_path`, `draft_hf_repo`,
    `draft_hf_file`, `draft_selection_policy`, `pairing_fault`,
    `draft_max_tokens`, `draft_min_tokens`, `draft_acceptance_threshold`,
    `draft_split_probability`, `spec_default`); resolver in
    `mesh-llm-host-runtime/.../skippy/resolver/speculative.rs`; native MTP
    defaults on (`native_mtp_enabled.unwrap_or(true)`).
  - DONE (status): `api/status.rs` exposes `speculative: Option<bool>` per slot.
  - OPEN: benchmark acceptance rate + latency on LAN and higher-RTT links.

- [ ] Tune prefill chunk policy for RTT.
  - Replace fixed `64` token embedded prefill chunks with an adaptive or
    scheduled policy by default.
  - Prefer smaller early chunks for first-token latency and larger later chunks
    when downstream wait is exposed.
  - Record cache-hit granularity impact when changing chunk size.

- [ ] Add decode frames or windowed decode for non-draft speculation.
  - Explore a protocol message that carries a bounded decode window and returns
    a token span.
  - Define rollback rules for sampling divergence.
  - Keep this separate from draft-model speculation unless the state model can
    be shared cleanly.

- [ ] Extend `TryRestorePrefillDecode`.
  - Support more warm-prefix cases, including sampling metadata where safe.
  - Keep fallback behavior explicit when any downstream stage misses the prefix
    cache.

## C. Transport Efficiency

- [ ] Benchmark `q8` activation transport against `f16` for prefill.
  - Measure wall time, encode/decode overhead, and correctness by model family.
  - Decide whether `q8` should become a high-RTT policy default.

- [ ] Prototype fp8 activation wire dtype.
  - Add as an explicit versioned dtype, not as a reinterpretation of `q8`.
  - Gate by family certification and correctness tests.

- [ ] Reduce activation copies on forwarding.
  - Audit encode/decode and `Vec<u8>` cloning in the forwarding path.
  - Prefer borrowing or reusable buffers where the runtime and codec APIs allow.

- [ ] Evaluate persistent QUIC streams for activation lanes.
  - Keep per-session ordering semantics clear.
  - Compare against the current persistent TCP lane pool in embedded stage0.

- [ ] Treat compression as prefill-only unless measurements prove otherwise.
  - Decode payloads are small and RTT-bound.
  - Prefill payloads can be large enough that bandwidth reduction may help.

## D. Pipeline Utilization

- [ ] Add bounded prefill lead across stages.
  - Allow stage0 to be ahead by N chunks while downstream stages drain.
  - Bound by lane count, memory pressure, and outstanding ACK count.

- [ ] Add wavefront scheduling for multi-stage prefill.
  - Keep all stages busy by allowing chunk `k+1` on stage0 while chunk `k`
    advances through later stages.
  - Report per-stage idle time and downstream wait in telemetry.

- [ ] Separate async writer backpressure from runtime compute.
  - Ensure a slow downstream write cannot hold the runtime lock or block
    unrelated lanes longer than necessary.

- [ ] Benchmark stage split choices using RTT-aware cost.
  - Prefer splits that reduce the slowest exposed downstream wait, not just
    balanced layer counts.

## E. State Lifecycle Efficiency

- [x] Make speculative positions authoritative in stage-state v9.
  - Decode and `VerifyWindow` messages carry the absolute position each stage
    must have before execution.
  - Stages ahead of that position rewind attention KV locally.
  - Speculation never checkpoints, restores, trims by control message, or
    replays a rejected prefix.
  - Recurrent-state stages reject positional speculation instead of falling
    back to the removed checkpoint protocol.

- [ ] Add checkpoint hierarchy.
  - Keep coarse prompt checkpoints for non-speculative session/cache workflows.
  - Evict checkpoint state with explicit memory accounting.

- [ ] Track state-control latency separately from activation latency.
  - Emit checkpoint, restore, trim, prefix-restore, and decode-fuse timings per
    stage.
  - Keep speculative position-rewind counts and trimmed-token totals separate.

## Validation Gates

- [ ] Run protocol tests: `cargo test -p skippy-protocol --lib`.
- [ ] Run server tests: `cargo test -p skippy-server --lib`.
- [ ] For protocol or staged-serving changes, run mesh skippy tests:
  `cargo test -p mesh-llm inference::skippy --lib`.
- [ ] For gossip, routing, API serialization, or topology-visible changes, run:
  `cargo test -p mesh-llm --lib`.
- [ ] Benchmark at least one LAN multi-stage topology before promoting any
  default change.
- [ ] For mixed-version mesh implications, keep mesh protocol changes additive
  and fail closed for incompatible skippy stage protocol versions.
