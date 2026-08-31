# mesh-llm-routing

`mesh-llm-routing` owns the shared routing primitives used by the host binary
and the embedded Rust client.

It intentionally stays small:

- `InferenceTarget` describes where a model request should go.
- `ModelTargets` stores per-model candidate targets and performs round-robin or
  sticky candidate selection.
- `cache_inventory` stores bounded, short-lived positive L1 receipts and emits
  rotating salted digests; raw prompt content and tokens never cross gossip.
- `cache_aware` compares a verified hit's queue, restore, and suffix-prefill
  cost with cold prefill before it may override normal routing.
- `affinity` extracts request namespaces and preserves explicit session/sticky
  fallback without learning a long-lived `prefix -> target` mapping.
- `total_model_bytes()` calculates GGUF model size, including split GGUF
  shard sets.

Higher-level request parsing, OpenAI transport behavior, peer observation, and
runtime orchestration stay in the owning application crates. This crate is the
common vocabulary those layers use when they exchange routing decisions.

The current wire contract advertises L1 evidence only. Lower storage tiers and
physical KV pinning are separate Skippy data-plane concerns.

## Cache-evidence trust boundary

Cache advertisements are routing hints from admitted peers, not cryptographic
proofs of residency. A malicious or compromised admitted peer can advertise a
forged hit, token depth, queue delay, or restore cost. The public rotating salt
limits how long an observed digest can be linked; it is not a secret and does
not authenticate the evidence. Routing therefore accepts only bounded,
short-lived, monotonically newer advertisements and falls back to normal
health/context ordering when evidence is absent, stale, malformed, or not
individually cheaper than cold prefill. Raw prompt content and token IDs are
never placed on the gossip wire.

The current L1 producer reports suffix-prefill work and queue delay. Restore
cost remains zero for L1 because no restore operation occurs; non-zero restore
cost and lower-tier evidence are reserved scaffolding for future data-plane
work and must not be treated as independently verified measurements.
