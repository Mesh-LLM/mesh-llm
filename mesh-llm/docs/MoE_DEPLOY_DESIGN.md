# MoE Auto-Deploy — Implementation Notes

This documents how MoE expert sharding is implemented in mesh-llm. Originally a design doc; updated to reflect the actual implementation.

## User Experience

```bash
# MoE auto-detected — splits if needed, runs solo if it fits
mesh-llm serve --model Qwen3-30B-A3B-Q4_K_M

# Force splitting even if model fits locally
mesh-llm serve --model Qwen3-30B-A3B-Q4_K_M --split
```

The system detects MoE from the GGUF header, computes expert assignments, splits the GGUF per node, and each node runs its own llama-server. Sessions are hash-routed. No manual steps.

## Implementation

### Step 1: Detect MoE (`moe.rs`)

`detect_moe(path)` reads the GGUF header — looks for `*.expert_count` and `*.expert_used_count` KV pairs. Returns `GgufMoeInfo` or None. Takes ~1ms.

### Step 2: Decide solo vs split (`election.rs`)

In `election_loop()`, after detecting MoE via `lookup_moe_config()`:

```
if model.is_moe && (force_split || !model_fits_locally) && node_count >= 2:
    → moe_election_loop() — each node gets a different shard GGUF
else:
    → normal election — solo or tensor split
```

`lookup_moe_config()` checks two tiers:
1. **Catalog** — pre-computed rankings (instant, optimal)
2. **GGUF header** — auto-detected, uses cached ranking if available, otherwise 50% shared core fallback

### Step 3: Compute assignments (`moe.rs`)

`compute_assignments(ranking, n_nodes, min_experts)`:
- Shared core = top `min_experts` by gate mass (replicated to every node)
- Remaining experts distributed round-robin across nodes
- Returns `Vec<NodeAssignment>` — each has `experts`, `n_shared`, `n_unique`

### Step 4: Split GGUF (`moe.rs` → `llama-moe-split`)

`run_split()` calls the external `llama-moe-split` tool with `--expert-list`. Produces a self-contained GGUF: full trunk + selected experts + adjusted router gates + updated metadata.

Splits are cached at `~/.cache/mesh-llm/splits/<model>/<n>-nodes/node-<i>.gguf`. Invalidated implicitly when node count changes (different directory).

### Step 5: Independent llama-servers

Each node runs `llama-server` with its split GGUF. No `--rpc`, no tensor splitting. Each node is fully independent with its own KV cache.

`moe_election_loop()` manages the lifecycle: start, restart on mesh changes, kill on shutdown.

### Step 6: Session routing (`proxy.rs` + `election.rs`)

`extract_session_hint()` parses `user` or `session_id` from the request body. `get_moe_target()` hashes it to pick a node. `MoeLocal`/`MoeRemote` targets handle local vs QUIC-tunneled forwarding.

## What's NOT implemented from the original design

- **Shard distribution over QUIC** — the design proposed pushing shards from host to workers. Instead, every node splits locally from its own copy of the full GGUF. Simpler, but requires every node to have the full model on disk.
- **Probe-based placement** — hash routing is used instead. Both nodes are equivalent with sufficient overlap.
- **Lazy `moe-analyze`** — not run automatically. Users can run it manually; cached rankings are picked up.

## Open Questions (from original design, still open)

1. **Node count changes**: Re-splitting when a 3rd node joins a 2-node mesh. Currently handled — `moe_election_loop` detects the change, re-computes assignments, re-splits if the new split doesn't exist in cache.
2. **Minimum viable calibration per model**: The 50% default is conservative. Different models may need more or less. Only Qwen3-30B-A3B has been properly calibrated (36% = 46/128 experts).
3. **Can we skip `moe-analyze`?** The 50% fallback works but wastes storage. Gate norms from GGUF weights (no inference needed) might give a cheap approximation of expert importance.

## Future Direction: Descriptor-Carried MoE Topology

As mesh-llm moves toward protocol-level `ServedModelDescriptor` objects, MoE should stop depending only on local GGUF inspection and instead consume `ModelTopology.moe` when it is available.

Planned source priority:

1. precomputed MoE data in this repo
2. Hugging Face metadata for exact `repository + revision + artifact`
3. GGUF header fallback
4. later, cached `moe-analyze` output for that exact descriptor identity

This gives us two important properties:

- **revision-aware grouping**: nodes can confirm they are serving the same exact MoE snapshot before coordinating
- **clean future analysis flow**: `moe-analyze` can improve topology later without changing the contract for how topology is identified

Longer term, we may also use lighter-weight local signals such as short warm-up inference or recent router statistics to improve unknown models. That data should remain explicitly lower-confidence than `moe-analyze`, and should be treated as a hinting/calibration layer rather than the canonical topology source.
