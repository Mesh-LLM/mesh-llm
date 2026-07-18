# Topology Planner

The topology planner separates model packaging from runtime placement. Small
layer slices are good cache/distribution units; runtime stages are execution
units that may group many cached slices together.

## Breakthrough

Falcon-H1/Qwen3Next-style recurrent models are not blocked from staged
distributed inference. The blocker is only exact recurrent-state transport.

The important distinction is:

- **move activation frames:** small, normal, Qwen-like;
- **do not move recurrent state during normal routing:** huge, owner-local,
  sequence-specific.

That means recurrent/hybrid models can still run as stage pipelines when the
topology preserves recurrent-state ownership. Future tokens for a sequence must
return to the node that owns the recurrent layer range, while downstream stages
receive only the small activation output.

## Long-Term Goal: Stage-Split Certification

The long-term goal is to certify that specific model families work very well
with the stage-split architecture, not merely that they can produce one matching
token in a smoke test. Certification should capture the knobs that matter for
that family:

| Certification axis | What we prove |
| --- | --- |
| Correctness | Staged execution matches full-model execution across representative prompts, split points, and decode lengths. |
| Stage transfer encoding | The family has a validated activation wire dtype policy, such as default `f16` and any allowed q8 cases. |
| Speculative decoding | Target/draft pairings and speculative strategy are known-good for the staged topology. This may land from separate branch work, but it belongs in the same certification record. |
| Topology | The planner knows valid split shapes, forbidden boundaries, sticky recurrent owners, sideband requirements, and node-placement constraints. |
| Performance | The certified knobs are not only correct; they are good operating points for throughput, latency, and VRAM/residency. |

The certification output should eventually say: this model family, with these
split points, transfer encodings, speculative settings, and topology rules, is
approved for stage-split deployment. Families that need special handling should
make that explicit rather than relying on tribal memory.

`just family-certify` is the first harness for collecting this evidence into a
dated artifact. It records correctness reports, dtype behavior, state-handoff
payload pressure, optional staged speculative corpus summaries, and a
`capability-draft.json` that can be reviewed before promoting new planner
policy; see `docs/FAMILY_CERTIFY.md`.

## Core Rule

For recurrent/hybrid models:

```text
activation crosses topology boundaries
recurrent state defines topology affinity
```

The planner should transfer the small output activation between stages. It
should not transfer recurrent state as part of normal routing.

## Planner Responsibilities

The planner maps:

```text
model layers -> cached layer slices -> execution stages -> node placement
```

For each stage it records:

- layer range;
- node owner;
- whether the range is stateless, attention-KV stateful, recurrent-stateful, or
  mixed;
- migration policy for active sequences;
- structured reason codes explaining sticky recurrent ownership, activation
  boundaries, shared-KV rejections, sideband needs, and wire dtype policy.

For each boundary it records:

- producer and consumer stage indexes;
- layer boundary;
- accepted/rejected decision;
- default activation wire dtype;
- raw activation bytes per token;
- actual wire payload bytes per token;
- reason codes and human-readable messages.

## Initial Policy

| Layer range | Normal activation routing | Active sequence migration |
| --- | --- | --- |
| Stateless/dense range | Allowed | Freely movable. |
| Attention-only range | Allowed | Costed KV migration. |
| Recurrent-only range | Allowed | Sticky owner unless explicitly recomputed or transferred. |
| Mixed attention/recurrent range | Allowed | Sticky owner unless explicitly recomputed or transferred. |

This lets Falcon-H1/Qwen3Next-style models participate in the same activation
pipeline as Qwen-style dense models while avoiding `~76-79 MB` recurrent-state
transfers during normal decode.

Activation wire dtype is also part of the plan. The conservative exact default
is `f16`: it halves Qwen3 dense activation payload and passed the Qwen3 dense
smoke. `q8` is not a global default. It is a per-family/per-split opt-in because
Qwen3 dense, Gemma3, GLM4, Gemma4 A4B, Gemma4 E4B, MiniMax M2.7, OLMo, and
Qwen3Next failed q8 exactness smokes while Llama, DeepSeek2, GLM-4.7 Flash,
Gemma2, and Falcon-H1 passed validated q8 smokes.

## Latency-Aware Stage Count

The planner should choose the number of physical stages from an end-to-end
decode latency budget. It should not maximize the stage count just because more
eligible peers are available.

Small layer slices remain useful package and cache units, but physical stages
are serialized execution hops during decode. Every additional remote stage adds
one more activation transfer on the critical path. Current upstream replies add
more serialized return hops; even with direct predicted-token return, the last
stage still pays one return hop to stage 0. For no speculative decoding, the
steady-state decode budget is:

```text
decode_ms_per_token =
  serialized_network_hops_ms + max(stage_decode_compute_ms) + protocol_overhead_ms
```

For a 30 tok/s target, the full decode loop must stay under `33.3 ms/token`.
With a working assumption of `10 ms` per inter-stage transfer, the optimistic
direct-return network floor already constrains feasible physical stage counts:

| Physical stages | Serialized hops per token | Network floor | Best possible before compute |
| --- | ---: | ---: | ---: |
| 2 | 2 | 20 ms/token | 50.0 tok/s |
| 3 | 3 | 30 ms/token | 33.3 tok/s |
| 4 | 4 | 40 ms/token | 25.0 tok/s |
| 5 | 5 | 50 ms/token | 20.0 tok/s |

Under that assumption, a four-stage topology cannot reach 30 tok/s even before
local decode compute is counted. A three-stage topology is only feasible if
compute and protocol overhead are almost zero. A two-stage topology leaves real
latency headroom, if the two peers can satisfy residency.

The old planning intuition was memory-first: use every feasible peer, split the
layers evenly, and accept the extra hops as the cost of fitting the model.

```mermaid
flowchart LR
    P["Planner"] --> C["Use all eligible peers<br/>balanced layer count"]
    C --> S0["stage 0<br/>host A"]
    S0 -->|"activation<br/>10 ms"| S1["stage 1<br/>host B"]
    S1 -->|"activation<br/>10 ms"| S2["stage 2<br/>host C"]
    S2 -->|"activation<br/>10 ms"| S3["stage 3<br/>host D"]
    S3 -->|"predicted token<br/>10 ms"| S0
    S0 --> O["OpenAI stream"]
```

The latency-aware planner should instead enumerate feasible physical stage
counts, reject candidates that do not fit resident memory, and choose the
lowest estimated time per output token among the survivors.

```mermaid
flowchart TB
    P["Latency-aware planner"] --> C2["2 physical stages<br/>2 serialized hops"]
    P --> C3["3 physical stages<br/>3 serialized hops"]
    P --> C4["4 physical stages<br/>4 serialized hops"]

    C2 --> M["Residency check"]
    C3 --> M
    C4 --> M

    M --> S["Score TPOT<br/>hops * RTT + max stage compute + overhead"]
    S --> Pick["Pick lowest TPOT<br/>that fits memory"]
    Pick --> R["Report target miss<br/>when no candidate can hit budget"]
```

This is a planner decision independent of the wire format: direct return
improves the constants, but it does not remove the monotonic cost of added
physical decode hops. The resource planner keeps one logical stage per physical
peer and uses measured or configured stage-transfer latency to score feasible
context/node/lane candidates. Later work can let one peer own multiple adjacent
logical stages, but the first planner decision is simpler: prefer fewer
physical decode hops whenever residency allows it.

## Family Capability Records

Family capability records let the planner reason from measured model-family
facts instead of hard-coding only dense Qwen assumptions. Reviewed records live
in `crates/skippy-topology/capabilities/reviewed-family-capabilities.json`
and are keyed by model coordinate/canonical artifact identity. Heuristic family
inference remains a fallback for unreviewed artifacts.

| Family | Planner facts |
| --- | --- |
| Llama | Dense activation-only staging, exact state mobility accepted in current smoke, default wire dtype `f16`, q8 wire validated for the reviewed artifact. |
| Qwen3 dense | Dense activation-only staging, exact state mobility accepted in current smoke, default wire dtype `f16`, q8 wire rejected for exactness. |
| DeepSeek2 | Dense/MLA activation staging, exact state mobility accepted in current smoke, default wire dtype `f16`, q8 wire validated for the reviewed artifact. |
| GLM-4.7 Flash | Plans as the DeepSeek2/MLA-style activation path, exact state mobility accepted in current smoke, default wire dtype `f16`, q8 wire validated for the reviewed artifact. |
| GLM4 | Dense activation-only staging, exact state mobility accepted in current smoke, default wire dtype `f16`, q8 wire rejected for exactness. |
| Gemma2 | Dense activation-only staging, exact state mobility accepted in current smoke, default wire dtype `f16`, q8 wire validated. |
| Gemma3 | Dense activation-only staging, exact state mobility accepted in current smoke, default wire dtype `f16`, q8 wire rejected for exactness. |
| Gemma4 A4B | Dense/MoE activation-only staging, exact state mobility accepted in current smoke, default wire dtype `f16`, q8 wire rejected for exactness. |
| Gemma4 E4B | Not recurrent, but requires token-id sideband for downstream auxiliary input reconstruction; q8 wire is rejected in the current certification split; known-bad shared-KV boundaries are rejected. |
| OLMo | Dense activation-only staging, exact state mobility accepted in current smoke, default wire dtype `f16`, q8 wire rejected for exactness. |
| MiniMax M2.7 | Dense activation-only staging in current GGUF runtime-slice smoke, exact state mobility accepted in current smoke, default wire dtype `f16`, q8 wire rejected for exactness; sharded GGUF stage materialization is supported. |
| Falcon-H1 | Every layer range owns recurrent state, exact state mobility rejected as too large, stage owners are sticky. |
| Qwen3Next | Recurrent ranges are supplied by the caller; ranges containing recurrent layers are sticky and exact state mobility is rejected as too large. |

The important behavioral distinction is:

| Planner output | Meaning |
| --- | --- |
| `activation_only_boundary` | Normal small activation handoff; this is the thing we want to move. |
| `recurrent_owner_sticky` | Future tokens for that live sequence must route back to this stage owner. |
| `shared_kv_region_cut` | Boundary is rejected because a measured shared-KV producer/consumer cut is unsafe. |
| `token_sideband_required` | Boundary may be accepted, but downstream execution needs token IDs in the activation frame. |
| `default_wire_dtype_f16` | Use f16 wire for exact staged execution unless explicitly overridden. |
| `q8_wire_validated` / `q8_wire_rejected` | q8 status is family/split evidence, not a universal rule. |

## First Crate

`crates/skippy-topology` is the first small implementation. It currently
provides deterministic contiguous planning over a set of nodes, supports
explicit split-boundary planning, classifies each stage by layer state
affinity, evaluates boundaries against optional family capability records, and
emits structured reason codes plus diagnostics.

The two entry points are:

| Function | Use |
| --- | --- |
| `plan_even_contiguous` | Produce a simple contiguous plan across the available nodes. |
| `plan_contiguous_with_splits` | Evaluate caller-provided split boundaries and return accepted/rejected boundary reasons. |

## Launcher Integration

The first runtime callers now use the planner as a preflight gate before
starting stage servers:

| Caller | Planner use |
| --- | --- |
| `skippy-bench run` | Validates distributed split plans before metrics/server launch. |
| `skippy-bench local-split-binary` | Validates the single local split before opening the model or starting stage 1. |
| `skippy-bench local-split-chain-binary` | Validates both local split boundaries before opening the model or starting stage servers. |
| `skippy-prompt prompt` | Validates local or remote prompt topology before slicing models or starting servers. |

Known preflight behavior:

| Case | Result |
| --- | --- |
| Qwen3 dense with `--activation-wire-dtype q8` | Rejected before launch because q8 failed exactness. |
| Gemma4 E4B with `--activation-wire-dtype q8` | Rejected before launch because the 2026-04-27 certification split changed the next token. |
| Gemma4 E4B split `14,28` or `12,24` | Rejected before launch because those boundaries are known-bad shared-KV cuts. |
| Gemma4 E4B split `21` | Accepted by planner; launch proceeds to normal model loading. |

The first implementation is intentionally simple. The important behavior is the
contract:

- recurrent state stays resident with the node that owns the recurrent layer
  range;
- future tokens for that sequence route back to the same owner;
- only activation frames cross stage boundaries in the normal pipeline;
- exact recurrent-state transfer is an explicit opt-in policy, not the default;
- exact activation wire defaults to `f16`;
- `q8` is opt-in only when the relevant family/split has passed correctness.

## Latency-Aware Placement: current behaviour and gaps

This section records what the planner actually does today for network-aware
placement (verified against `crates/skippy-coordinator/src/topology.rs`,
`crates/skippy-topology/src/{lib,edge_order}.rs`, and the host-runtime call site
`crates/mesh-llm-host-runtime/src/inference/skippy/topology.rs`), and the gaps
that matter as the mesh grows to many nodes.

### What works today

- **Latency is a cost, not just an exclusion.** `node_package_score` subtracts
  `rtt_ms × 16MiB` from a node's placement score, so higher-RTT nodes are
  deprioritised but still usable. A relay-only peer (no direct path) is the one
  case that is excluded outright.
- **The planner selects a node subset; it does not have to use every eligible
  node.** `plan_topology` enumerates `node_count in minimum_nodes..=usable_nodes`
  and, for each count, every node subset (`for_each_node_subset`). It can and
  does leave nodes out.
- **Stage count is gated on a decode-TPOT target.** When any node has a measured
  hop latency, planning is latency-aware: candidates are ranked
  target-met-first, then by lowest `estimated_decode_network_ms_per_token`
  (`latency_candidate_ordering` / `decode_tpot_target_met`, default target
  `DEFAULT_TARGET_DECODE_TPOT_MS = 33`). A shallower split that meets the target
  is preferred over a deeper split that does not — deeper is not chosen just
  because it fits.

### Gaps for the many-node / co-located future

1. **No peer-to-peer RTT matrix in production.** `edge_order.rs` implements
   adjacent-stage RTT ordering (exhaustive for ≤8 stages, greedy beyond) driven
   by `StageEdgeSignal { source, target, rtt_ms }`. **The host runtime never
   supplies edge signals** — it calls
   `plan_package_aware_contiguous_with_signals` (no transport variant), so the
   edge-matrix machinery is effectively dead in production. Placement sees only
   each node's RTT *to the coordinator*, not node↔node RTT.
2. **The network estimate is a worst-case proxy, not a path sum.**
   `estimate_decode_network_ms_per_token = max(per-node coordinator RTT) ×
   node_count`. It assumes every hop costs the slowest node's coordinator-RTT.
   This is why **co-located nodes cannot be exploited**: two nodes close to each
   other but each ~25 ms from the coordinator are modelled as ~25 ms hops, even
   if their real A↔B hop is ~1 ms. Combined with vast.ai's NAT hairpin (two
   containers behind one host cannot hole-punch and fall back to a ~200 ms
   relay), co-located pipeline stages are currently not achievable there.
3. **`minimum_nodes` forces splitting in labs, but there is no first-class
   "prefer fewer/none" cost knob beyond the TPOT gate.** For single-fit models
   the TPOT gate already discourages needless splitting; for forced-split lab
   runs `minimum_nodes` overrides it. Fine today, but a future many-node mesh
   wants an explicit placement policy (e.g. "only add a stage if it improves
   TPOT by X" and "never place on nodes above Y ms").

### Implications

- Adding a real node↔node RTT matrix (wire `edge_signals` from measured
  peer-to-peer RTT into `plan_package_aware_contiguous_with_transport`) is the
  single change that would unlock co-location and correct many-node ordering.
- Until then, treat placement as **coordinator-RTT-aware only**: it will
  deprioritise distant nodes and pick a sensible subset/stage-count, but it
  cannot reason about which nodes are close to *each other*.
