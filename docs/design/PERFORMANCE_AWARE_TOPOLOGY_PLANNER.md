# Performance-Aware Topology Planner and Placement Simulator

## Status: Phases 0-3 implemented; 4-5 planned

- Date: 2026-08-26
- Owner: TBD
- Origin: skippy-topology channel discussion (2026-08-26); requested by James.
- Implementation: PR #1454 (branch `docs/perf-aware-topology-planner`).
  Phases 0-3 (metric plumbing, perf-aware span assignment, directed-edge
  network model, modeled-TPOT candidate selection, placement simulator +
  scenario corpus, execution sim calibration, passive link measurement,
  live per-stage timing feedback, and RTT-floor confidence) are implemented
  and tested there. As-built notes are inline below. Phases 4-5 (reference-
  hardware A/B and adaptive replanning) remain planned.

## Problem

Automatic split placement is decided by a capacity fitter that ignores
performance. The planner in `crates/skippy-coordinator/src/topology.rs`
distributes contiguous layer ranges using memory arithmetic only —
per-layer weights, KV bytes/token, recurrent state, a fixed compute reserve —
and its only network term is `stage_count × max(coordinator RTT)`
(`estimate_decode_network_ms_per_token`). Two consequences:

1. **Node performance is invisible.** A node with half the sustained memory
   bandwidth receives the same layer span as an equal-VRAM node and becomes
   the pipeline straggler. Quantized decode is usually memory-bandwidth-bound,
   so this is the dominant term, not FLOPs.
2. **Link performance is invisible.** Activation wire time is ignored
   entirely; latency enters only as a scalar hop count times the worst
   coordinator-measured RTT. Directed per-hop differences (e.g. an
   asymmetric Wi-Fi hop, a cross-continent edge) do not influence stage
   ordering or span sizes.

Meanwhile the fleet already measures the missing signals: peers gossip
`gpu_mem_bandwidth_gbps` and `gpu_compute_tflops_fp16/fp32`
(`crates/mesh-llm-host-runtime/src/mesh/gossip.rs`), and the policy crate has
a directed edge model (`StageEdgeSignal { rtt_ms, large_frame_bytes_per_sec }`
in `crates/skippy-topology/src/edge_order.rs`) that automatic placement never
consults. The data exists; the planner does not receive it.

Measured cost of getting this wrong (`docs/BENCHMARKS.md`): GLM-4.7-Flash
Q4_K_M on M4 Max + Mac mini over Wi-Fi runs 68 tok/s solo, 21 tok/s at a
2-way split, 12-13 tok/s at 3-way. Splitting always costs; splitting without
performance awareness costs more than it must.

## Goal

One production planner that takes a complete input set — memory, node
performance, directed link performance, model-side legality, workload
intent, and operational stability — and chooses the topology that best meets
the stated objective. Plus a simulator that evaluates planner decisions
under synthetic conditions without a cluster, so the cost model is testable,
calibratable, and regression-guarded.

## Non-goals

- No change to stage execution semantics, the activation wire protocol, or
  llama.cpp integration. Runtime work is limited to passive timing
  instrumentation; this work still re-plans *placement*.
- No speculative/exotic parallelism (tensor/pipeline-parallel hybrid graphs).
  Contiguous layer pipelines only, as today.
- No live adaptive replanning in the first phases (see rollout — hysteresis
  and migration come last, after the model is calibrated).

## Current state

| Capability | Where | Used by automatic placement? |
|---|---|---|
| Capacity fitting (exact per-layer weights, KV/token, recurrent/lane, 100/85 KV compute reserve, 10% runtime headroom) | `skippy-coordinator/src/topology.rs` | Yes |
| Candidate search (context ↓, node count ↑, lanes ↓, all subsets), stage-0 binding, 33 ms decode TPOT target, 64K shared-context floor | `skippy-coordinator/src/topology.rs`, `mesh-llm-host-runtime/src/runtime/split_planning.rs` | Yes |
| Latency estimate `stage_count × max RTT` | `estimate_decode_network_ms_per_token` | Superseded when edge data is present (modeled per-hop estimate); legacy estimate otherwise |
| GPU benchmarking (mem bw, fp16/fp32 TFLOPS) | `mesh-llm-gpu-bench`, `mesh-llm-system/src/benchmark.rs` | Metrics gossiped; **flow into the planner as of PR #1454** (auto-runs at node startup on non-client nodes) |
| Directed edge signals (RTT + large-frame bandwidth per edge, prediction-return support) | `skippy-topology/src/edge_order.rs` (exhaustive ordering ≤ 8 stages, greedy beyond) | The automatic planner consumes `TopologyEdge` RTT/bandwidth for scoring as of PR #1454. Production currently synthesizes symmetric pair estimates from coordinator-to-peer observations; directed node ordering and prediction-return capability remain confined to the explicit `skippy-topology` planner |
| Perf-aware span assignment (DP over layer boundaries minimizing max modeled stage time) | `skippy-coordinator/src/topology.rs` (`perf_balanced_spans`) | Yes, when every node in a subset reports sustained bandwidth; exact legacy greedy otherwise |
| Modeled single-stream decode TPOT (Σ stages + Σ hops) for candidate selection | `skippy-coordinator/src/topology.rs` (`modeled_decode_tpot_us`) | Yes, when both compared candidates carry complete bandwidth signals; legacy ordering otherwise |
| Observed steady-decode timing (µs/layer, sample count, age) | `skippy-server/src/stage_performance.rs`, additive gossip fields in `AdvertisedModelThroughput` | Yes; a fresh observation is a measured floor on the analytical stage estimate in both span DP and serial TPOT scoring |
| RTT-floor confidence (sample count + first/latest sample age) | `mesh/peer_state.rs`, `runtime/local_package.rs` | Yes; remote perf signals are withheld until the minimum RTT is corroborated across the 5-second settle window |
| Placement simulator + scenario corpus | `skippy-topology-sim` crate | CI surface for planner behavior; corpus in `crates/skippy-topology-sim/scenarios/` |
| Model-family cut rules, state affinity, shared-KV cut bans, wire dtype, sidebands | `skippy-topology/src/planning.rs`, `validation.rs` | **No** (explicit-split validation only) — folding legality inputs into automatic planning is future work |

The table above reflects the tree as of PR #1454 head; the original
`9feef0c1` survey that motivated the design is preserved in the PR's
first commit.

## Input contract

### 1. Node performance (per node)

| Field | Source | Notes |
|---|---|---|
| `usable_vram_bytes` | existing `TopologyNode` | after 10% runtime headroom |
| `sustained_mem_bw_gbps` | gossip (`gpu_mem_bandwidth_gbps`), gpu-bench | measured, not spec |
| `sustained_compute_tflops` | gossip (`gpu_compute_tflops_fp16` preferred) | fallback signal only; decode is usually memory-bound |
| `host_ram_bytes` | node status | workspace/scratch headroom |
| `observed_decode_us_per_layer`, `sample_count`, `sample_age_ms` | staged runtime observation, additive gossip hint | steady decode after warmup; observations older than 30 minutes are omitted |
| `load_ewma`, `metric_age_ms` | future probe | stale load signals decay to neutral |

### 2. Directed link performance (per ordered node pair)

| Field | Source |
|---|---|
| `min_rtt_ms` | existing direct peer RTT floor |
| `large_frame_bytes_per_sec` | existing `StageEdgeSignal` field |
| `sample_count`, `first_sample_age_ms`, `last_sample_age_ms` | RTT observation window behind the minimum |
| `direct_prediction_return_supported` | existing `StageEdgeSignal` field; not yet part of the automatic planner contract |

The explicit `skippy-topology` planner prices unknown edges with its pessimistic
`UNKNOWN_EDGE_RTT_MS` default. The automatic coordinator planner instead tries
the directed edge, then its reverse, then an endpoint's coordinator RTT; if no
RTT signal exists anywhere for a required hop, modeled TPOT is unavailable.
The planner deliberately does not estimate distribution tails or variance here:
iroh owns path selection, while placement only needs to distinguish a one-off
early minimum from a floor corroborated after the direct-path settle recheck.

`direct_prediction_return_supported` is intentionally not consumed by
automatic placement in this PR. The stage runtime continues to enforce its
existing direct-return/fallback contract. Folding this capability into the
automatic planner requires an explicit mode: either reject an unsupported
return edge when direct return is mandatory, or price the downstream fallback
path when it is allowed.

### 3. Model-side legality (per family)

Existing policy inputs from `skippy-topology`: legal cut points, state
affinity, forbidden shared-KV cuts, activation wire dtype and bytes/frame,
required sidebands, backend/kernel support. Hard filters — a plan that
violates them is invalid regardless of score.

### 4. Workload intent

Objective selector: `interactive` (TTFT + decode TPOT targets; today's 33 ms
target generalizes) vs `throughput` (aggregate tok/s at concurrency).
Includes prompt/decode mix, context distribution, and lane count.

### 5. Operational stability

Artifact/package locality (existing eligibility filter), cold-load time,
node reliability, KV/state migration cost, and a minimum-improvement
threshold + hysteresis window so topologies do not churn.

## Cost model

Per stage `i` with layer span `L_i` on node `n(i)` and egress edge `e(i)`:

```text
stage_time_ms(L_i) = max( Σ_{l∈L_i} weight_bytes(l) / mem_bw(n(i)),          # weight streaming
                         flops(L_i) / sustained_compute(n(i)),                # compute-bound regimes
                         observed_us_per_layer(n(i)) × |L_i| )                # live measured floor
                   + kv_touch_ms(L_i)                                        # resident KV scan per token
edge_time_ms(i)    = act_bytes(L_i) / large_frame_bw(e(i)) + min_rtt(e(i))
single_stream_tpot_ms = Σ_i ( stage_time_ms(i) + edge_time_ms(i) )           # measured decode regime
pipelined_period_ms   = max_i ( stage_time_ms(i) + edge_time_ms(i) )         # lanes > 1
prefill_ms         = Σ_i ( stage_time_ms(i) + edge_time_ms(i) )              # sequential fill
```

The BENCHMARKS.md anchors established that today's single-stream decode is
serial through the complete pipeline, while multiple in-flight lanes can
approach the pipelined period. The planner scores the measured single-stream
regime against the existing 33 ms target.

**As-built (PR #1454):** decode is modeled as weight-streaming
(`streamed weight bytes / sustained_mem_bw`, integer microseconds, scaled
by `active_weight_fraction_permil` for MoE models) **plus both calibrated
overhead terms** — `per_stage_overhead` (1.3 ms) and `per_hop_overhead`
(13 ms) — inherited from the execution sim's BENCHMARKS.md calibration
(`CALIBRATED_PER_STAGE_OVERHEAD_US` / `CALIBRATED_PER_HOP_OVERHEAD_US` in
`skippy-coordinator`). The coordinator's modeled decode TPOT is the
**serial form**: Σ stage service times + Σ hop times across all hops
including the prediction return — every token traverses every stage, the
regime the BENCHMARKS.md anchors prove for single-stream decode. The
planner's number is locked to the calibrated execution sim by the
`planner_model_matches_execution_sim` test (≤1% divergence on the anchor
scenario). Live steady-decode observations are normalized to µs/layer after
warmup, bounded to an initial 256-sample arithmetic mean and then a 1/8 EWMA,
gossiped with count and age, and applied as a measured floor in both the span
DP and serial TPOT score. The compute term and KV-touch term from the formula
above remain plumbed-but-unused pending calibration against broader measured
data.
Missing node bandwidth is **all-or-nothing per candidate**:
a subset missing any node's bandwidth keeps the exact capacity-greedy span
assignment and carries `modeled_decode_tpot_us = None` — note the network
estimate still uses edge-aware per-hop estimation whenever edge data
exists, and only falls back to the legacy `hop_count × max-RTT` estimate
when edges are empty or disabled. A missing edge bandwidth contributes
zero transfer time (latency-only hop); an unmatched hop falls back to node
RTT, and a hop with no RTT signal anywhere declines to model TPOT
(`None`) rather than treating the hop as free. Canonical units: sustained
bandwidth MiB/s (1 MiB = 1_048_576 bytes), edge bandwidth MiB/s, all
modeled times integer microseconds; conversions happen once at parse
(GB/s → MiB/s, TFLOP/s → GFLOP/s). Stage observations older than 30 minutes
are omitted. For remote candidates, the RTT/edge signal and all node-
performance signals are withheld until at least two valid RTT observations
span 5 seconds and the latest is no older than 30 seconds; this reuses the
capacity-only fallback instead of trusting an early post-connect minimum.

**Scope of the fallback guarantee:** the all-or-nothing signal check and
the fallback span assignment are **per candidate subset**, not fleet-wide.
In a mixed fleet (some nodes reporting bandwidth, some not), fully-signaled
subsets get perf-balanced spans while subsets containing a signal-less node
keep the capacity-greedy walk — so which subsets win candidate selection
can differ from a signal-less fleet. Additionally, any non-empty edge data
switches the network estimate to edge-aware per-hop accounting for *every*
candidate, including capacity-greedy plans from signal-less subsets. The
bit-identical guarantee holds only when the fleet reports **no node
bandwidth signals and no edge data at all**; it is exercised by
`missing_perf_signals_keep_capacity_only_placement`,
`signalless_subset_placement_unchanged_by_other_nodes_signals`, and
`edge_data_changes_capacity_greedy_candidate_ordering`.

## Search algorithm

Preserve the existing candidate enumeration (it is correct and tested); add
performance to scoring and ordering. This is the target algorithm; the
as-built exceptions immediately below distinguish the implementation landed in
PR #1454:

1. **Hard feasibility filter** (unchanged): memory fit with reserves,
   family cut legality, stage-0 binding, sidebands, artifact access.
2. **Enumerate candidates** (unchanged): context lengths highest→lowest,
   node counts fewest→most, lanes highest→lowest, node subsets.
3. **Order nodes** on the directed link graph (adopt
   `order_pipeline_nodes`: exhaustive ≤ 8 stages, greedy beyond) instead of
   VRAM-descending order.
4. **Span assignment**: replace greedy largest-fit with DP over contiguous
   layer boundaries that minimizes the maximum modeled stage service time
   subject to per-node
   memory ceilings. The recurrence compares every prior boundary, so a
   candidate costs `O(layers² × nodes)` — at current scales (≤ ~100 layers,
   ≤ ~8 nodes) that is ≤ ~80K comparisons per candidate, trivially cheap;
   Knuth-style optimization could reduce it to `O(layers × nodes)` if
   fleets grow.
5. **Score lexicographically**: correctness → SLO met → objective-specific
   performance (TPOT or throughput) → context/lane utility → confidence and
   headroom → deterministic tie-breaks (existing `latency_candidate_ordering`
   shape).

**As-built exceptions (PR #1454):** automatic stages remain ordered by usable
VRAM descending with a node-id tie-break; the coordinator does not yet call
`order_pipeline_nodes`. Automatic planning also does not yet consume the
model-family legality/sideband policy or prediction-return support from
`skippy-topology` (see the current-state table). The span DP balances modeled
stage service time in that fixed order, while directed edge data affects
candidate scoring only. Edge-aware node ordering and policy integration remain
explicit follow-up work.

## Simulator

Two layers, sharing one scenario format (`toml`) — see the as-built corpus in
`crates/skippy-topology-sim/scenarios/`:

```toml
[nodes.m4max]
vram_bytes = 68719476736              # 64 GiB
sustained_mem_bandwidth_mib_per_s = 546000   # measured
sustained_compute_gflop_per_s = 34000
observed_decode_us_per_layer = 2400           # optional live measured floor

[nodes.mini]
vram_bytes = 17179869184              # 16 GiB
sustained_mem_bandwidth_mib_per_s = 120000
sustained_compute_gflop_per_s = 2000

[links."m4max -> mini"]               # directed edge, spaces in key
rtt_ms = 3
large_frame_mib_per_s = 30            # Wi-Fi large-frame prior

[model]
layer_count = 40
weight_bytes_per_layer = 1610612736
kv_bytes_per_token = 4096
native_context_length = 65536
activation_frame_bytes = 8192

[workload]
minimum_nodes = 2
```

1. **Placement sim** (deterministic, fast, in-crate): scenario → planner →
   assert chosen topology and score. Runs in CI as the planner's unit-test
   surface: property tests over synthetic grids ("10× slower link must move
   the boundary", "half-bandwidth node must receive fewer layers",
   "absent signals must reproduce current placement exactly").
2. **Execution sim** (discrete-event): pipeline of stages with service times
   from the cost model + edge models; replays synthetic workload traces;
   emits TTFT/TPOT/throughput curves per candidate topology. Covers
   degenerate conditions: straggler node, degrading link, cold-start after
   failure, mixed prompt lengths at concurrency.

**Calibration bar:** the execution sim must reproduce the measured ratios in
`docs/BENCHMARKS.md` (68 → 21 → 12-13 tok/s across 1/2/3-way splits on the
documented hardware; 10-25 tok/s at ~20 ms RTT, RPC-latency-dominated) from
the documented inputs, within tolerance. If it cannot, the cost model is
wrong and gets fixed before any production behavior depends on it.

**As-built (PR #1454):** the execution layer (`skippy-topology-sim::execution`)
models two regimes — **serial** decode (single stream: every token traverses
every stage and returns; TPOT = Σ stages + Σ hops — this is the regime the
BENCHMARKS.md anchors measured) and **pipelined** (lanes > 1: bounded by the
slowest stage+egress pair). Stage service time is streamed-bytes/bandwidth
with `active_weight_fraction` capturing MoE active-expert bytes; two
calibration knobs record what the pure model cannot see: per-stage software
overhead and per-hop RPC overhead (the "per-token RPC latency" BENCHMARKS.md
names as dominant). Calibration scenario `benchmarks_anchor_pair.toml` +
tests reproduce all three anchors within ~10% (tolerance ±15%). Coefficients
are recorded in the scenario with their derivation so real measurements
(passive edge observations in particular) can tighten them.

## Scenario corpus: realistic hardware, links, and backends

The simulator is only as honest as its inputs. The corpus spans the hardware
and transports mesh actually runs on, from consumer laptops to datacenter
nodes, with per-backend variation. Numbers below are *prior* starting points
(spec-class), to be replaced by `mesh-llm-gpu-bench` measurements as they are
collected — the corpus format records `source = "spec" | "measured"` per
field and the calibration phase upgrades specs to measurements.

### Node tiers

| Tier | Example | VRAM | Sustained mem bw (prior) | Sustained fp16 (prior) | Backends |
|---|---|---|---|---|---|
| Consumer laptop, CPU | 16-32 GB LPDDR5 | shared | 60-100 GB/s | 0.1-0.5 TFLOPS | CPU (AVX2/NEON) |
| Consumer laptop, iGPU | 16-64 GB unified | shared | 100-546 GB/s | 1-34 TFLOPS | Metal (Apple), Vulkan |
| Consumer desktop GPU | RTX 3060/4070, 8-12 GB | 8-12 GB | 360-504 GB/s | 15-30 TFLOPS | CUDA, Vulkan |
| Prosumer GPU | RTX 3090/4090, 24 GB | 24 GB | 936-1008 GB/s | 40-83 TFLOPS | CUDA |
| Prosumer multi-GPU | 2-4× above | 48-96 GB | per-GPU, NVLink absent | per-GPU | CUDA |
| Datacenter GPU | A100 80 GB | 80 GB | 1.9-2.0 TB/s | 78-312 TFLOPS | CUDA |
| Datacenter GPU | H100 80 GB | 80 GB | 3.3-3.4 TB/s | 197-990 TFLOPS | CUDA |
| Datacenter GPU | MI300X 192 GB | 192 GB | 5.3 TB/s | 163-1307 TFLOPS | ROCm |

Backend matters independently of the chip: the same GPU on CUDA vs Vulkan can
differ materially in sustained throughput, and llama.cpp quant kernels vary by
backend and quant (Q4_K_M, Q8_0, f16). Corpus entries therefore carry
`(hardware, backend, quant)` triples, not hardware alone.

### Link tiers

| Tier | Example | Typical RTT | Large-frame throughput (prior) |
|---|---|---|---|
| Loopback / same host | localhost | 0.05-0.2 ms | 20-60 GB/s |
| Direct cable | Thunderbolt/2.5-10GbE point-to-point | 0.1-0.5 ms | 1-20 GB/s |
| LAN wired | 1-10 GbE switched | 0.3-2 ms | 100 MB/s-1 GB/s |
| LAN Wi-Fi | Wi-Fi 5/6/6e | 2-10 ms | 10-60 MB/s |
| Metro WAN | same-city fiber | 5-15 ms | 10-100 MB/s |
| Continental WAN | Sydney↔QLD class | 15-40 ms | 5-50 MB/s |
| Intercontinental | US↔EU/US↔APAC | 60-250 ms | 1-20 MB/s |

Asymmetry is first-class: edges are directed, so scenarios include pairs
where A→B and B→A differ (asymmetric Wi-Fi, rate-limited cloud egress).

### Corpus scenarios (initial set)

Landed in `crates/skippy-topology-sim/scenarios/` as of PR #1454:
`heterogeneous_pair.toml` (2), `straggler_triplet.toml` (3),
`cross_continent_chain.toml` (4). Remaining from the initial set —
homogeneous pair (1), mixed-quant fleet (5), load/staleness sweep (6),
failure cold-start (7) — are open corpus work tracked in issue #1455.

1. **Homogeneous pair** (2× M4 Max, Thunderbolt): baseline sanity. *(pending)*
2. **Heterogeneous pair** (M4 Max + Mac mini, Wi-Fi): reproduces the
   `docs/BENCHMARKS.md` 68 → 21 tok/s anchor. **landed**
3. **Straggler triplet** (A100 + 4090 + laptop-CPU): the laptop must get
   few layers or be excluded; tests performance-aware span assignment
   against capacity-only. **landed**
4. **Cross-continent chain** (3 nodes, 60-150 ms edges): tests that
   edge-aware ordering minimizes high-latency hops and rejects infeasible
   TPOT targets rather than accepting them. **landed**
5. **Mixed-quant fleet** (same model, Q4/Q8/f16 on different nodes):
   activation wire dtype interacts with per-node bytes/layer. *(pending)*
6. **Load and staleness sweep** (one node busy/stale): confidence decay
   must fall back toward capacity-only placement. *(pending)*
7. **Failure cold-start** (node rejoins empty): migration/dwell-time
   accounting under phase 5 policies. *(pending)*

### Where the data comes from

- **Priors:** vendor spec sheets (bandwidth classes, not marketing peaks),
  recorded in the corpus with `source = "spec"`.
- **Measurements:** `mesh-llm-gpu-bench` runs on real fleet nodes over time
  (`source = "measured"`, timestamped, superseding specs), plus edge probes
  already producing `StageEdgeSignal` data.
- **Anchors:** the measured results in `docs/BENCHMARKS.md` are regression
  anchors the execution sim must reproduce within tolerance.

The corpus lives in-repo as scenario TOML files so CI, the planner tests, and
the execution sim all consume the same data.

## Changing network conditions

Bandwidth is not static: Wi-Fi fades, links get congested, VPNs re-route.
The planner's job under drift is **detect → re-estimate → decide**, with
anti-churn protection so a transient dip does not cause a topology stampede.

**What exists today (as of PR #1454):**
- Node perf metrics (mem bw, compute) and per-participant RTT are part of the
  split-participant signature (`split_participant_signature`), so a measured
  change re-triggers planning automatically.
- **Edge bandwidth is measured passively**: every artifact transfer (either
  direction) records bytes/second on that peer link (`LargeFrameObservation`),
  age-gated to 30 minutes, conservatively min-merged into plan edges. As
  conditions change, the next transfer re-measures the link — drift detection
  rides the traffic the mesh already generates. Active probing (synthetic
  frames between idle stage peers) remains future work.
- The planner edge type is directed, and simulator scenarios can supply truly
  asymmetric A→B/B→A values. Production does not yet measure remote pair
  directions independently: `participant_edges` synthesizes both directions
  with the same conservative max RTT and min bandwidth from coordinator-to-peer
  observations.
- The best-seen RTT remains a minimum, but its observation count and first/
  latest sample ages are retained. Remote performance-aware placement waits
  for two samples spanning the 5-second direct-path recheck; a lone 200 ms-old
  sample falls back to capacity-only placement. Distribution tails are not
  planner inputs.
- Embedded stages record steady-decode compute time after warmup, normalize it
  per loaded layer, gossip the bounded timing hint, and use it as a measured
  floor on analytical stage service time. The participant signature includes
  the observation so a fresh planning round cannot silently reuse a stale
  claim.
- `MESH_TOPOLOGY_PERF_AWARE=0/false/off/no` is an operator kill-switch that
  strips perf signals + edges and reproduces capacity-only placement exactly
  (checked per planning attempt, no restart needed).

**The three detection windows and their design:**

| Window | Signal | Response |
|---|---|---|
| Per-token (immediate) | In-flight decode misses TPOT target | Runtime concern, not planner's — no topology change; the plan already priced this link into its estimate |
| Per-probe (minutes) | Re-measured edge/node metrics shift | Signature change flows into the coordinator claim's `participant_set_hash`, invalidating the current generation's identity and forcing a fresh planning round. **Today the fresh round replaces the incumbent unconditionally** — the minimum-improvement threshold below is phase-5 design, not yet implemented |
| Per-epoch (hours/days) | Slow drift, new nodes, day/night load | Same replan trigger; hysteresis (phase 5) dampens noise |

**Why not react instantly:** re-sharding a live mesh costs KV migration +
pipeline stall. A Wi-Fi blip that halves bandwidth for 20 seconds should not
evict a topology that took minutes to load. The planner therefore treats
edge measurements as *estimates with age and confidence*, not instantaneous
truth — the same design as metric-age decay in the input contract.

**What phase 5 adds:** adaptive replanning with explicit hysteresis and
migration budgets — re-estimating when an edge's sustained (not transient)
bandwidth drops materially below what the plan assumed, and migrating only
when the modeled improvement exceeds the migration cost. Until then the
system degrades to today's behavior: the plan made at startup holds until
membership or a signature change forces a re-plan.

**Open question (phase 5):** how to age/degrade edge bandwidth measurements
between probes. Candidates are an EWMA of sustained transfer samples or a
conservative age-based floor. The execution sim's calibration against
BENCHMARKS.md anchors is the testbed for choosing between these; planner-side
distribution-tail estimation is intentionally out of scope.

## Phased rollout

| Phase | Deliverable | Gate | Status |
|---|---|---|---|
| 0 | Thread gossiped perf metrics through `SplitTopologyPlanInput → TopologyNode` | no behavior change (signals recorded, unused) | **Done** (PR #1454) — metrics flowed through and joined the replan signature |
| 1 | Cost model + merged scoring in `skippy-coordinator`; absent-signal fallback = exact current behavior | placement-parity tests vs old planner on signal-less inputs | **Done** (PR #1454) — `perf_balanced_spans` DP + parity tests |
| 2 | Placement sim in CI; scenario corpus incl. BENCHMARKS.md anchors | property tests green; parity suite green | **Done** (PR #1454) — `skippy-topology-sim` + 3 corpus scenarios |
| 3 | Passive edge measurement; execution sim calibration; observed stage-timing feedback; settle-time RTT confidence | calibration tolerance met; uncorroborated remote signals fall back safely | **Done** (PR #1454) — passive edge bandwidth from real artifact transfers (both directions, age-gated 30 min, conservative min-merge, replan signature); execution sim + BENCHMARKS.md calibration tests (±15% tolerance, currently within ~10% on all three anchors); live steady-decode µs/layer feeds span DP and serial TPOT as a measured floor; min RTT carries sample count + first/latest age and requires corroboration across the settle window. Active synthetic probing remains optional future corpus work, not a phase-4 prerequisite |
| 4 | Performance-aware placement live (default on) | A/B on staging meshes vs capacity-only | **Code path default-on in PR #1454; reference-hardware A/B gate pending** |
| 5 | Adaptive replanning with hysteresis + migration budgets | dwell-time threshold; no churn under synthetic perturbations | Planned |

Phase 1's fallback property is the safety story: with no signals *and no
edge data anywhere in the fleet*, the merged planner is bit-identical to
today's (per-subset scope above). Each phase is independently mergeable.

## Alternatives considered

- **Third planner, purpose-built.** Rejected: duplicates feasibility logic
  that already exists and is tested in two places; the merge is smaller than
  a rewrite and keeps the policy crate's validation as the legality
  authority.
- **Pure simulation-first (build sim, decide later).** Rejected: the sim
  needs the cost model, the cost model needs the input plumbing; phase 0/1
  deliver both and the sim then has something honest to simulate.
- **vLLM/SGLang-style profile-guided autotuning.** Deferred: calibration
  from observed stage timings (phase 5) gets most of the value without a
  profile store and offline tuning loop.

## Risks

- **Cost model error → worse placements.** Mitigated by the fallback
  property, calibration gates, and phase 4 A/B before default-on.
- **Stale/lying gossip.** Mitigated by age-gated stage timing, RTT-floor
  corroboration, per-candidate absent-signal fallback, and pessimistic
  unknown-edge defaults. Static gpu-bench claims remain soft hints.
- **Search blowup on large fleets.** Node subsets are already bounded;
  DP span assignment is `O(layers² × nodes)` per candidate. Automatic edge
  ordering is not wired in today; once adopted, the existing policy planner's
  exhaustive ≤8-stage / greedy >8-stage split bounds that additional search.
