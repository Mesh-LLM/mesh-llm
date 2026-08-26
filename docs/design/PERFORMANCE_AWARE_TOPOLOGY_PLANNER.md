# Performance-Aware Topology Planner and Placement Simulator

## Status: Design proposal

- Date: 2026-08-26
- Owner: TBD
- Origin: skippy-topology channel discussion (2026-08-26); requested by James.

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

- No change to the stage runtime, wire protocol, or llama.cpp integration.
  This work re-plans *placement*; execution is untouched.
- No speculative/exotic parallelism (tensor/pipeline-parallel hybrid graphs).
  Contiguous layer pipelines only, as today.
- No live adaptive replanning in the first phases (see rollout — hysteresis
  and migration come last, after the model is calibrated).

## Current state (verified at `9feef0c1`)

| Capability | Where | Used by automatic placement? |
|---|---|---|
| Capacity fitting (exact per-layer weights, KV/token, recurrent/lane, 100/85 KV compute reserve, 10% runtime headroom) | `skippy-coordinator/src/topology.rs` | Yes |
| Candidate search (context ↓, node count ↑, lanes ↓, all subsets), stage-0 binding, 33 ms decode TPOT target, 64K shared-context floor | `skippy-coordinator/src/topology.rs`, `mesh-llm-host-runtime/src/runtime/split_planning.rs` | Yes |
| Latency estimate `stage_count × max RTT` | `estimate_decode_network_ms_per_token` | Yes (latency-aware ordering only) |
| GPU benchmarking (mem bw, fp16/fp32 TFLOPS) | `mesh-llm-gpu-bench`, `mesh-llm-system/src/benchmark.rs` | Metrics gossiped, **dropped before planner** |
| Directed edge signals (RTT + large-frame bandwidth per edge, prediction-return support) | `skippy-topology/src/edge_order.rs` (exhaustive ordering ≤ 8 stages, greedy beyond) | **No** |
| Model-family cut rules, state affinity, shared-KV cut bans, wire dtype, sidebands | `skippy-topology/src/planning.rs`, `validation.rs` | **No** (explicit-split validation only) |

The two planners are complementary halves of one optimizer. The design below
merges them rather than adding a third.

## Input contract

### 1. Node performance (per node)

| Field | Source | Notes |
|---|---|---|
| `usable_vram_bytes` | existing `TopologyNode` | after 10% runtime headroom |
| `sustained_mem_bw_gbps` | gossip (`gpu_mem_bandwidth_gbps`), gpu-bench | measured, not spec |
| `sustained_compute_tflops` | gossip (`gpu_compute_tflops_fp16` preferred) | fallback signal only; decode is usually memory-bound |
| `host_ram_bytes` | node status | workspace/scratch headroom |
| `load_ewma`, `metric_age_ms` | new probe | stale signals decay to neutral |

### 2. Directed link performance (per ordered node pair)

| Field | Source |
|---|---|
| `p50_latency_ms`, `p95_latency_ms` | extended `StageEdgeSignal` |
| `large_frame_bytes_per_sec` | existing `StageEdgeSignal` field |
| `jitter_ms`, `sample_age_ms` | new probe |
| `direct_prediction_return_supported` | existing `StageEdgeSignal` field |

Unknown edges get the existing pessimistic default (`UNKNOWN_EDGE_RTT_MS`).

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
                         flops(L_i) / sustained_compute(n(i)) )              # compute-bound regimes
                   + kv_touch_ms(L_i)                                        # resident KV scan per token
edge_time_ms(i)    = act_bytes(L_i) / large_frame_bw(e(i)) + p50_latency(e(i))
pipeline_tpot_ms   = max_i ( stage_time_ms(i) + edge_time_ms(i) )            # steady-state decode
prefill_ms         = Σ_i ( stage_time_ms(i) + edge_time_ms(i) )              # sequential fill
```

Pipeline TPOT is the max, not the sum: stages process consecutive tokens
concurrently in steady state. `pipeline_tpot` replaces
`estimated_decode_network_ms_per_token`; the 33 ms target check carries over
unchanged. Confidence weights degrade each term toward today's behavior as
`metric_age_ms` grows, so absent signals reproduce current placement exactly
— the safe fallback.

## Search algorithm

Preserve the existing candidate enumeration (it is correct and tested); add
performance to scoring and ordering:

1. **Hard feasibility filter** (unchanged): memory fit with reserves,
   family cut legality, stage-0 binding, sidebands, artifact access.
2. **Enumerate candidates** (unchanged): context lengths highest→lowest,
   node counts fewest→most, lanes highest→lowest, node subsets.
3. **Order nodes** on the directed link graph (adopt
   `order_pipeline_nodes`: exhaustive ≤ 8 stages, greedy beyond) instead of
   VRAM-descending order.
4. **Span assignment**: replace greedy largest-fit with DP over contiguous
   layer boundaries that minimizes `pipeline_tpot_ms` subject to per-node
   memory ceilings. `O(layers × nodes)` per candidate — tractable at current
   scales (≤ ~100 layers, ≤ ~8 nodes).
5. **Score lexicographically**: correctness → SLO met → objective-specific
   performance (TPOT or throughput) → context/lane utility → confidence and
   headroom → deterministic tie-breaks (existing `latency_candidate_ordering`
   shape).

## Simulator

Two layers, sharing one scenario format (`toml`):

```toml
[nodes.m4max]
vram_gb = 48
mem_bw_gbps = 546          # measured
compute_tflops_fp16 = 34

[links."m4max->mini"]
p50_latency_ms = 2.1
large_frame_gbps = 31      # measured activation throughput

[model]
package = "GLM-4.7-Flash-Q4_K_M"
context = 65536

[workload]
objective = "interactive"
decode_tpot_target_ms = 33
```

1. **Placement sim** (deterministic, fast, in-crate): scenario → planner →
   assert chosen topology and score. Runs in CI as the planner's unit-test
   surface: property tests over synthetic grids ("10× slower link must move
   the boundary", "half-bandwidth node must receive fewer layers",
   "absent signals must reproduce current placement exactly").
2. **Execution sim** (discrete-event): pipeline of stages with service times
   from the cost model + edge models; replays synthetic workload traces;
   emits TTFT/TPOT/throughput curves per candidate topology. Covers
   degenerate conditions: straggler node, jittery link, cold-start after
   failure, mixed prompt lengths at concurrency.

**Calibration bar:** the execution sim must reproduce the measured ratios in
`docs/BENCHMARKS.md` (68 → 21 → 12-13 tok/s across 1/2/3-way splits on the
documented hardware; 10-25 tok/s at ~20 ms RTT, RPC-latency-dominated) from
the documented inputs, within tolerance. If it cannot, the cost model is
wrong and gets fixed before any production behavior depends on it.

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

| Tier | Example | p50 latency | Large-frame throughput (prior) |
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

1. **Homogeneous pair** (2× M4 Max, Thunderbolt): baseline sanity.
2. **Heterogeneous pair** (M4 Max + Mac mini, Wi-Fi): reproduces the
   `docs/BENCHMARKS.md` 68 → 21 tok/s anchor.
3. **Straggler triplet** (A100 + 4090 + laptop-CPU): the laptop must get
   few layers or be excluded; tests performance-aware span assignment
   against capacity-only.
4. **Cross-continent chain** (3 nodes, 60-150 ms edges): tests that
   edge-aware ordering minimizes high-latency hops and rejects infeasible
   TPOT targets rather than accepting them.
5. **Mixed-quant fleet** (same model, Q4/Q8/f16 on different nodes):
   activation wire dtype interacts with per-node bytes/layer.
6. **Load and staleness sweep** (one node busy/stale): confidence decay
   must fall back toward capacity-only placement.
7. **Failure cold-start** (node rejoins empty): migration/dwell-time
   accounting under phase 5 policies.

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

## Phased rollout

| Phase | Deliverable | Gate |
|---|---|---|
| 0 | Thread gossiped perf metrics through `SplitTopologyPlanInput → TopologyNode`; instrumentation of observed stage timings | no behavior change (signals recorded, unused) |
| 1 | Cost model + merged scoring in `skippy-coordinator`; absent-signal fallback = exact current behavior | placement-parity tests vs old planner on signal-less inputs |
| 2 | Placement sim in CI; scenario corpus incl. BENCHMARKS.md anchors | property tests green; parity suite green |
| 3 | Execution sim validated against measured data | calibration tolerance met |
| 4 | Performance-aware placement live (default on) | A/B on staging meshes vs capacity-only |
| 5 | Adaptive replanning with hysteresis + migration budgets | dwell-time threshold; no churn under synthetic jitter |

Phase 1's fallback property is the safety story: with no signals, the merged
planner is bit-identical to today's. Each phase is independently mergeable.

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
- **Stale/lying gossip.** Mitigated by metric age decay toward neutral and
  pessimistic unknown-edge defaults.
- **Search blowup on large fleets.** Node subsets are already bounded;
  DP span assignment is `O(layers × nodes)` per candidate. Beyond ~8 nodes
  the greedy edge ordering path applies as today.
