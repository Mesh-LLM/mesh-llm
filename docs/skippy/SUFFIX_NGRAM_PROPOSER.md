# Suffix N-gram Proposer

## Purpose

The suffix N-gram proposer is a request-local prompt-lookup source for staged
speculative decoding. It retrieves the continuation that followed the longest
exact suffix seen earlier in the prompt or target-committed output. It can run
as the complete proposer (`ngram-suffix`) or extend a native-MTP prefix
(`native-mtp+ngram-suffix`).

The proposer is designed for coding-agent and tool-loop traffic with large
verbatim regions: file re-emission with a localized edit, repeated structured
configuration, and continuations grounded in earlier tool output. It is not a
semantic model and makes no proposal when the request contains no sufficiently
long exact match.

The target model remains authoritative. Suffix tokens are candidates only;
verification commits the matching prefix and replaces the first mismatch with
the target token. The implementation refinements in this document were made
after maintainer review.[^review-changes]

## Standalone and Hybrid Modes

All three N-gram implementations are first-class standalone proposers:

| Strategy | Proposer | Requires native MTP | On lookup miss |
|---|---|---:|---|
| `ngram-cache` | Request-local llama.cpp cache | No | Decode one target token |
| `ngram-suffix` | Request-local exact-suffix index | No | Decode one target token |

Cache and suffix do not silently fall back to Simple. A configured proposer is
the sole N-gram source for that strategy, which keeps activation, acceptance,
and performance attribution honest. “Standalone” here means independent of
MTP in the staged decode loop. Single-stage local-runtime N-gram verification
is a separate implementation surface and is not added by this design.

In standalone suffix mode, the query is the target-committed history and the
retrieved continuation goes directly to target verification:

```text
committed history ──► exact-suffix lookup ──► suffix candidate
        ▲                                         │
        └──────── target verification ◄───────────┘
```

Native MTP and suffix retrieval have complementary roles:

```text
target-committed history ──► incremental exact-seed index
          │                              │
          ├──► native MTP prefix ────────┤ read-only lookup query
          │                              ▼
          │                       copied suffix tail
          │                              │
          └──────────────► [MTP prefix | suffix tail]
                                      │
                                      ▼
                              target verification
                                      │
                                      ▼
                            newly committed history
```

1. Native MTP predicts the first uncertain or changed token from model state.
2. The suffix proposer treats that MTP output as a read-only continuation of
   target-committed history.
3. If the extended suffix matches an earlier request position, the proposer
   copies the tokens that followed that position.
4. The target verifies the MTP prefix and suffix tail as one candidate while
   preserving their boundary for acceptance telemetry.

The MTP continuation never mutates suffix history. Only prompt tokens and
target-committed generation tokens enter the index.

## Lookup Algorithm

Configuration supplies:

- `ngram_min`: minimum exact suffix match; at least 3;
- `ngram_max`: maximum backward comparison window; at most 64;
- `ngram_max_proposal_tokens`: independent output-token budget.

The index seed length is `min(ngram_min, 8)`. Each target-committed token that
completes a seed adds its end position to an `AHashMap<SeedKey, Vec<u32>>`.
`SeedKey` stores the exact seed tokens rather than a precomputed fingerprint,
so different token sequences never share an entry merely because their
fingerprints collide.

For each proposal:

1. Incrementally append newly committed tokens and their completed seeds.
2. Build the current seed from committed history plus the optional read-only
   MTP continuation.
3. Visit every earlier position indexed by that exact seed.
4. Compare tokens backwards, up to `ngram_max`, and retain the longest match.
   Equal-length matches prefer the most recent position.
5. Stay silent when the longest match is shorter than `ngram_min`.
6. Copy the tokens following the selected position. Draft length is bounded by
   the caller budget, `ngram_max_proposal_tokens`, twice the match length, and
   the historical continuation available.

Index construction is O(committed tokens), normal synchronization is O(newly
committed tokens), and lookup is O(candidate positions multiplied by the
bounded comparison window). A divergence or request reset rebuilds the index.

## Why AHashMap

The lookup operates on small fixed-size token keys and is called on the decode
hot path. `AHashMap` provides a fast randomized map on the supported platforms
without relying on a hand-written fingerprint or a SIMD-only implementation.
Exact `SeedKey` equality remains the correctness boundary; target verification
remains the output-correctness boundary.

The map choice should be revisited only if profiling shows hashing rather than
candidate comparison or model execution is material. Telemetry exposes lookup
and synchronization time for that purpose.

## Configuration

Select suffix as a standalone strategy when the target model has no MTP head or
when measuring prompt lookup independently:

```toml
[models.speculative]
strategy = "ngram-suffix"
ngram_min = 5
ngram_max = 32
ngram_max_proposal_tokens = 48
verify_window_min_tokens = 1
verify_window_max_tokens = 32
```

Select it as an MTP extension when native MTP is available:

```toml
[models.speculative]
strategy = "mtp"
ngram_proposer = "suffix"
ngram_min = 5
ngram_max = 32
ngram_max_proposal_tokens = 48
extension_max_tokens = 48
verify_window_min_tokens = 1
verify_window_max_tokens = 32
verify_window_pipeline_depth = 2
```

`ngram-cache` and `ngram-suffix` are valid standalone package
strategy types. A suffix package proposer must declare request-local history:

```json
{
  "generation": {
    "speculative_decoding": {
      "default": "ngram-suffix",
      "proposers": {
        "suffix": {
          "type": "ngram-suffix",
          "ngram_min": 5,
          "ngram_max": 32,
          "max_proposal_tokens": 48,
          "history_scope": "request"
        }
      },
      "strategies": {
        "ngram-suffix": {
          "type": "ngram-suffix",
          "proposer": "suffix"
        }
      }
    }
  }
}
```

The cache and suffix limits intentionally differ. Cache uses llama.cpp's
stateful lookup with a match window no larger than four tokens. Suffix may use
a much longer exact match. For both proposers, proposal output length is
independent from match-window length.

| Proposer | State | Match horizon | Candidate source | Supported role |
|---|---|---:|---|---|
| Simple | None | Configured minimum | Accepted token history | Standalone or MTP sidecar |
| Cache | Request-local llama.cpp cache | Up to 4 tokens | Recent cache match | Standalone or MTP sidecar |
| Suffix | Request-local exact-seed index | Up to 64 tokens | Longest earlier exact suffix | Standalone or MTP sidecar |

## Why the Review Added These Changes

The initial implementation established the important idea: long prompt overlap
can provide a much wider speculative horizon than a four-token cache lookup.
The follow-up deliberately strengthens that idea without replacing it:

- Exact `SeedKey` map entries replace a hand-written FNV fingerprint so a hash
  collision cannot merge unrelated token sequences before exact comparison.
- `AHashMap` keeps those small exact-key lookups inexpensive on the decode hot
  path without committing the design to a platform-specific SIMD hasher.
- Incremental synchronization indexes only newly committed tokens; copying and
  rebuilding the full history on every decode step would erase the proposer’s
  throughput benefit.
- Retaining every seed position preserves an older long match when many recent
  short matches share its ending. Candidate-count and lookup-time telemetry
  make the resulting repetitive-input cost visible.
- Match length and proposal budget are configured separately because evidence
  quality and useful pipeline width are different controls.
- Source labels and per-proposer counters prevent Simple, Cache, Suffix, and MTP
  work from being combined into an attractive but uninterpretable TPS number.
- Exclusive standalone selection prevents a Cache or Suffix miss from silently
  becoming a Simple proposal.
- Resolver, package-preflight, CLI, and runtime tests cover the same typed plan
  so configuration cannot claim suffix is active while the decode loop ignores
  it.

These changes are review hardening around Daniel’s original proposer design,
not a change in authorship or intent.[^review-changes]

## Telemetry

Request summaries and response timings identify the configured proposer and
report:

- proposal attempts, hits, and proposed tokens;
- suffix match-length sum and maximum;
- candidates examined;
- incrementally appended tokens and rebuilds;
- synchronization and lookup microseconds;
- native-MTP tokens, N-gram tail tokens, and accepted counts;
- N-gram tail rejections and sidecar backoff.

Prompt text, token IDs, candidate tokens, filesystem paths, endpoints, and raw
node identities are not exported.

## Correctness Invariants

- The index is scoped to one request.
- Only target-committed history mutates the index.
- An MTP continuation is read-only.
- A divergent history clears and rebuilds the index.
- Candidate selection compares exact tokens; hashes do not establish matches.
- Verification, not retrieval, determines committed output.
- A suffix rejection must not count as a native-MTP rejection when the MTP
  prefix was accepted.

## Expected Workloads and Failure Modes

Long exact repeats can provide enough future tokens to keep a staged
verification pipeline occupied. Novel prose or novel code normally produces no
suffix proposal. Short or common matches can select the wrong continuation;
`ngram_min`, sidecar backoff, bounded verification windows, and stale-work
telemetry control that cost.

Retaining all exact-seed positions avoids silently dropping an older long
prompt match. Highly repetitive inputs may therefore increase candidate
comparison work, which is visible through candidate and lookup-time telemetry.

## Benchmark Contract

A reportable MTP-composite comparison requires:

- an MTP-capable model on at least a two-stage split;
- MTP-only, simple, cache, and suffix arms on the same topology;
- release builds, identical sampling and verification settings, and sequential
  arms on shared hardware;
- long coding/edit workloads plus a low-overlap control;
- warmups, per-sample artifacts, wall throughput, server throughput,
  acceptance, output hashes, and finish reasons;
- pipeline occupancy, simultaneous stage compute, downstream wait, and stale
  work when claiming latency hiding;
- explicit activation evidence showing the suffix proposer produced tokens.

The expected outcome is a hypothesis until measured. Acceptance rate alone is
not the success criterion: a lower-acceptance arm may still win throughput when
its longer horizon keeps distributed stages busy.

A standalone comparison should use `disabled`, `ngram-cache`, and
`ngram-suffix` arms on the same split. The benchmark runner uses generic
draft acceptance for standalone arms and N-gram tail acceptance for MTP hybrid
arms; activation requires the configured proposer’s own hit/source telemetry.

### Standalone GLM-4.7 split result (2026-07-21)

We ran the first standalone matrix on
`meshllm/GLM-4.7-Flash-MTP-GGUF:Q4_K_M` with a two-stage 47/1-layer split,
eight lanes, and no injected inter-stage latency. Each arm received the same
deterministic coding prompt: re-emit a source file with one function renamed,
then generate 384 tokens. Results below are medians from five measured requests
after two warmups. Prefix caching was explicitly disabled for every arm so a
cross-request cache restore could not change the comparison.

| Standalone arm | Server decode tok/s | End-to-end tok/s | Server uplift vs target | Acceptance | Proposed / accepted tokens per request |
|---|---:|---:|---:|---:|---:|
| Target only | 22.60 | 19.87 | — | — | 0 / 0 |
| Simple | 107.22 | 64.38 | +374.5% | 53.2% | 694 / 369 |
| Cache | 77.96 | 52.94 | +245.0% | 67.1% | 523 / 351 |
| Suffix | **119.24** | **68.49** | **+427.6%** | **84.9%** | 425 / 361 |

Suffix was the fastest arm: 11.2% higher server decode throughput and 6.4%
higher end-to-end throughput than Simple. Its lookup cost was 63 microseconds
across all five measured requests, about 12.6 microseconds per request, so
hashing was not a material cost in this workload.

The Simple-versus-Cache result is also important. Simple had lower acceptance
(53.2% versus 67.1%) yet delivered 37.5% more server throughput because it
proposed and accepted more tokens in absolute terms. Acceptance percentage is
therefore not enough to judge a proposer: proposal width, accepted tokens per
verification window, and resulting pipeline occupancy matter too.

This is a mechanism microbenchmark, not yet a production or quality result:

- It measured standalone N-gram strategies, not MTP plus N-gram composition.
- Each arm was deterministic by itself, but output hashes differed at the one
  novel edit line before reconverging on the copied body. Exact greedy-trajectory
  equivalence remains unresolved, so the throughput numbers must not be read as
  proof of equivalent code quality.
- With prefix caching enabled, a second standalone Simple request returned 502
  after a `chain_restore_hit`. Disabling the cache made sequential requests
  reliable. This points to speculative checkpoint state interacting with
  cross-request KV restore and must be fixed before a cache-enabled rerun.
- The previous cache-enabled target-only baseline was 22.93 server tok/s and
  19.68 end-to-end tok/s. The new target-only row differs by -1.5% and +1.0%
  respectively, which suggests the underlying baseline was stable.

Two small observability fixes were made while obtaining these results. An
explicit `prefix_cache.enabled = false` now survives model-family defaults, and
standalone response timings now report generic proposed, accepted, rejected,
and acceptance totals. These fixes make the arms comparable and do not change
the suffix lookup algorithm.

### MTP extension GLM-4.7 split result (2026-07-21)

We then repeated the same workload as an MTP extension matrix: MTP-only,
MTP+Simple, MTP+Cache, and MTP+Suffix. The model, 47/1-layer split, eight lanes,
zero injected delay, disabled prefix cache, prompt, 384-token output limit,
sampling, warmups, and five measured requests were unchanged. Arms ran
sequentially on the shared lab hardware, and response telemetry confirmed that
each configured extension contributed N-gram tokens. Every metrics run
completed with seven requests, no dropped events, and no export errors.

| MTP arm | Server decode tok/s | End-to-end tok/s | Server / end-to-end uplift vs MTP | Relevant acceptance | N-gram proposed / accepted per request |
|---|---:|---:|---:|---:|---:|
| MTP only | 30.53 | 25.84 | — | 53.9% native | — |
| MTP + Simple | 56.56 | 42.16 | +85.2% / +63.2% | 12.6% N-gram tail | 2,059 / 259 |
| MTP + Cache | 64.62 | 46.40 | +111.6% / +79.5% | 37.6% N-gram tail | 712 / 268 |
| MTP + Suffix | **78.05** | **53.24** | **+155.6% / +106.0%** | **35.9% N-gram tail** | **806 / 289** |

This is the clearest evidence for evaluating useful speculative width rather
than acceptance percentage alone. Suffix has lower N-gram-tail acceptance than
Cache (35.9% versus 37.6%) but accepts more tail tokens per request (289 versus
268). It is consequently 20.8% faster in server decode throughput and 14.7%
faster end to end than Cache.

The comparison against MTP-only is stronger still. Overall composite
acceptance falls from 53.9% for MTP-only to 37.0% for MTP+Suffix, while accepted
speculative tokens rise from 103 to 315 per request. Server throughput improves
by 155.6% and end-to-end throughput by 106.0%. A wider proposal can therefore
produce much more accepted work per verification sequence even when its
acceptance ratio is lower.

The hybrid matrix does not supersede the standalone result. On this
exact-copy-heavy prompt, standalone Suffix remains faster at 119.24 server
tok/s and 68.49 end-to-end tok/s. The hybrid result answers the narrower
question: extending an already active MTP proposer materially beats MTP-only,
and Suffix was the fastest MTP extension tested.

The same correctness limitation applies. MTP-only reproduced the target-only
output hash, while every extension arm diverged at the novel edit line before
returning to copied source. All arms were deterministic within-arm, but this
microbenchmark proves a throughput mechanism rather than equivalent code
quality or an exact greedy trajectory.

Finally, legacy `draft_n` response fields are inconsistent on some fragmented
direct-return hybrid replies; the Cache arm can report more accepted than
proposed tokens there. The table uses the explicit, internally consistent
`native_mtp_hybrid_*` counters. The known direct-return aggregation fix must be
present before legacy draft fields are used in a public report.

## Production Readiness

This branch establishes runtime capability, not production certification. The
following work should be completed before making suffix a package default:

### Model package generation

- Teach package generation—not only preflight—to emit the `ngram-suffix`
  proposer and standalone strategy shown above.
- For MTP-capable packages, optionally emit an `mtp-suffix` composite strategy
  with the native proposer as `primary` and suffix as `extender`.
- Keep `history_scope: "request"`, `3 <= ngram_min <= ngram_max <= 64`, and a
  separately bounded `max_proposal_tokens` in generated manifests.
- Do not change `generation.speculative_decoding.default` to suffix until that
  model/package has workload and topology-specific certification results.
- Add golden-manifest and package-preflight fixtures for generated standalone
  and composite plans.

### CLI and configuration

- The resolver and hidden advanced CLI accept `ngram-cache` and `ngram-suffix`;
  preserve tests proving both reach the typed stage-0
  plan without native MTP.
- Before general release, expose discoverable CLI help or a strategy-listing
  command instead of requiring operators to know hidden flags.
- Add a documented CLI example including `--speculative-ngram-max-proposal-tokens`
  and fail early when required bounds are absent.
- Surface proposer-specific constraints and descriptions in the exported
  config schema/UI: Cache max 4, Suffix min 3/max 64, request-local history,
  and proposal budget independent from match window.
- Add valid and invalid `config.toml` fixtures for defaults, per-model override,
  CLI precedence, and package-strategy precedence.

### Runtime and performance certification

- Expand the measured GLM 4.7 mechanism matrix beyond one repeated coding
  prompt: add longer multi-file edits, tool-loop traffic, a low-overlap control,
  and quality scoring while retaining both standalone and MTP-composite arms.
- Repeat at 0, 20, and 100 ms injected inter-stage RTT and report stage overlap,
  downstream wait, stale work, acceptance, wall TPS, and server TPS.
- Set memory and lookup-cost guardrails for adversarial repetitive contexts;
  retaining all exact candidates is correct but needs a production bound based
  on measured context sizes.
- Verify cancellation, context exhaustion, stop tokens, rollback/repair, and
  concurrent request isolation under suffix hits and misses.
- Run exact-output comparisons against the disabled baseline and soak repeated
  tool loops before enabling suffix automatically.
- Roll out opt-in first, retain per-request fallback to target-only decoding,
  and define regression thresholds that automatically disable the proposer.

[^review-changes]: Follow-up changes requested by [@i386](https://github.com/i386) replaced the original FNV fingerprint and eight-position buckets with exact `SeedKey` entries in `AHashMap`, made history synchronization incremental, separated proposal budget from match length, tightened suffix validation, corrected source attribution, and required reproducible benchmark evidence. These changes preserve the original prompt-lookup design while making its behavior measurable on long coding workloads.
