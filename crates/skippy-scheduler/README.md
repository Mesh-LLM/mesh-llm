# skippy-scheduler

`skippy-scheduler` provides iteration-level scheduling policy for concurrent
Skippy staged serving. It admits and preempts sequences, budgets prefill and
decode work against runtime memory, and produces shared iteration plans for
`skippy-server` to execute through the native Skippy ABI.

The crate owns scheduling policy only; model execution and transport remain in
the Skippy runtime and server crates.

## Scheduler lab

Run the deterministic scheduler-only workload suite in release mode:

```bash
just with-lld cargo bench -p skippy-scheduler --features scheduler-lab --bench scheduler_lab
```

The lab drives the production `Scheduler` directly without loading a model or
native runtime. Synthetic request arrivals, prompt/decode lengths, prefix hits,
and a configurable virtual prefill/decode cost model produce queue wait, TTFT,
inter-token gap, throughput, batch-size, and token-occupancy measurements. The
reported times are modeled scheduler outcomes rather than hardware inference
claims; use them to compare policy changes under an identical cost model.

The `radix-fcfs` and `radix-cache-aware` scenarios seed the production
`UnifiedRadixCache`, probe it without changing LRU recency, and run the same
request trace with and without cache-affinity ordering. Affinity is retained as
a per-stage vector so split stages can score their own hits independently.

## Cache-aware admission

Equal-priority waiting work is ranked by estimated prefill work saved minus
restore cost. Each scheduler turn adds an aging credit, so a cold request
eventually outranks newly arriving hot-prefix requests. Explicit request
priority remains the primary ordering key. The KV-enabled server path applies
the same policy to restore/prefill runtime operations and alternates those turns
with live decode work; native batches remain phase-homogeneous.

After priority, materialized-cache value, and aging, an ephemeral waiting-request
radix groups equal-value prompts by weighted DFS order. Heavier shared-prefix
subtrees run together, allowing the first cold request to materialize reusable
KV before its peers execute. The ordering is computed once for a drained queue,
so the group remains adjacent across the server's alternating cache-runtime and
decode turns. This request radix is scheduling-only and never changes unified
cache recency.

## License

Licensed under Apache-2.0.
