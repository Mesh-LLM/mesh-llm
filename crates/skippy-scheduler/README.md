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
cargo bench -p skippy-scheduler --features scheduler-lab --bench scheduler_lab
```

The lab drives the production `Scheduler` directly without loading a model or
native runtime. Synthetic request arrivals, prompt/decode lengths, prefix hits,
and a configurable virtual prefill/decode cost model produce queue wait, TTFT,
inter-token gap, throughput, batch-size, and token-occupancy measurements. The
reported times are modeled scheduler outcomes rather than hardware inference
claims; use them to compare policy changes under an identical cost model.

## License

Licensed under Apache-2.0.
